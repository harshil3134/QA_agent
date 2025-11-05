import os
from typing import List,TypedDict,Annotated
from operator import add
from dotenv import load_dotenv
from pydantic import BaseModel,Field

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, END


CHROMA_PERSIST_DIR='./chroma_db'
DOCUMENTS_DIR="./documents"

load_dotenv()



def initialize_vectorstore():
    """Initializing chromadb with document embeddings"""
    print("➡️ Initializing vector store")

    embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(CHROMA_PERSIST_DIR):
        print("➡️ loading existing vector store")
        vectorstore=Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            
        )
        return vectorstore
    
    print("➡️ Loading documents from directory")
    loader=DirectoryLoader(
        DOCUMENTS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"➡️ Loaded {len(documents)} documents")

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks=text_splitter.split_documents(documents)
    print(f"➡️ split into {len(chunks)} chunks")


    #creating vector store
    print("➡️ creating embeddings and storing in chromadb")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    print("➡️ vector store created and persisted")
    
    return vectorstore

vectorstore=initialize_vectorstore()

# initializing llm 
llm = ChatGroq(temperature=0,
                 model="meta-llama/llama-4-maverick-17b-128e-instruct")

#langgraph state
class AgentState(TypedDict):
    question:str
    need_retrival:bool
    retrieved_docs:List[str]
    answer:str
    reflection:dict
    messages:Annotated[List[str],add]
    
class Plan(BaseModel):
    """The plan for handling the user's question"""
    decision: str=Field(description="One of 'retrieve' or 'answer_only'")
    reasoning:str=Field(description="A brief explanation for the decision.")

#langraph nodes

def plan_node(state:AgentState)-> dict:
    """Planning Node analyze the question and decide if retrieval is needed"""
    print('💬 Plan node analyzing question')
    question=state['question']
    print(f"Question: {question}")

    llm_gpt = ChatGroq(temperature=0, model="openai/gpt-oss-20b")

    planner_prompt = ChatPromptTemplate([
        ("system", 
         "You are an expert router. Analyze the user's question and decide if it requires looking up external knowledge ('retrieve') "
         "from the knowledge base or can be answered from general knowledge ('answer_only'). "
         "If the question relates to climate change, renewable energy, sustainability, or sustainable tech, choose 'retrieve'. "
         "You must respond with a JSON object that adheres to the provided schema."),
        ("human", "{question}")
    ])
    structured_llm=planner_prompt | llm_gpt.with_structured_output(Plan)

    try:
        response=structured_llm.invoke({"question":question})
        need_retrival = response.decision == 'retrieve'
        print(f'➡️ Decision: {response}')
        return {"need_retrival": need_retrival}
    
    except Exception as e:
        print(f"Error in LLM plan:{e}")
        return {"need_retrival": True}


def retrieve_node(state:AgentState)->dict:
    """Get relevant documents from vector store"""
    print("➡️ Calling retirval node")

    if not state["need_retrival"]:
        print("⏭️  Skipping retrieval (not needed)")
        return {
            "retrieved_docs": [],
            "messages": ["RETRIEVE: Skipped - No retrieval needed"]
        }
    
    question = state["question"]
    
    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    docs = retriever.invoke(question)
    
    # Extract content
    retrieved_docs = [doc.page_content for doc in docs]

    print(f" Retrieved {len(retrieved_docs)} relevant documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        preview = doc[:150] + "..." if len(doc) > 150 else doc
        print(f"\n  Doc {i}: {preview}")
    
    return {
        "retrieved_docs": retrieved_docs,
        "messages": [f"RETRIEVE: Found {len(retrieved_docs)} relevant documents"]
    }

def answer_node(state: AgentState) -> AgentState:
    """
    Node 3: Answer - Generate answer using LLM and retrieved context
    """
    print("\n" + "="*60)
    print("� ANSWER NODE - Generating response...")
    print("="*60)
    
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]
    prompt=""
    if not state["need_retrival"]:
        # Simple response for greetings
        prompt_template = """You are a helpful, concise AI assistant. Answer the user's question using general knowledge. If you do not know the answer or the question requires external documents/citations, reply: \"I don't have enough information to answer that question.\" If the question is ambiguous, ask a brief clarifying question.
        Question: {question}
        Answer:"""
                    
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question"]
        )
      
    else:
            # Create context from retrieved documents
        context = "\n\n".join(retrieved_docs)
            
            # Create prompt
        prompt_template = """You are a helpful AI assistant. Answer the question based on the context provided below. 
        If the answer cannot be found in the context, say "I don't have enough information to answer that question."

        Context:
        {context}

        Question: {question}

        Answer:"""
    
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    
    # Generate answer
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    answer = response.content
    
    print(f"Answer: {answer}")
    
    state["answer"] = answer
    state["messages"].append("ANSWER: Generated using LLM and retrieved context")
    
    return state


def reflect_node(state: AgentState) -> AgentState:
    """
     Reflect - Evaluate answer for relevance and completeness
    """
    print("\n" + "="*60)
    print("➡️ REFLECT NODE - Evaluating answer quality...")
    print("="*60)
    
    question = state["question"]
    answer = state["answer"]
    retrieved_docs = state["retrieved_docs"]
    
    # Use LLM to evaluate the answer
    reflection_prompt = f"""Evaluate if the following answer is relevant and complete for the given question.

Question: {question}

Answer: {answer}

Evaluation criteria:
1. Is the answer relevant to the question?
2. Is the answer complete and informative?
3. Does the answer make sense?

Respond with:
- Relevance: relevant/not_relevant
- Completeness: complete/incomplete/partial
- Overall: good/needs_improvement

Format: Relevance: X | Completeness: Y | Overall: Z"""
    
    reflection_response = llm.invoke(reflection_prompt)
    reflection_text = reflection_response.content
    
    print(f"Reflection: {reflection_text}")
    
    # Parse reflection (simple parsing)
    reflection = {
        "evaluation": reflection_text,
        "num_docs_used": len(retrieved_docs),
        "answer_length": len(answer)
    }
    
    # Determine if answer is acceptable
    is_good = "good" in reflection_text.lower() and "relevant" in reflection_text.lower()
    reflection["is_acceptable"] = is_good
    
    print(f"\n✅ Answer is {'acceptable' if is_good else 'needs improvement'}")
    
    state["reflection"] = reflection
    state["messages"].append(f"REFLECT: Answer evaluated - {'Acceptable' if is_good else 'Needs improvement'}")
    
    return state

def build_graph():
    """Build the LangGraph workflow"""
    print("\n🔧  Building LangGraph workflow...")
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("reflect", reflect_node)
    
    # Define edges (flow)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", "reflect")
    workflow.add_edge("reflect", END)
    
    # Compile graph
    app = workflow.compile()
    
    print("✅ LangGraph workflow built successfully")
    return app



initial_state = {
        "question": "What are the benefits of renewable energy?",
        "need_retrival": False,
        "retrieved_docs": [],
        "answer": "",
        "reflection": {},
        "messages": []
}
app = build_graph()
final_state = app.invoke(initial_state)
    
#     # Print summary
# print("\n" + "="*30)
# print("SUMMARY")
# print("="*30)
# print(f"\n❓ Question: {final_state['question']}")
# print(f"\n💡 Answer: {final_state['answer']}")
# print(f"\n📄 Retrieved Docs: {len(final_state['retrieved_docs'])}")
# print(f"\n📄 Retrieved Docs: {(final_state['retrieved_docs'])}")