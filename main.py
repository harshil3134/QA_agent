import os
from typing import List,TypedDict,Annotated
from operator import add
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_core.prompts import PromptTemplate


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

