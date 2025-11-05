import streamlit as st
from main import build_graph, vectorstore

# Initialize
if 'app' not in st.session_state:
    st.session_state.app = build_graph()

st.title("🤖 RAG AI Agent")
st.write("Ask questions about renewable energy, climate change, or sustainable tech!")

# User input
question = st.text_input("Your Question:", placeholder="What are the benefits of renewable energy?")

if st.button("Ask") and question:
    with st.spinner("Processing..."):
        initial_state = {
            "question": question,
            "need_retrieval": False,
            "retrieved_docs": [],
            "answer": "",
            "reflection": {},
            "messages": []
        }
        
        final_state = st.session_state.app.invoke(initial_state)
        
        # Display results
        st.subheader("Answer")
        st.write(final_state["answer"])
        
        with st.expander("See Details"):
            st.write(f"**Retrieval Used:** {'Yes' if final_state['need_retrieval'] else 'No'}")
            st.write(f"**Documents Retrieved:** {len(final_state['retrieved_docs'])}")
            
            if final_state['retrieved_docs']:
                st.write("**Retrieved Context:**")
                for i, doc in enumerate(final_state['retrieved_docs'], 1):
                    st.text_area(f"Doc {i}", doc, height=100)
            
            st.write("**Reflection:**")
            st.json(final_state['reflection'])
            
            st.write("**Workflow Steps:**")
            for msg in final_state['messages']:
                st.write(f"- {msg}")