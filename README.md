
# QA Agent 

Minimal project demonstrating a Retrieval‑Augmented Generation (RAG) Q&A agent that uses local document embeddings (Chroma + HuggingFace embeddings) and a Groq LLM for planning/answering.

Quick start
-----------

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
# If you have a requirements.txt
python -m pip install -r requirements.txt

# Or install the common packages used by this project
python -m pip install python-dotenv langchain langchain-huggingface langchain-chroma langchain-text-splitters langchain-groq langchain-community chromadb sentence-transformers streamlit
```

3. Configure API keys and environment variables:

- Copy `.env.example` to `.env` (or create `.env`) and set your keys. Example variables used by this project:

```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_hf_api_key_here
```

4. Initialize or refresh the vector database (first run / after document changes):

```bash
python main.py
```

Running `main.py` will load documents from the `documents/` folder, split them into chunks, create embeddings, and persist them into the local Chroma database.

5. Run the Streamlit UI (optional):

```bash
source venv/bin/activate
streamlit run app.py
```

Make sure Streamlit is using the same Python interpreter (venv) where you installed the packages.
