# app.py – Streamlit UI for DocMind (local RAG) + hybrid retrieval & re-ranking
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import streamlit as st
from document_loader import load_vectorstore
from llm import get_llm
from retrieval import build_retriever
from langchain_classic.chains import RetrievalQA   # keep your working import
from agent import  get_agent
# feedback_db py for data store in sql 
from feedback_db import init_db, save_feedback
init_db()
# Page config
st.set_page_config(page_title="DocMind", page_icon="📄", layout="centered")
st.title("📄 DocMind — Chat with your PDFs")
st.markdown("Ask questions about your document. All processing is local and private.")

# Sidebar
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    chunk_size = st.slider("Chunk size", 300, 1500, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 200, 10)
    k_chunks = st.slider("Retrieved chunks", 1, 10, 5)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
    use_rerank = st.checkbox("Use cross-encoder re-ranking", value=True)
    st.markdown("---")
    mode = st.radio("Mode", ["Simple RAG", "Agent (tools)"], index=0)
    st.caption("Runs fully locally with Ollama + Mistral 7B")

if mode == "Simple RAG":
    # ----- everything that was there before -----
    if uploaded_file is not None:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = "temp_uploaded.pdf"
    else:
        pdf_path = ""

    if not os.path.exists(pdf_path):
        st.error("No PDF found. Upload a document or place 'paper.pdf' in the project folder.")
        st.stop()

    @st.cache_resource
    def cached_load_vectorstore(path, cs, co):
        return load_vectorstore(path, chunk_size=cs, chunk_overlap=co)

    with st.spinner("Loading PDF and building vector store..."):
        vectorstore, num_pages, num_chunks, all_chunks = cached_load_vectorstore(pdf_path, chunk_size, chunk_overlap)

    if vectorstore is None:
        st.error("Failed to load the document.")
        st.stop()

    st.success(f"Loaded {num_pages} pages → {num_chunks} chunks")

else:
    # Agent mode – no PDF needed, set dummy values
    vectorstore = None
    all_chunks = []
    # optionally show a message
    st.info("Agent mode – no document loaded. You can ask anything.")

# LLM
llm = get_llm(model="mistral:7b", temperature=temperature)

# Hybrid retriever (all logic hidden inside build_retriever)
if mode == "Simple RAG":
    retriever = build_retriever(vectorstore, all_chunks, k=k_chunks, use_rerank=use_rerank)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
else:
    chain = get_agent(llm, verbose=False)   # False keeps Streamlit clean

# Chat
question = st.text_input("Ask a question about the document:", placeholder="e.g., What is the main contribution of this paper?")

if question:
    with st.spinner("Thinking..."):
        if mode == "Simple RAG":
            result = chain.invoke({"query": question})
            answer = result["result"]
            sources = result.get("source_documents", [])
        else:
            result = chain.invoke({"input": question})
            answer = result["output"]
            sources = []

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Sources")
    for i, doc in enumerate(sources):
        page = doc.metadata.get('page', '?')
        st.caption(f"Source {i+1} — page {page}")
        with st.expander("Show text"):
            st.text(doc.page_content[:500])