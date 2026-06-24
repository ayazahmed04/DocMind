# app.py – Streamlit UI for DocMind (local RAG) + hybrid retrieval & re-ranking
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import streamlit as st
from document_loader import load_vectorstore
from llm import get_llm
from retrieval import build_retriever
from langchain_classic.chains import RetrievalQA   # keep your working import

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
    st.caption("Runs fully locally with Ollama + Mistral 7B")

# Load PDF
if uploaded_file is not None:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_path = "temp_uploaded.pdf"
else:
    pdf_path = "paper.pdf"

if not os.path.exists(pdf_path):
    st.error("No PDF found. Upload a document or place 'paper.pdf' in the project folder.")
    st.stop()

# Cached loading
@st.cache_resource
def cached_load_vectorstore(path, cs, co):
    return load_vectorstore(path, chunk_size=cs, chunk_overlap=co)

with st.spinner("Loading PDF and building vector store..."):
    vectorstore, num_pages, num_chunks, all_chunks = cached_load_vectorstore(pdf_path, chunk_size, chunk_overlap)

if vectorstore is None:
    st.error("Failed to load the document.")
    st.stop()

st.success(f"Loaded {num_pages} pages → {num_chunks} chunks")

# LLM
llm = get_llm(model="mistral:7b", temperature=temperature)

# Hybrid retriever (all logic hidden inside build_retriever)
retriever = build_retriever(vectorstore, all_chunks, k=k_chunks, use_rerank=use_rerank)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Chat
question = st.text_input("Ask a question about the document:", placeholder="e.g., What is the main contribution of this paper?")

if question:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": question})

    st.markdown("### Answer")
    st.write(result["result"])

    st.markdown("### Sources")
    for i, doc in enumerate(result["source_documents"]):
        page = doc.metadata.get('page', '?')
        st.caption(f"Source {i+1} — page {page}")
        with st.expander("Show text"):
            st.text(doc.page_content[:500])