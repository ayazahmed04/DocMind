# app.py – Streamlit UI for DocMind (local RAG)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os, sys
import torch
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA


# Page config
st.set_page_config(page_title="DocMind", page_icon="📄", layout="centered")
st.title("📄 DocMind — Chat with your PDFs")
st.markdown("Ask questions about your document. All processing is local and private.")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    chunk_size = st.slider("Chunk size", 300, 1500, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 200, 10)
    k_chunks = st.slider("Retrieved chunks", 1, 10, 5)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
    st.markdown("---")
    st.caption("Runs fully locally with Ollama + Mistral 7B")

# Load or reload the vector store
@st.cache_resource
def load_vectorstore(file_path, _chunk_size, _chunk_overlap):
    if not os.path.exists(file_path):
        return None
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_ui_db"
    )
    return vectorstore, len(documents), len(chunks)

# Load PDF
if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_path = "temp_uploaded.pdf"
else:
    pdf_path = "paper.pdf"   # fallback to local file

if not os.path.exists(pdf_path):
    st.error("No PDF found. Upload a document or place 'paper.pdf' in the project folder.")
    st.stop()

with st.spinner("Loading PDF and building vector store..."):
    vectorstore, num_pages, num_chunks = load_vectorstore(pdf_path, chunk_size, chunk_overlap)

if vectorstore is None:
    st.error("Failed to load the document.")
    st.stop()

st.success(f"Loaded {num_pages} pages → {num_chunks} chunks")

# LLM (cached)
@st.cache_resource
def load_llm():
    return OllamaLLM(model="mistral:7b", temperature=temperature)

llm = load_llm()

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": k_chunks}),
    return_source_documents=True
)

# Chat interface
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