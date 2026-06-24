# app.py – Streamlit UI for DocMind (local RAG) + hybrid retrieval & re-ranking
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
from langchain_classic.chains import RetrievalQA

# NEW: imports for hybrid retrieval
from rank_bm25 import BM25Okapi
from retrieval import HybridRetriever, CrossEncoderReranker
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

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
    # NEW: re-ranking toggle
    use_rerank = st.checkbox("Use cross-encoder re-ranking", value=True)
    st.markdown("---")
    st.caption("Runs fully locally with Ollama + Mistral 7B")

# Load or reload the vector store – NEW: returns chunks as well
@st.cache_resource
def load_vectorstore(file_path, _chunk_size, _chunk_overlap):
    if not os.path.exists(file_path):
        return None, 0, 0, []          # return empty list for chunks
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
    return vectorstore, len(documents), len(chunks), chunks   # NEW: return chunks

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

with st.spinner("Loading PDF and building vector store..."):
    vectorstore, num_pages, num_chunks, all_chunks = load_vectorstore(pdf_path, chunk_size, chunk_overlap)  # NEW: unpack chunks

if vectorstore is None:
    st.error("Failed to load the document.")
    st.stop()

st.success(f"Loaded {num_pages} pages → {num_chunks} chunks")

# LLM (cached)
@st.cache_resource
def load_llm():
    return OllamaLLM(model="mistral:7b", temperature=temperature)

llm = load_llm()

# ================== NEW: Hybrid retrieval + re-ranking ==================
# Build BM25 index from all chunks
corpus = [doc.page_content for doc in all_chunks]
tokenized_corpus = [text.split() for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Dense retriever (Chroma)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Hybrid retriever using RRF
hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    bm25=bm25,
    all_docs=all_chunks,
    k=k_chunks,          # final number of docs
    k_dense=10,
    k_sparse=10
)

# Optional cross-encoder re-ranker
reranker = CrossEncoderReranker() if use_rerank else None

# Custom retriever that wraps hybrid + re-ranking
class CustomRetriever(BaseRetriever):
    hybrid_retriever: HybridRetriever
    reranker: CrossEncoderReranker = None
    use_rerank: bool = False
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # docs = self.hybrid_retriever.get_relevant_documents(query)
        docs = self.hybrid_retriever.invoke(query)
        if self.use_rerank and self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=self.k)
        return docs

final_retriever = CustomRetriever(
    hybrid_retriever=hybrid_retriever,
    reranker=reranker,
    use_rerank=use_rerank,
    k=k_chunks
)
# ========================================================================

# QA chain – now uses the hybrid retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=final_retriever,   # NEW: hybrid retriever
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