# rag_final.py — fully clean, M1‑optimised, no warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os, sys
import torch

# 1. Load PDF
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "paper.pdf"
if not os.path.exists(pdf_path):
    print(f"ERROR: {pdf_path} not found. Place a PDF named 'paper.pdf' in the same folder.")
    sys.exit(1)

loader = PyPDFLoader(pdf_path)
documents = loader.load()
#print("DEBUG first page text:", documents[0].page_content[:500])
print(f"Loaded {len(documents)} pages.")

# 2. Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# 3. Embeddings — use MPS if available
from langchain_huggingface import HuggingFaceEmbeddings

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device for embeddings: {device}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. Vector store (Chroma)
from langchain_community.vectorstores import Chroma

persist_dir = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_dir
)
# No need for persist() – Chroma handles it automatically
print(f"Vector store saved to {persist_dir}")

# 5. LLM via Ollama
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral:7b", temperature=0.0)

# 6. RetrievalQA chain
# from langchain import RetrievalQA   # correct import
from langchain_classic.chains import RetrievalQA


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# 7. Ask a question
question = input("Enter your question: ")
print(f"\nQuestion: {question}")

# Optional: print retrieved chunks for debugging
print("\n=== RETRIEVED CHUNKS (debug) ===")
retrieved = vectorstore.similarity_search(question, k=3)
for i, doc in enumerate(retrieved):
    print(f"Chunk {i+1} (page {doc.metadata.get('page', '?')}):")
    print(doc.page_content[:300])
    print("---")

result = qa_chain.invoke({"query": question})

print("\n=== ANSWER ===")
print(result["result"])

print("\n=== SOURCES ===")
for i, doc in enumerate(result["source_documents"]):
    src = doc.metadata.get('source', 'unknown')
    page = doc.metadata.get('page', 'unknown')
    print(f"Source {i+1}: {src}, page {page}")