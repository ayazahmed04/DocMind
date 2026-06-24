# retrieval.py — Hybrid retrieval with BM25 + dense + cross-encoder re-ranking
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

class HybridRetriever(BaseRetriever):
    """
    Combines dense (Chroma) and sparse (BM25) retrieval using Reciprocal Rank Fusion (RRF).
    """
    dense_retriever: Any          # fixed: was 'any'
    bm25: BM25Okapi
    all_docs: List[Document]
    k: int = 5
    k_dense: int = 10
    k_sparse: int = 10
    rrf_k: int = 60

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Retrieve from dense and sparse
        # dense_docs = self.dense_retriever.get_relevant_documents(query)[:self.k_dense]
        dense_docs = self.dense_retriever.invoke(query)[:self.k_dense]
        sparse_docs = self._bm25_search(query)[:self.k_sparse]

        # 2. Merge using Reciprocal Rank Fusion
        all_docs = dense_docs + sparse_docs
        doc_to_score = {}
        for rank, doc in enumerate(dense_docs):
            doc_to_score[doc.page_content] = doc_to_score.get(doc.page_content, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, doc in enumerate(sparse_docs):
            doc_to_score[doc.page_content] = doc_to_score.get(doc.page_content, 0) + 1 / (self.rrf_k + rank + 1)

        # 3. Sort and take top k
        sorted_contents = sorted(doc_to_score.items(), key=lambda x: x[1], reverse=True)
        top_contents = [c for c, _ in sorted_contents[:self.k]]
        return [doc for doc in self.all_docs if doc.page_content in top_contents]

    def _bm25_search(self, query: str) -> List[Document]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:self.k_sparse]
        return [self.all_docs[i] for i in top_indices]


class CrossEncoderReranker:
    """Re-ranks a list of documents using a cross-encoder model."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return [docs[i] for i in sorted_indices]