# retrieval.py — Hybrid retrieval with BM25 + dense + cross-encoder re-ranking
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

class HybridRetriever(BaseRetriever):
    """Combines dense (Chroma) and sparse (BM25) using Reciprocal Rank Fusion (RRF)."""
    dense_retriever: Any
    bm25: BM25Okapi
    all_docs: List[Document]
    k: int = 5
    k_dense: int = 10
    k_sparse: int = 10
    rrf_k: int = 60

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_docs = self.dense_retriever.invoke(query)[:self.k_dense]
        sparse_docs = self._bm25_search(query)[:self.k_sparse]

        all_docs = dense_docs + sparse_docs
        doc_to_score = {}
        for rank, doc in enumerate(dense_docs):
            doc_to_score[doc.page_content] = doc_to_score.get(doc.page_content, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, doc in enumerate(sparse_docs):
            doc_to_score[doc.page_content] = doc_to_score.get(doc.page_content, 0) + 1 / (self.rrf_k + rank + 1)

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


class CustomRetriever(BaseRetriever):
    """Wraps hybrid + optional re-ranker."""
    hybrid_retriever: HybridRetriever
    reranker: CrossEncoderReranker = None
    use_rerank: bool = False
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.hybrid_retriever.invoke(query)
        if self.use_rerank and self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=self.k)
        return docs


def build_retriever(vectorstore, all_chunks, k: int = 5, use_rerank: bool = True):
    """
    Build a complete hybrid retriever with optional re-ranking.
    Takes the Chroma vectorstore and the list of chunk Documents.
    """
    corpus = [doc.page_content for doc in all_chunks]
    tokenized_corpus = [text.split() for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25=bm25,
        all_docs=all_chunks,
        k=k,
        k_dense=10,
        k_sparse=10
    )

    reranker = CrossEncoderReranker() if use_rerank else None

    return CustomRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
        use_rerank=use_rerank,
        k=k
    )