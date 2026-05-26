import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_store):
        self.store = vector_store
        self.bm25 = None
        self.corpus = []

    def build_bm25(self):
        self.corpus = [
            meta["text"].split()
            for meta in self.store.metadata
        ]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, query_embedding, top_k=5):
        # 1. FAISS results
        faiss_results = self.store.search(query_embedding, top_k)

        # 2. BM25 results
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

        bm25_results = [
            (float(bm25_scores[i]), self.store.metadata[i])
            for i in bm25_top
        ]

        # 3. Merge results
        combined = faiss_results + bm25_results

        # 4. Sort final
        combined.sort(key=lambda x: x[0], reverse=True)

        return combined[:top_k]
    