class RAG:
    def __init__(self, store, hybrid, reranker):
        self.store = store
        self.hybrid = hybrid
        self.reranker = reranker

    def retrieve(self, query, query_embedding):
        results = self.hybrid.search(query, query_embedding)

        # rerank
        reranked = self.reranker.rerank(query, results)

        return reranked[:3]