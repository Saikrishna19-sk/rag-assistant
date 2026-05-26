class HybridRetriever:
    def __init__(self, store):
        self.store = store

    def build_bm25(self):
        pass

    def search(self, query, query_embedding, top_k=3):
        return self.store.search(query_embedding, top_k)