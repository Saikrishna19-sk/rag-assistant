import numpy as np
import faiss

class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(1536)
        self.metadata = []

    def add(self, embedding, meta):
        vec = np.array(embedding, dtype="float32")
        faiss.normalize_L2(vec.reshape(1, -1))
        self.index.add(vec.reshape(1, -1))
        self.metadata.append(meta)

    def search(self, query_embedding, top_k=3):
        query_vec = np.array(query_embedding, dtype="float32")
        faiss.normalize_L2(query_vec.reshape(1, -1))
        scores, indices = self.index.search(query_vec.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx]))
        return results