import numpy as np

class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.metadata = []

    def add(self, embedding, meta):
        self.embeddings.append(np.array(embedding, dtype="float32"))
        self.metadata.append(meta)

    def search(self, query_embedding, top_k=3):
        if not self.embeddings:
            return []
        query_vec = np.array(query_embedding, dtype="float32")
        scores = [
            float(np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb)))
            for emb in self.embeddings
        ]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(scores[i], self.metadata[i]) for i in top_indices]