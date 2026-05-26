from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, results):
        pairs = [
            (query, r[1]["text"])
            for r in results
        ]

        scores = self.model.predict(pairs)

        reranked = list(zip(scores, [r[1] for r in results]))
        reranked.sort(reverse=True, key=lambda x: x[0])

        return reranked