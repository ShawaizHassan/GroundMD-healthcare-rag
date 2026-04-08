from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(self.model_name)
        print(f"[INFO] Reranker model '{self.model_name}' loaded successfully")

    def rerank(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if not retrieved_docs:
            return []

        pairs = [(query, doc["document"]) for doc in retrieved_docs]
        scores = self.model.predict(pairs)

        reranked_output = []
        for doc, score in zip(retrieved_docs, scores):
            item = doc.copy()
            item["rerank_score"] = float(score)
            reranked_output.append(item)

        reranked_output.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_output[:top_k]