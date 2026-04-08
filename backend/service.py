from typing import List, Dict

class Service:
    def retrieve(self, query: str, top_k:int) -> List[Dict]:
        return [
            {"text": f"Chunk 1 for {query}", "Score": 0.76},
            {"text": f"Chunk 2 for {query}", "Score": 0.83},
            {"text": f"Chunk 3 for {query}", "Score": 0.93}
        ][:top_k]
    
    def generate_answer(self, query: str, contexts: list) ->Dict:
        joined_context = " ".join([c["text"] for c in contexts])
        return {
            "answer": f"Mock answer for '{query}'using contexts{joined_context}",
            "citations":["doc1.pdf"]
        }
    
    def process_query(self, query:str, top_k: int) ->Dict:
        contexts = self.retrieve(query, top_k)
        llm_output = self.generate_answer(query, contexts)

        return {
            "answer": llm_output["answer"],
            "confidence": 0.93,
            "sources": llm_output["citations"],
            "status": "success"
        }