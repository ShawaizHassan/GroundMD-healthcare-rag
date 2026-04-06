import ollama
from privacy_filter import PrivacyFilter
from medical_knowledge_base import MedicalKnowledgeBase

class HealthcareRAG:
    def __init__(self):
        self.privacy_filter = PrivacyFilter()
        self.knowledge_base = MedicalKnowledgeBase()
        self.model = "gemma3:4b"
    
    def process_query(self, query: str, top_k: int = 3):
        phi_detected = self.privacy_filter.detect_phi(query)
        anonymized_query = self.privacy_filter.anonymize(query)
        contexts = self.knowledge_base.search(anonymized_query, top_k)
        context_text = "\n\n".join([ctx["text"] for ctx in contexts])
        
        prompt = f"""Medical Context: {context_text}

Question: {anonymized_query}

Answer based only on the context:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        
        answer = response["message"]["content"]
        final_answer = self.privacy_filter.deanonymize(answer)
        
        citations = [{"text": ctx["text"][:150], "source": ctx["source"], "relevance_score": ctx["score"]} for ctx in contexts]
        self.privacy_filter.clear_mapping()
        
        return {
            "answer": final_answer,
            "citations": citations,
            "confidence": max([ctx["score"] for ctx in contexts]) if contexts else 0.5,
            "phi_detected": phi_detected,
            "status": "success"
        }