import ollama
import re
import numpy as np

class RetrievalRAG:
    def __init__(self):
        self.model = "gemma3:4b"
        self.patient_mapping = {}
        self.documents = []
        self.document_embeddings = []
        self.load_all_documents()
    
    def load_all_documents(self):
        """Load all medical documents"""
        self.documents = [
            {
                "id": "doc_001",
                "text": "Diabetes: Type 2 diabetes is a chronic condition affecting blood sugar. Treatment: Metformin, lifestyle changes, regular exercise.",
                "metadata": {"source": "medical_db", "topic": "diabetes"}
            },
            {
                "id": "doc_002",
                "text": "Heart Disease: Symptoms include chest pain, shortness of breath. Risk factors: smoking, high blood pressure, obesity.",
                "metadata": {"source": "medical_db", "topic": "heart"}
            },
            {
                "id": "doc_003",
                "text": "COVID-19: Caused by SARS-CoV-2 virus. Symptoms: fever, cough, loss of taste or smell. Prevention: vaccination, masks, social distancing.",
                "metadata": {"source": "medical_db", "topic": "covid"}
            },
            {
                "id": "doc_004",
                "text": "Cancer: Abnormal cell growth that can spread to other body parts. Common types: breast, lung, colon, prostate. Treatment: surgery, chemotherapy, radiation.",
                "metadata": {"source": "medical_db", "topic": "cancer"}
            },
            {
                "id": "doc_005",
                "text": "Pregnancy: Duration is 40 weeks. Prenatal care: regular checkups, vitamins, healthy diet, avoid alcohol and smoking.",
                "metadata": {"source": "medical_db", "topic": "pregnancy"}
            },
            {
                "id": "doc_006",
                "text": "Hypertension: High blood pressure above 130/80 mmHg. Risk factors: age, family history, obesity, stress. Treatment: lifestyle changes, medication.",
                "metadata": {"source": "medical_db", "topic": "blood_pressure"}
            },
            {
                "id": "doc_007",
                "text": "Asthma: Chronic lung disease causing breathing difficulties. Triggers: allergens, exercise, cold air. Treatment: inhalers, avoiding triggers.",
                "metadata": {"source": "medical_db", "topic": "asthma"}
            }
        ]
        
        # Create embeddings
        print("Creating embeddings for documents...")
        for doc in self.documents:
            embedding = self.get_embedding(doc["text"])
            self.document_embeddings.append(embedding)
        print(f"Loaded {len(self.documents)} documents")
    
    def get_embedding(self, text):
        """Get embedding using Ollama"""
        response = ollama.embeddings(
            model=self.model,
            prompt=text
        )
        return response["embedding"]
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query, top_k=3):
        """Search for relevant documents"""
        query_embedding = self.get_embedding(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            score = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, score))
        
        # Sort by score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top_k results
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "text": self.documents[i]["text"],
                "score": round(score, 3),
                "source": self.documents[i]["metadata"]["source"],
                "topic": self.documents[i]["metadata"]["topic"]
            })
        
        return results
    
    def detect_phi(self, text):
        """Detect PHI (Personal Health Information)"""
        detected = []
        # Detect name
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
        if name_match:
            detected.append(('NAME', name_match.group(1)))
        # Detect MRN
        mrn_match = re.search(r'MRN\s*[:]?\s*(\d+)', text, re.IGNORECASE)
        if mrn_match:
            detected.append(('MRN', mrn_match.group(1)))
        return detected
    
    def anonymize(self, text):
        """Replace PHI with placeholders"""
        anonymized = text
        self.patient_mapping = {}
        counter = 1
        
        # Anonymize MRN
        mrn_match = re.search(r'MRN\s*[:]?\s*(\d+)', anonymized, re.IGNORECASE)
        if mrn_match:
            mrn = mrn_match.group(1)
            placeholder = f'PATIENT_{counter}'
            self.patient_mapping[placeholder] = {'mrn': mrn}
            anonymized = re.sub(r'MRN\s*[:]?\s*' + mrn, placeholder, anonymized, flags=re.IGNORECASE)
            counter += 1
        
        # Anonymize name
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', anonymized)
        if name_match:
            name = name_match.group(1)
            placeholder = f'PATIENT_{counter}'
            self.patient_mapping[placeholder] = {'name': name}
            anonymized = anonymized.replace(name, placeholder)
        
        return anonymized
    
    def deanonymize(self, text):
        """Replace placeholders with real PHI"""
        result = text
        for placeholder, info in self.patient_mapping.items():
            if 'name' in info:
                result = result.replace(placeholder, info['name'])
            if 'mrn' in info:
                result = result.replace(placeholder, f"MRN {info['mrn']}")
        return result
    
    def process_query(self, query, top_k=3):
        """Process query with RAG (Retrieval + Generation)"""
        
        # Step 1: Detect PHI
        phi_detected = self.detect_phi(query)
        print(f"Detected PHI: {phi_detected}")
        
        # Step 2: Anonymize query
        anonymized_query = self.anonymize(query)
        print(f"Anonymized: {anonymized_query}")
        
        # Step 3: Search relevant documents (RETRIEVAL)
        relevant_docs = self.search(anonymized_query, top_k)
        
        # Step 4: Prepare context
        context_text = "\n\n".join([doc["text"] for doc in relevant_docs])
        
        # Step 5: Generate answer using LLM
        prompt = f"""Medical Information from database:
{context_text}

Question: {anonymized_query}

Instructions:
1. Answer based ONLY on the medical information above
2. If information is not in the context, say "I cannot find this information"
3. Be concise and accurate

Answer:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        
        answer = response["message"]["content"]
        
        # Step 6: Deanonymize answer
        final_answer = self.deanonymize(answer)
        
        # Step 7: Prepare citations
        citations = []
        for doc in relevant_docs:
            citations.append({
                "text": doc["text"][:150] + "...",
                "source": doc["source"],
                "topic": doc["topic"],
                "relevance_score": doc["score"]
            })
        
        return {
            "answer": final_answer,
            "citations": citations,
            "phi_detected": phi_detected,
            "retrieved_docs_count": len(relevant_docs),
            "status": "success"
        }

# Test the system
if __name__ == "__main__":
    print("="*60)
    print("Healthcare RAG System with Privacy Filter")
    print("="*60)
    
    rag = RetrievalRAG()
    
    test_queries = [
        "What is diabetes?",
        "Tell me about heart disease",
        "What are COVID-19 symptoms?",
        "How is cancer treated?",
        "What is pregnancy care?",
        "For Richard Wang, what is hypertension?"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"Query: {query}")
        print("="*50)
        
        result = rag.process_query(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nPHI Detected: {result['phi_detected']}")
        print(f"\nCitations:")
        for i, cit in enumerate(result['citations'], 1):
            print(f"  {i}. Topic: {cit['topic']} (Score: {cit['relevance_score']})")