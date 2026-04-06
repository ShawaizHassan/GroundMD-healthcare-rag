import ollama
import re

class KeywordRAG:
    def __init__(self):
        self.model = "gemma3:4b"
        self.patient_mapping = {}
        
        # Medical documents
        self.documents = [
            {"id": "doc_001", "text": "Diabetes is a chronic condition where blood sugar levels are too high. Type 2 diabetes is common. Treatment includes Metformin and lifestyle changes.", "keywords": ["diabetes", "blood sugar", "metformin"], "topic": "diabetes"},
            {"id": "doc_002", "text": "Heart disease symptoms include chest pain, shortness of breath, and fatigue. Risk factors include smoking, high blood pressure, and obesity.", "keywords": ["heart", "chest pain", "blood pressure"], "topic": "heart"},
            {"id": "doc_003", "text": "COVID-19 is caused by coronavirus. Symptoms: fever, cough, loss of taste or smell. Prevention: vaccination and masks.", "keywords": ["covid", "coronavirus", "fever", "cough"], "topic": "covid"},
            {"id": "doc_004", "text": "Cancer is abnormal cell growth. Common types: breast, lung, colon cancer. Treatments: surgery, chemotherapy, radiation.", "keywords": ["cancer", "tumor", "chemotherapy"], "topic": "cancer"},
            {"id": "doc_005", "text": "Hypertension or high blood pressure is when blood pressure is above 130/80 mmHg. Treatment includes medication and lifestyle changes.", "keywords": ["hypertension", "blood pressure", "high bp"], "topic": "blood_pressure"},
            {"id": "doc_006", "text": "Asthma is a lung disease causing breathing difficulty. Triggers: allergies, exercise, cold air. Treatment: inhalers.", "keywords": ["asthma", "breathing", "inhaler"], "topic": "asthma"},
            {"id": "doc_007", "text": "Pregnancy lasts about 40 weeks. Prenatal care includes regular checkups, vitamins, healthy diet, and avoiding alcohol.", "keywords": ["pregnancy", "prenatal", "baby"], "topic": "pregnancy"}
        ]
    
    def search_by_keyword(self, query, top_k=3):
        """Search documents by keyword matching"""
        query_lower = query.lower()
        scores = []
        
        for doc in self.documents:
            score = 0
            # Check if query matches keywords
            for keyword in doc["keywords"]:
                if keyword in query_lower:
                    score += 1
            # Check if query matches topic
            if doc["topic"] in query_lower:
                score += 2
            # Check if query matches text
            if any(word in doc["text"].lower() for word in query_lower.split()[:3]):
                score += 0.5
            
            if score > 0:
                scores.append((doc, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for doc, score in scores[:top_k]:
            results.append({
                "text": doc["text"],
                "score": min(score / 3, 0.95),  # Normalize score
                "source": "medical_db",
                "topic": doc["topic"]
            })
        
        return results
    
    def detect_phi(self, text):
        detected = []
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
        if name_match:
            detected.append(('NAME', name_match.group(1)))
        mrn_match = re.search(r'MRN\s*[:]?\s*(\d+)', text, re.IGNORECASE)
        if mrn_match:
            detected.append(('MRN', mrn_match.group(1)))
        return detected
    
    def anonymize(self, text):
        anonymized = text
        self.patient_mapping = {}
        counter = 1
        
        mrn_match = re.search(r'MRN\s*[:]?\s*(\d+)', anonymized, re.IGNORECASE)
        if mrn_match:
            mrn = mrn_match.group(1)
            placeholder = f'PATIENT_{counter}'
            self.patient_mapping[placeholder] = {'mrn': mrn}
            anonymized = re.sub(r'MRN\s*[:]?\s*' + mrn, placeholder, anonymized, flags=re.IGNORECASE)
            counter += 1
        
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', anonymized)
        if name_match:
            name = name_match.group(1)
            placeholder = f'PATIENT_{counter}'
            self.patient_mapping[placeholder] = {'name': name}
            anonymized = anonymized.replace(name, placeholder)
        
        return anonymized
    
    def deanonymize(self, text):
        result = text
        for placeholder, info in self.patient_mapping.items():
            if 'name' in info:
                result = result.replace(placeholder, info['name'])
            if 'mrn' in info:
                result = result.replace(placeholder, f"MRN {info['mrn']}")
        return result
    
    def process_query(self, query, top_k=3):
        print(f"Original query: {query}")
        
        # Detect and anonymize PHI
        phi_detected = self.detect_phi(query)
        print(f"PHI detected: {phi_detected}")
        
        anonymized_query = self.anonymize(query)
        print(f"Anonymized: {anonymized_query}")
        
        # Search by keyword
        relevant_docs = self.search_by_keyword(anonymized_query, top_k)
        print(f"Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            return {
                "answer": "I cannot find information about this topic in my medical database.",
                "citations": [],
                "phi_detected": phi_detected,
                "status": "no_results"
            }
        
        # Prepare context
        context_text = "\n\n".join([doc["text"] for doc in relevant_docs])
        
        # Generate answer
        prompt = f"""Medical Information:
{context_text}

Question: {anonymized_query}

Answer based ONLY on the medical information above. Be concise:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        
        answer = response["message"]["content"]
        final_answer = self.deanonymize(answer)
        
        # Prepare citations
        citations = []
        for doc in relevant_docs:
            citations.append({
                "text": doc["text"][:120] + "...",
                "source": doc["source"],
                "topic": doc["topic"],
                "relevance_score": doc["score"]
            })
        
        return {
            "answer": final_answer,
            "citations": citations,
            "phi_detected": phi_detected,
            "retrieved_count": len(relevant_docs),
            "status": "success"
        }

# Test
if __name__ == "__main__":
    rag = KeywordRAG()
    
    test_queries = [
        "What is diabetes?",
        "Tell me about heart disease",
        "COVID-19 symptoms",
        "What is hypertension?",
        "For Richard Wang, what is cancer?"
    ]
    
    for q in test_queries:
        print("\n" + "="*50)
        result = rag.process_query(q)
        print(f"Answer: {result['answer']}")