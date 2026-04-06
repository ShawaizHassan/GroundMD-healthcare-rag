import ollama
import re

class SimpleHealthcareAI:
    def __init__(self):
        self.model = "gemma3:4b"
        self.patient_mapping = {}
        
        self.medical_knowledge = {
            "medications": "Metformin 500 mg twice daily, Lisinopril 10 mg once daily",
            "diabetes": "Type 2 Diabetes Mellitus. HbA1c: 8.2%",
            "blood_pressure": "145/90 mmHg",
            "allergies": "No known drug allergies"
        }
    
    def detect_phi(self, text):
        detected = []
        mrn_match = re.search(r'MRN\s*[:]?\s*(\d+)', text, re.IGNORECASE)
        if mrn_match:
            detected.append(('MRN', mrn_match.group(1)))
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
        if name_match:
            detected.append(('NAME', name_match.group(1)))
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
    
    def get_context(self, query):
        query_lower = query.lower()
        contexts = []
        
        if 'medication' in query_lower or 'drug' in query_lower or 'taking' in query_lower:
            contexts.append(self.medical_knowledge['medications'])
        if 'diabetes' in query_lower:
            contexts.append(self.medical_knowledge['diabetes'])
        if 'blood pressure' in query_lower:
            contexts.append(self.medical_knowledge['blood_pressure'])
        if 'allergy' in query_lower:
            contexts.append(self.medical_knowledge['allergies'])
        
        if not contexts:
            contexts = ["Medical information not available"]
        
        return contexts
    
    def process_query(self, query):
        phi_detected = self.detect_phi(query)
        anonymized_query = self.anonymize(query)
        contexts = self.get_context(anonymized_query)
        context_text = "\n".join(contexts)
        
        prompt = f"""Medical Information: {context_text}

Question: {anonymized_query}

Answer based on medical information:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        
        answer = response["message"]["content"]
        final_answer = self.deanonymize(answer)
        
        return {
            "answer": final_answer,
            "phi_detected": phi_detected,
            "status": "success"
        }