from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import re

app = FastAPI()

# Medical knowledge base
medical_info = {
    "diabetes": "Diabetes is a chronic condition where blood sugar levels are too high. Type 2 diabetes is common. Treatment includes Metformin and lifestyle changes.",
    "heart_disease": "Heart disease symptoms include chest pain, shortness of breath. Risk factors: smoking, high blood pressure, obesity.",
    "covid": "COVID-19 symptoms: fever, cough, loss of taste or smell. Prevention: vaccination and masks.",
    "cancer": "Cancer is abnormal cell growth. Treatments: surgery, chemotherapy, radiation.",
    "hypertension": "Hypertension or high blood pressure is when blood pressure is above 130/80 mmHg.",
    "asthma": "Asthma causes breathing difficulty. Treatment includes inhalers.",
    "pregnancy": "Pregnancy lasts about 40 weeks. Prenatal care includes regular checkups and healthy diet."
}

def find_relevant_info(query):
    query_lower = query.lower()
    for key, info in medical_info.items():
        if key in query_lower:
            return info
    return "I cannot find information about this topic in my database."

def detect_name(query):
    name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', query)
    if name_match:
        return name_match.group(1)
    return None

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    query = request.query
    
    # Detect name
    patient_name = detect_name(query)
    
    # Find relevant info
    context = find_relevant_info(query)
    
    # Generate answer with Ollama
    prompt = f"""Medical Information: {context}

Question: {query}

Answer concisely:"""
    
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    
    answer = response["message"]["content"]
    
    return {
        "answer": answer,
        "patient": patient_name,
        "status": "success"
    }

@app.get("/")
def root():
    return {"message": "Healthcare API", "status": "healthy"}