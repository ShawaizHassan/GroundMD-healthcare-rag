from healthcare_rag import HealthcareRAG

rag = HealthcareRAG()
query = "For Richard Wang (MRN 103203), what medications is he currently taking?"
result = rag.process_query(query, 3)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"PHI Detected: {result['phi_detected']}")