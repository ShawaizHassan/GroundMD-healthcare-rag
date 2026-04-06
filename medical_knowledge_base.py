import chromadb
from chromadb.utils import embedding_functions

class MedicalKnowledgeBase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./medical_db")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="medical_records",
            embedding_function=self.embedding_fn
        )
        if self.collection.count() == 0:
            self.load_documents()
    
    def load_documents(self):
        documents = [
            {
                "id": "patient_103203",
                "text": "Patient: Richard Wang, MRN: 103203. Type 2 Diabetes Mellitus. Medications: Metformin 500 mg twice daily, Lisinopril 10 mg once daily. HbA1c: 8.2%, Blood Pressure: 145/90 mmHg. No known drug allergies.",
                "metadata": {"source": "patient_record", "mrn": "103203"}
            },
            {
                "id": "doc_001",
                "text": "Wound Care Treatment: Clean wound with sterile saline. Apply antiseptic. Cover with sterile dressing. Follow-up within 3 days.",
                "metadata": {"source": "Medical Treatment Guide"}
            },
            {
                "id": "doc_002",
                "text": "Amoxicillin is a penicillin-type antibiotic. Contraindications: Penicillin allergy. Check for drug allergies before prescribing.",
                "metadata": {"source": "Drug Information"}
            },
            {
                "id": "doc_003",
                "text": "Type 2 Diabetes Management: Metformin is first-line medication. Monitor HbA1c every 3-6 months. Target HbA1c < 7%.",
                "metadata": {"source": "Clinical Guide"}
            }
        ]
        for doc in documents:
            self.collection.add(
                ids=[doc["id"]],
                documents=[doc["text"]],
                metadatas=[doc["metadata"]]
            )
        print(f"Loaded {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        contexts = []
        for i in range(len(results['documents'][0])):
            contexts.append({
                "text": results['documents'][0][i],
                "score": 1 - results['distances'][0][i] if results['distances'] else 0.8,
                "source": results['metadatas'][0][i].get('source', 'Unknown'),
                "id": results['ids'][0][i]
            })
        return contexts