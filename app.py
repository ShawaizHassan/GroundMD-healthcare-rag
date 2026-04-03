import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vectorstores.faiss_store import FaissVectorStore
from ingestion.data_loader import DataLoader



# Example usage
if __name__ == "__main__":
    dataloader = DataLoader()
    docs = dataloader.load_all_documents()
    store = FaissVectorStore()
    store.build_from_documents(docs)
    store.load()
    print(store.query("What are the recommended screening interventions for adolescents (10–19 years) living with HIV?", top_k=2))