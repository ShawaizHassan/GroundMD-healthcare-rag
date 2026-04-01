import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.vector_store import FaisVectorStore
from ingestion.data_loader import DataLoader



# Example usage
if __name__ == "__main__":
    dataloader = DataLoader()
    docs = dataloader.load_all_documents("../data/raw/guidelines")
    store = FaisVectorStore()
    store.build_from_documents(docs)