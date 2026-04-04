import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vectorstores.faiss_store import FaissVectorStore
from vectorstores.chroma_store import ChromaVectorStore
from retriever.search import ChromaRetriever, FaissRetriever
from ingestion.data_loader import DataLoader



# Example usage
if __name__ == "__main__":
    # dataloader = DataLoader(r"C:\Users\PMLS\Desktop\IEDE\GroundMD-healthcare-rag\data\raw\guidelines")
    # docs = dataloader.load_all_documents()
    # store = ChromaVectorStore()
    # store.build_from_documents(docs)
    chroma_retriever = ChromaRetriever()
    
    
    print(chroma_retriever.query("What are the recommended HbA1c targets for patients with type 2 diabetes?", top_k=2))