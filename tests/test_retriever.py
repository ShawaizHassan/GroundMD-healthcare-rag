import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vectorstores.faiss_store import FaissVectorStore
from vectorstores.chroma_store import ChromaVectorStore
from retriever.search import ChromaRetriever, FaissRetriever
from ingestion.data_loader import DataLoader
from retriever.reranker import Reranker



# Example usage
if __name__ == "__main__":
    # dataloader = DataLoader(r"C:\Users\PMLS\Desktop\IEDE\GroundMD-healthcare-rag\data\raw\guidelines")
    # docs = dataloader.load_all_documents()
    # store = ChromaVectorStore()
    # store.build_from_documents(docs)
    chroma_retriever = ChromaRetriever()
    reranker = Reranker()
    query = "What are the recommended HbA1c targets for patients with type 2 diabetes?"
    retrieved_docs = chroma_retriever.query(query, top_k=10)
    final_answer = reranker.rerank(query, retrieved_docs, top_k=3)
    print(final_answer)