import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.data_loader import load_all_documents
from ingestion.embedder import EmbeddingPipeline



# Example usage
if __name__ == "__main__":
    docs = load_all_documents("C:/Users/PMLS/Desktop/IEDE/GroundMD-healthcare-rag/data/raw/guidelines")
    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunkembeddings = EmbeddingPipeline().generate_embeddings(chunks)
    print(chunkembeddings)