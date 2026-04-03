import chromadb
from typing import List, Any
from pathlib import Path

from vectorstores.chromadb import ChromaVectorStore
from ingestion.embedder import EmbeddingPipeline

class RetrieverPipeline:
    def __init__(self, 
                 embedding_model: str = "multi-qa-MiniLM-L6-cos-v1", 
                 persist_dir: str = "/data/chroma_store",
                 collection_name: str = "healthcare_docs"
                 ):
        self.embedding_model = embedding_model
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_collection(name=self.collection_name)
        
    def query(self, query_text: str, top_k: int = 5) -> List[dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("[DEBUG] query cannot be empty")
        
        self.embedder = EmbeddingPipeline()
        embeddings = self.embedder.generate_query_embeddings(query_text)
        