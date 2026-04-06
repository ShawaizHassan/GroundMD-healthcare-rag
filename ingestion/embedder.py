from sentence_transformers import SentenceTransformer
from typing import List, Any
import numpy as np
class EmbeddingPipeline:
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(self.model_name)
        print(f"[INFO] model {self.model_name} has been loaded")
    
    def generate_embeddings(self, chunks: List[Any]) -> np.ndarray:
        if not self.model:
            raise ValueError("[DEBUG] model is not loaded")
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] generating embeddings for {len(texts)} texts..")
        embeddings = self.model.encode(texts, show_progress_bar = True)
        print(f"Generated the embeddings with shape {embeddings.shape}")
        
        return embeddings