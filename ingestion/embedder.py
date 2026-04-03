from sentence_transformers import SentenceTransformer
from typing import List
from langchain_core.documents import Document
import numpy as np


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "multi-qa-MiniLM-L6-cos-v1",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(self.model_name)
        print(f"[INFO] Model '{self.model_name}' has been loaded")

    def generate_embeddings(self, chunks: List[Document]) -> np.ndarray:
        if self.model is None:
            raise ValueError("[ERROR] Embedding model is not loaded.")

        if not chunks:
            raise ValueError("[ERROR] Chunks list is empty.")

        texts = [chunk.page_content.strip() for chunk in chunks if chunk.page_content.strip()]
        if not texts:
            raise ValueError("[ERROR] No valid chunk text found for embedding.")

        print(f"[INFO] Generating embeddings for {len(texts)} document chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        print(f"[INFO] Generated document embeddings with shape {embeddings.shape}")
        return embeddings

    def generate_query_embedding(self, text: str) -> np.ndarray:
        if self.model is None:
            raise ValueError("[ERROR] Embedding model is not loaded.")

        if not text or not text.strip():
            raise ValueError("[ERROR] Query text cannot be empty.")

        print("[INFO] Generating embedding for query text...")
        query_embedding = self.model.encode(
            [text.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        print(f"[INFO] Generated query embedding with shape {query_embedding.shape}")
        return query_embedding