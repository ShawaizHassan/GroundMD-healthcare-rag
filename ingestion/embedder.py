from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "multi-qa-MiniLM-L6-cos-v1",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name)
        print(f"[INFO] Model '{self.model_name}' loaded successfully")

    def filter_valid_chunks(self, chunks: List[Document]) -> List[Document]:
        if not chunks:
            raise ValueError("Chunks list is empty")

        valid_chunks = []
        for chunk in chunks:
            text = chunk.page_content.strip() if chunk.page_content else ""
            if text:
                valid_chunks.append(chunk)

        if not valid_chunks:
            raise ValueError("No valid chunk text found for embedding")

        return valid_chunks

    def generate_embeddings(self, chunks: List[Document]) -> list[list[float]]:
        valid_chunks = self.filter_valid_chunks(chunks)
        texts = [chunk.page_content.strip() for chunk in valid_chunks]

        print(f"[INFO] Generating embeddings for {len(texts)} chunks")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        print(f"[INFO] Generated {len(embeddings)} embeddings")
        return embeddings

    def generate_query_embedding(self, text: str) -> list[float]:
        if self.model is None:
            raise ValueError("Embedding model is not loaded")

        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")

        clean_text = text.strip()

        print("[INFO] Generating query embedding")

        query_embedding = self.model.encode(
            clean_text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        print(f"[INFO] Generated query embedding of dimension {len(query_embedding)}")
        return query_embedding