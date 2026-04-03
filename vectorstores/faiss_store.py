from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunker
from ingestion.embedder import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = r"C:\Users\PMLS\Desktop\IEDE\GroundMD-healthcare-rag\data\faiss_store",
        embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []

        # Same model used for query embeddings
        self.query_model = SentenceTransformer(embedding_model)

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    @property
    def index_path(self) -> Path:
        return self.persist_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.persist_dir / "metadata.pkl"

    def build_from_documents(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents were provided to build the vector store.")

        print(f"[INFO] Building vector store from {len(documents)} raw documents")

        chunker = Chunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = chunker.chunk_documents(documents)

        if not chunks:
            raise ValueError("Chunking produced no chunks.")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        embeddings = emb_pipe.generate_embeddings(chunks)

        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        if embeddings_array.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D, but got shape {embeddings_array.shape}"
            )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)

        metadatas: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadatas.append(
                {
                    "text": chunk.page_content,
                    **chunk.metadata,
                }
            )

        self.add_embeddings(embeddings_array, metadatas)
        self.save()

        print(
            f"[INFO] Vector store built with {len(chunks)} chunks and saved to {self.persist_dir}"
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array, but got shape {embeddings.shape}"
            )

        if len(embeddings) != len(metadatas):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) and metadata count ({len(metadatas)}) must match."
            )

        dim = embeddings.shape[1]

        if self.index is None:
            # Inner product on normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} embeddings to FAISS index")

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Cannot save because the FAISS index is empty.")

        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built yet.")

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim != 2:
            raise ValueError(
                f"Query embedding must be 2D with shape (1, dim), got {query_embedding.shape}"
            )

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append(
                {
                    "index": int(idx),
                    "score": float(score),
                    "metadata": meta,
                }
            )

        return results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        print(f"[INFO] Querying vector store for: {query_text!r}")

        query_embedding = self.query_model.encode(
            [query_text],
            convert_to_numpy=True,
        ).astype(np.float32)

        return self.search(query_embedding, top_k=top_k)