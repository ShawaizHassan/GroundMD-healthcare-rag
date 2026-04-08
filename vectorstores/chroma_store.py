from __future__ import annotations

from typing import List, Any
from pathlib import Path
import hashlib

import chromadb
from langchain_core.documents import Document

from ingestion.chunker import Chunker
from ingestion.embedder import EmbeddingPipeline


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = r"C:\Users\PMLS\Desktop\IEDE\GroundMD-healthcare-rag\data\chroma_store",
        collection_name: str = "healthcare_docs",
        embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 1000,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        self.embedder = EmbeddingPipeline(
            model_name=self.model_name
        )

        print(f"[INFO] Chroma collection '{self.collection_name}' is ready")

    def build_from_documents(
        self,
        documents: List[Document],
        reset: bool = False,
    ) -> None:
        if not documents:
            raise ValueError("No documents provided")

        if reset:
            self.reset_collection()

        print(f"[INFO] Building Chroma store from {len(documents)} documents")

        chunker = Chunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = chunker.chunk_documents(documents=documents)

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        print(f"[INFO] Total chunks created: {len(chunks)}")

        total_inserted = 0

        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start:start + self.batch_size]

            ids = []
            texts = []
            metadatas = []

            for local_idx, chunk in enumerate(batch_chunks):
                global_idx = start + local_idx

                chunk_id = self._generate_chunk_id(chunk, global_idx)

                metadata = dict(chunk.metadata) if chunk.metadata else {}

                metadata.setdefault("source", "unknown")
                metadata.setdefault("source_file", Path(metadata.get("source", "unknown")).name)
                metadata.setdefault("page", -1)
                metadata.setdefault("disease_name", "unknown")
                metadata.setdefault("chunk_index", global_idx)

                ids.append(chunk_id)
                texts.append(chunk.page_content)
                metadatas.append(metadata)

            embeddings = self.embedder.generate_embeddings(chunks=batch_chunks)

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            total_inserted += len(batch_chunks)
            print(
                f"[INFO] Inserted batch {start // self.batch_size + 1} | "
                f"chunks: {len(batch_chunks)} | total inserted: {total_inserted}"
            )

        print(
            f"[INFO] Finished storing {total_inserted} chunks "
            f"from {len(documents)} documents into '{self.collection_name}'"
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = self.embedder.generate_query_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        documents: List[Document] = []

        result_docs = results.get("documents", [[]])[0]
        result_metas = results.get("metadatas", [[]])[0]

        for text, metadata in zip(result_docs, result_metas):
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata or {},
                )
            )

        return documents

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"[INFO] Deleted collection '{self.collection_name}'")
        except Exception:
            print(f"[INFO] Collection '{self.collection_name}' did not exist, creating fresh one")

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"[INFO] Collection '{self.collection_name}' is reset and ready")

    def count(self) -> int:
        return self.collection.count()

    def _generate_chunk_id(self, chunk: Document, chunk_index: int) -> str:
        metadata = chunk.metadata or {}

        source_file = str(metadata.get("source_file", "unknown"))
        page = str(metadata.get("page", "na"))
        text = chunk.page_content.strip()

        raw = f"{source_file}|{page}|{text}"
        text_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

        safe_source = Path(source_file).stem.replace(" ", "_")

        return f"{safe_source}_p{page}_c{chunk_index}_{text_hash}"