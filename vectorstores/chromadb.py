from __future__ import annotations

from typing import List, Any
from pathlib import Path

import chromadb

from langchain_core.documents import Document
from ingestion.chunker import Chunker
from ingestion.embedder import EmbeddingPipeline

class ChromaVectorStore:
    def __init__(self, 
                 persist_dir: str = "/data/chroma_store", 
                 collection_name: str = "healthcare_docs", 
                 embdedding_model: str = "multi-qa-MiniLM-L6-cos-v1", 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200
                 ):
        
        self.model_name = embdedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        self.embedder = EmbeddingPipeline(
            model_name = self.model_name,
            chunk_overlap = self.chunk_overlap,
            chunk_size = self.chunk_size,
        )
        
        print(f"[INFO] chromadb collection: {self.collection_name} is ready")
        
    def build_from_docs(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("[DEBUG]No documents are provided")
        
        print(f"[INFO] building chroma store from {len(documents)} documents")
        
        chunker = Chunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = chunker.chunk_documents(documents=documents)
        
        if not chunks:
            raise ValueError("[DEBUG] chunking produced no chunks")
        
        print(f"[INFO] generating embeddings for {len(chunks)} chunks")
        
        embeddings = self.embedder.generate_embeddings(chunks=chunks)
        
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]]
        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            texts.append(chunk.page_content)
            metadatas.append(
                {
                    "texts": chunk.page_content,
                    **chunk.metadata
                }
            )
            
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(
            f"stored {len(documents)} in chroma store of collection name {self.collection_name}"
        )
        
    def reset_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.client.get_or_create_collection(name=self.collection_name)
        print(f"[INFO] collection {self.collection_name} is reset now")