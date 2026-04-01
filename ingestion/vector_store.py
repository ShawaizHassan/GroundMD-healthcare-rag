import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from ingestion.embedder import EmbeddingPipeline
from ingestion.chunker import Chunker

class FaisVectorStore:
    def __init__(self, persist_dir: str = "C:/Users/PMLS/Desktop/IEDE/GroundMD-healthcare-rag/data/faiss_store", embedding_model: str = "multi-qa-MiniLM-L6-cos-v1", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        os.makedirs(self.persist_dir, exist_ok=True)
        print(f"[INFO] Loaded embedding model: {embedding_model}")
        
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunker = Chunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.chunk_documents(documents)
        embeddings = emb_pipe.generate_embeddings(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), self.metadatas)
        self.save()
        print(f"[INFO] vector store built and saved to {self.persist_dir}")
        
    def add_embeddings(self, embeddings: np.ndarray, metadatas: list[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] added {embeddings.shape[0]} Faiss index")
        
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path =os.path.join(self.persist_dir, "metadatas.pk1")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] saved faiss index and metadata to {self.persist_dir}")
        
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.idex")
        meta_path =os.path.join(self.persist_dir, "metadatas.pk1")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
            print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results  
    
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        quer_emb = self.model.encode([query_text]).astype('float32')
        return self.search(quer_emb, top_k=top_k)      