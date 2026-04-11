import chromadb
from typing import Any, Dict, List
from pathlib import Path

from ingestion.embedder import EmbeddingPipeline

BASE_DIR = Path(__file__).resolve().parent.parent

class ChromaRetriever:
    def __init__(
        self,
        embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
        persist_dir: str = None,
        collection_name: str = "healthcare_docs"
    ):
        if persist_dir is None:
            self.persist_dir = BASE_DIR / "data" / "chroma_store"
        else:
            self.persist_dir = Path(persist_dir)
            
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_collection(name=self.collection_name)
        self.embedder = EmbeddingPipeline()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("query cannot be empty")

        query_embedding = self.embedder.generate_query_embedding(query_text)

        print(f"[INFO] Querying vector store for: {query_text!r}")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        ids_data = results.get("ids") or [[]]
        documents_data = results.get("documents") or [[]]
        metadatas_data = results.get("metadatas") or [[]]
        distances_data = results.get("distances") or [[]]

        ids = ids_data[0] if ids_data else []
        documents = documents_data[0] if documents_data else []
        metadatas = metadatas_data[0] if metadatas_data else []
        distances = distances_data[0] if distances_data else []

        output: List[Dict[str, Any]] = []

        for doc_id, doc_text, doc_metadata, distance in zip(ids, documents, metadatas, distances):
            output.append({
                "id": doc_id,
                "document": doc_text,
                "metadata": doc_metadata or {},
                "distance": float(distance)
            })

        return output
    
# class FaissRetriever:
#     def __init__(
#         self,
#         persist_dir: str = r"C:\Users\PMLS\Desktop\IEDE\GroundMD-healthcare-rag\data\faiss_store",
#         embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"
#         ):
        
#         self.persist_dir = Path(persist_dir)
#         self.embedding_model = embedding_model
#         self.index: faiss.Index | None = None
    
#     @property
#     def index_path(self) -> Path:
#         return self.persist_dir /  "faiss.index"
    
#     @property
#     def metadata_path(self) -> Path:
#         return self.persist_dir / "metadata.pkl"  
    
#     def load(self) -> None:
#         if not self.index_path.exists():
#             raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")

#         if not self.metadata_path.exists():
#             raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

#         self.index = faiss.read_index(str(self.index_path))

#         with open(self.metadata_path, "rb") as f:
#             self.metadata = pickle.load(f)

#         print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")

#     def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
#         if self.index is None:
#             raise ValueError("FAISS index is not loaded or built yet.")

#         query_embedding = np.asarray(query_embedding, dtype=np.float32)
#         if query_embedding.ndim != 2:
#             raise ValueError(
#                 f"Query embedding must be 2D with shape (1, dim), got {query_embedding.shape}"
#             )

#         faiss.normalize_L2(query_embedding)

#         scores, indices = self.index.search(query_embedding, top_k)

#         results: List[Dict[str, Any]] = []
#         for idx, score in zip(indices[0], scores[0]):
#             if idx == -1:
#                 continue

#             meta = self.metadata[idx] if idx < len(self.metadata) else None
#             results.append(
#                 {
#                     "index": int(idx),
#                     "score": float(score),
#                     "metadata": meta,
#                 }
#             )

#         return results

#     def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         if not query_text.strip():
#             raise ValueError("Query text cannot be empty.")
        
#         if self.index is None:
#             self.load()

#         print(f"[INFO] Querying vector store for: {query_text!r}")

#         self.embedder = EmbeddingPipeline()
#         query_embedding = self.embedder.generate_query_embedding(query_text)

#         return self.search(query_embedding, top_k=top_k)
        