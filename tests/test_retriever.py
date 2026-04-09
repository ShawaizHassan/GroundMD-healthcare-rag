import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from vectorstores.faiss_store import FaissVectorStore
from vectorstores.chroma_store import ChromaVectorStore
from retriever.search import ChromaRetriever
from ingestion.data_loader import DataLoader
from retriever.reranker import Reranker
from generator.prompt_builder import PromptBuilder



# Example usage
if __name__ == "__main__":
    # dataloader = DataLoader()
    # docs = dataloader.load_all_documents()
    # store = ChromaVectorStore()
    # store.build_from_documents(docs)
    chroma_retriever = ChromaRetriever()
    query = "What are the recommended HbA1c targets for patients with type 2 diabetes?"
    retrieved_docs = chroma_retriever.query(query, top_k=10)

    reranker = Reranker()
    prompt_builder = PromptBuilder()
    reranked_docs = reranker.rerank(query, retrieved_docs, top_k=1)
    context = prompt_builder.build_context(reranked_docs)
    prompt = prompt_builder.build_prompt(query, reranked_docs)
    
    os.makedirs("prompts", exist_ok=True)
    with open("prompts/rag_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    print(reranked_docs)