from typing import List, Dict, Any
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.search import ChromaRetriever
from retriever.reranker import Reranker
from generator.prompt_builder import PromptBuilder


class Services:
    def __init__(self, llm=None):
        print("[INFO] Initializing Services")
        self.chroma_retriever = ChromaRetriever()
        self.reranker = Reranker()
        self.prompt_builder = PromptBuilder()
        self.llm = llm
        print("[INFO] Services initialized successfully")

    def process_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        print("[INFO] process_query started")

        print("[INFO] Step 1: calling retriever.query()")
        retrieved_docs = self.chroma_retriever.query(query, top_k=10)
        print(f"[INFO] Step 1 done: retrieved {len(retrieved_docs)} docs")

        print("[INFO] Step 2: calling reranker.rerank()")
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k)
        print(f"[INFO] Step 2 done: reranked to {len(reranked_docs)} docs")

        print("[INFO] Step 3: building prompt")
        prompt = self.prompt_builder.build_prompt(query, reranked_docs=reranked_docs)
        print("[INFO] Step 3 done: prompt built successfully")

        if self.llm is None:
            print("[INFO] No LLM connected, returning dummy response")
            return {
                "answer": "LLM is not connected yet.",
                "citations": self._extract_citations(reranked_docs),
                "status": "success"
            }

        print("[INFO] Step 4: calling llm.generate()")
        llm_answer = self.llm.generate(prompt)
        print("[INFO] Step 4 done: LLM response generated")

        return {
            "answer": llm_answer,
            "citations": self._extract_citations(reranked_docs),
            "status": "success"
        }

    def _extract_citations(self, docs: List[Dict[str, Any]]) -> List[str]:
        citations = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "unknown")
            page = metadata.get("page", "unknown")
            disease = metadata.get("disease_name", "unknown")
            relevance = doc.get("rerank_score", "unknown")
            citations.append(f"[Source: {source}, page {page}]\n Disease: {disease} \n Relevance: {relevance}")
        return citations