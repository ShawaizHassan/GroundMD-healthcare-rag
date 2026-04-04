from typing import List, Dict, Any


class PromptBuilder:
    def __init__(
        self,
        system_instruction: str | None = None,
        max_context_docs: int = 3
    ):
        self.max_context_docs = max_context_docs
        self.system_instruction = system_instruction or (
            "You are a healthcare RAG assistant.\n"
            "Answer the user's question using only the provided context.\n"
            "Do not use outside knowledge.\n"
            "If the answer is not present in the context, say: "
            "\"I could not find the answer in the provided documents.\"\n"
            "Keep the answer concise, accurate, and grounded in the sources.\n"
            "At the end, provide citations with source file name and page number."
        )

    def build_context(self, reranked_docs: List[Dict[str, Any]]) -> str:
        if not reranked_docs:
            return "No context available."

        context_blocks = []

        for i, doc in enumerate(reranked_docs[: self.max_context_docs], start=1):
            metadata = doc.get("metadata", {}) or {}
            source_file = metadata.get("source_file", "unknown")
            page = metadata.get("page", "unknown")
            disease_name = metadata.get("disease_name", "unknown")
            chunk_index = metadata.get("chunk_index", "unknown")
            text = doc.get("document", "").strip()
            rerank_score = doc.get("rerank_score", None)

            block = (
                f"[Source {i}]\n"
                f"Source File: {source_file}\n"
                f"Page: {page}\n"
                f"Disease: {disease_name}\n"
                f"Chunk Index: {chunk_index}\n"
            )

            if rerank_score is not None:
                block += f"Rerank Score: {rerank_score:.4f}\n"

            block += f"Content:\n{text}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def build_prompt(self, query: str, reranked_docs: List[Dict[str, Any]]) -> str:
        if not query.strip():
            raise ValueError("query cannot be empty")

        context = self.build_context(reranked_docs)

        prompt = (
            f"{self.system_instruction}\n\n"
            f"User Question:\n"
            f"{query}\n\n"
            f"Retrieved Context:\n"
            f"{context}\n\n"
            f"Instructions for answering:\n"
            f"1. Answer only from the retrieved context.\n"
            f"2. Do not invent facts.\n"
            f"3. If multiple sources provide relevant details, combine them carefully.\n"
            f"4. If the answer is missing or incomplete in the context, clearly say so.\n"
            f"5. End with citations in this format:\n"
            f"   - source_file, page X\n\n"
            f"Final Answer:"
        )

        return prompt

    def __call__(self, query: str, reranked_docs: List[Dict[str, Any]]) -> str:
        return self.build_prompt(query, reranked_docs)