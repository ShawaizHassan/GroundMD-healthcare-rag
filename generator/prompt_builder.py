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
            "Answer the user's question using only the retrieved context.\n"
            "Do not use outside knowledge.\n"
            "If the answer is not clearly present in the context, say: "
            "\"I could not find the answer in the provided documents.\"\n"
            "Keep the answer concise, factual, and grounded in the sources.\n"
            "End the answer with citations using source file name and page number."
        )

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.split())

    def build_context(self, reranked_docs: List[Dict[str, Any]]) -> str:
        if not reranked_docs:
            return "No retrieved context available."

        context_blocks = []

        for i, doc in enumerate(reranked_docs[:self.max_context_docs], start=1):
            metadata = doc.get("metadata", {}) or {}
            source_file = metadata.get("source_file", "unknown")
            page = metadata.get("page", "unknown")
            text = self._clean_text(doc.get("document", ""))

            if not text:
                continue

            block = (
                f"[Source {i}]\n"
                f"File: {source_file}\n"
                f"Page: {page}\n"
                f"Content: {text}"
            )
            context_blocks.append(block)

        if not context_blocks:
            return "No retrieved context available."

        return "\n\n".join(context_blocks)

    def build_prompt(self, query: str, reranked_docs: List[Dict[str, Any]]) -> str:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        context = self.build_context(reranked_docs)

        prompt = (
            f"{self.system_instruction}\n\n"
            f"Rules:\n"
            f"- Use only the context below.\n"
            f"- Do not add facts from memory.\n"
            f"- Prefer the most directly relevant source.\n"
            f"- If multiple sources agree, combine carefully.\n"
            f"- If the context is insufficient, say so clearly.\n\n"
            f"Question:\n"
            f"{query.strip()}\n\n"
            f"Context:\n"
            f"{context}\n\n"
            f"Return the answer in exactly this format:\n"
            f"Answer:\n"
            f"<your answer>\n\n"
            f"Citations:\n"
            f"- [source_file, page X]\n"
        )

        return prompt

    def __call__(self, query: str, reranked_docs: List[Dict[str, Any]]) -> str:
        return self.build_prompt(query, reranked_docs)