import ollama

class OllamaLLM:
    def __init__(self, model="qwen2.5:1.5b", base_url=None):
        self.model = model
        self.client = ollama.Client(host=base_url) if base_url else ollama

    def generate(self, prompt: str, context: str = "") -> str:
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response["message"]["content"]