import ollama

class OllamaLLM:
    def __init__(self, model="gemma3:4b"):
        self.model = model
    
    def generate(self, prompt: str, context: str = "") -> str:
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": full_prompt}])
        return response["message"]["content"]