import os
import requests


class OllamaLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 250
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "phi3")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/backend/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()