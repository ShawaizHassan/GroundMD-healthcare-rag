import requests


class OllamaLLM:
    def __init__(
        self,
        model: str = "phi3",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 150
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()