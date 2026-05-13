from typing import List, Dict, Any
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.ollama_client import OllamaLLM


class Services:
    def __init__(self, llm=None):
        print('[INFO] Initializing Services')
        ollama_host = 'http://ollama:11434'
        ollama_model = 'qwen2.5:1.5b'

        if llm is None:
            print(f'[INFO] Connecting to Ollama at {ollama_host} with model {ollama_model}')
            self.llm = OllamaLLM(model=ollama_model, base_url=ollama_host)
        else:
            self.llm = llm
        print('[INFO] Services initialized successfully')

    def process_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        print('[INFO] process_query started')
        prompt = f'Medical question: {query}\n\nAnswer concisely:'
        
        try:
            llm_answer = self.llm.generate(prompt)
            return {
                'answer': llm_answer,
                'citations': [],
                'status': 'success'
            }
        except Exception as e:
            print(f'[ERROR] LLM generation failed: {e}')
            return {
                'answer': f'LLM generation failed: {e}',
                'citations': [],
                'status': 'error'
            }


def get_answerer(query: str) -> str:
    services = Services()
    result = services.process_query(query)
    return result.get('answer', 'No answer generated.')