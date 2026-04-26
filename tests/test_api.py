import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.service import Services

if __name__ == "__main__":
    service = Services()
    res = service.process_query("What are the recommended HbA1c targets for patients with type 2 diabetes?", top_k=3)
    print(res)