from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
diseases_list = ["hiv", "dengue", "malaria", "diabetes", "tb"]
def get_disease_name(filename, disease_names):
    filename_lower = filename.lower()
    
    for disease in disease_names:
        if disease in filename_lower:
            return disease
    return "unknown"
class DataLoader:
    def __init__(self, data_dir = r"C:\Users\PMLS\Desktop\IEDE\GroundMD-healthcare-rag\data\sample_test"):
        self.data_dir = data_dir
    
    def load_all_documents(self, data_dir = None) -> List[Document]:
        """
        Load all supported files from the data directory and convert to LangChain document structure
        Supported: PDF, TXT, CSV, Excel, Word, JSON
        """
        # use project root data folder
        data_path = Path(data_dir or self.data_dir).resolve()
        print(f"[DEBUG] Data path: {data_path}")
        all_documents = []
        
        # PDF files
        pdf_files = list(data_path.glob('**/*.pdf'))
        print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
        for pdf_file in pdf_files:
            print(f"[DEBUG] loading: {pdf_file}")
            disease_name = get_disease_name(pdf_file.name, diseases_list)
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()
                print(f"[DEBUG] Loaded {len(documents)} PDF docs from {pdf_file}")
                for doc in documents:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['disease_name'] = disease_name
                all_documents.extend(documents)
            except Exception as e:
                print(f"[DEBUG] got an error: {e} while loading PDF {pdf_file}")
                
        return all_documents