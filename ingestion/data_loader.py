from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []
    
    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] loading: {pdf_file}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[DEBUG] got an error: {e} while loading PDF {pdf_file}")
            
    return documents