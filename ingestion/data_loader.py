from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


DISEASES_LIST = ["hiv", "dengue", "malaria", "diabetes", "tb"]


def get_disease_name(filename: str, disease_names: List[str]) -> str:
    filename_lower = filename.lower()

    for disease in disease_names:
        if disease in filename_lower:
            return disease

    return "unknown"


class DataLoader:
    def __init__(
        self,
        data_dir: str = r"C:\Users\PMLS\Desktop\IEDE\GroundMD-healthcare-rag\data\raw\guidelines",
    ) -> None:
        self.data_dir = Path(data_dir).resolve()

    def load_all_documents(self, data_dir: str | None = None) -> List[Document]:
        data_path = Path(data_dir).resolve() if data_dir else self.data_dir
        print(f"[INFO] Data path: {data_path}")

        all_documents: List[Document] = []

        pdf_files = list(data_path.glob("**/*.pdf"))
        print(f"[INFO] Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            print(f"[INFO] Loading PDF: {pdf_file}")

            disease_name = get_disease_name(pdf_file.name, DISEASES_LIST)

            try:
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()

                print(f"[INFO] Loaded {len(documents)} pages from {pdf_file.name}")

                for page_idx, doc in enumerate(documents):
                    metadata = dict(doc.metadata) if doc.metadata else {}

                    normalized_metadata = {
                        "source": str(pdf_file.resolve()),         # stable source
                        "source_file": pdf_file.name,              # human-readable citation
                        "page": metadata.get("page", page_idx),    # preserve loader page if available
                        "disease_name": disease_name,
                    }

                    doc.metadata = normalized_metadata

                all_documents.extend(documents)

            except Exception as e:
                print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

        print(f"[INFO] Total loaded documents/pages: {len(all_documents)}")
        return all_documents