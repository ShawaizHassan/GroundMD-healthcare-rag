from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any
from langchain_core.documents import Document

class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        spiltter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = [          "\n\n",
                                    "\n• ",
                                    "\n- ",
                                    "\n1. ",
                                    "\n2. ",
                                    "\n3. ",
                                    "\n",
                                    ". ",
                                    "; ",
                                    " ",
                                    ""
                                    ]
        )
                
        chunks = spiltter.split_documents(documents=documents)
        print(f"Chunked len{len(documents)} documents into {len(chunks)} chunks")
        
        return chunks