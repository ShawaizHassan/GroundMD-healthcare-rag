from tempfile import TemporaryDirectory

import chromadb

from retriever.search import ChromaRetriever

def main():
    with TemporaryDirectory(prefix="groundmd-retrieval-") as persist_dir:
        print("STARTING TEST...")

        client = chromadb.PersistentClient(path=persist_dir)
        client.get_or_create_collection("healthcare_docs")
        retriever = ChromaRetriever(persist_dir=persist_dir)

        # TESTING DATASETS
        documents = [
            "Diabetes is a chronic disease that affects blood sugar levels and requires insulin management.",
            "Hypertension is a condition where blood pressure is consistently too high and may cause headaches.",
            "Asthma is a respiratory condition treated with inhalers and avoiding environmental triggers.",
            "Heart disease refers to conditions affecting the heart, often caused by high cholesterol and lifestyle.",
            "Flu is a viral infection causing fever, cough, and body aches."
        ]

        retriever.collection.add(
            documents=documents,
            embeddings=[retriever.embedder.generate_query_embedding(doc) for doc in documents],
            metadatas=[{"source": f"doc{i}"} for i in range(len(documents))],
            ids=[str(i) for i in range(len(documents))]
        )

        # TESTING QUERIES
        queries = [
            "Explain diabetes briefly",
            "What causes high blood pressure?",
            "How do you treat asthma?",
            "Why do people get heart disease?"
        ]

        # RETRIEVAL
        for q in queries:
            print("\n==============================")
            print(f"QUERY: {q}")

            results = retriever.query(q, top_k=3)

            print("\n--- RESULTS ---")
            if not results:
                print("No results found")
            else:
                for i, r in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"Text: {r['document']}")
                    print(f"Score: {r['distance']}")
                    print(f"Source: {r['metadata'].get('source', 'unknown')}")


if __name__ == "__main__":
    main()
