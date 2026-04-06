from retriever.search import ChromaRetriever

queries = [
    "What is diabetes?",
    "Symptoms of hypertension",
    "Treatment for asthma",
    "Causes of heart disease"
]

chroma = ChromaRetriever()

for q in queries:
    print("\n==============================")
    print(f"QUERY: {q}")

    results = chroma.query(q, top_k=3)

    print("\n--- CHROMA RESULTS ---")
    if not results:
        print("No results found")
    else:
        for i, r in enumerate(results):
            print(f"\nResult {i+1}:")
            print(r["document"][:200])