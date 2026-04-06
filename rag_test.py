from retriever.search import ChromaRetriever

retriever = ChromaRetriever()

# TESTING DATA
retriever.collection.add(
    documents=["Diabetes is a chronic disease that affects blood sugar levels."],
    metadatas=[{"source": "test"}],
    ids=["1"]
)

# QUERY
results = retriever.query("What is diabetes?", top_k=3)

print("RESULTS:")
for r in results:
    print(r)

print("DONE")