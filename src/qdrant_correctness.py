from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import sys

# Suppress the messy warnings if you can't update immediately
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='requests')

try:
    client = QdrantClient(path="./qdrant_db_clean")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 1. Check Collection Info
    info = client.get_collection("salem_balhamer_knowledge")
    print(f"📊 Total Points in DB: {info.points_count}")

    # 2. Perform a Raw Search
    query = "Tell me about the industrial sector"
    results = client.query_points(
        collection_name="salem_balhamer_knowledge",
        query=embeddings.embed_query(query),
        limit=3
    )

    for i, hit in enumerate(results.points):
        payload = hit.payload or {}
        # Adjusting to how LangChain usually stores metadata in Qdrant
        metadata = payload.get("metadata") or {}
        source_file = metadata.get("source") or "Unknown source"
        page_content = payload.get("page_content") or ""

        print(f"\n--- Match {i+1} (Score: {hit.score:.4f}) ---")
        print(f"Source: {source_file}")
        print(f"Text Snippet: {page_content[:200].strip()}...")

finally:
    # Explicitly close to prevent the msvcrt shutdown error
    if 'client' in locals():
        client.close()