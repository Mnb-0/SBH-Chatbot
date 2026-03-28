from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import warnings

# Suppress that annoying requests/urllib3 warning at runtime
warnings.filterwarnings("ignore", category=UserWarning, module='requests')

# Initialize Client & Embeddings
client = QdrantClient(path="./qdrant_db_clean")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_context(query, limit=5):
    """
    Retrieves unique, high-quality chunks from Qdrant.
    """
    vector = embeddings.embed_query(query)
    
    # We pull more than we need (limit=5) to account for duplicates 
    # we might filter out in the next step.
    results = client.query_points(
        collection_name="salem_balhamer_knowledge",
        query=vector,
        limit=limit
    ).points

    unique_chunks = []
    seen_content = set()

    for hit in results:
        content = hit.payload.get("page_content", "").strip()
        
        # 1. De-duplication Logic
        # We normalize the string to check if we've seen this exact text before
        normalized_content = " ".join(content.split()) 
        if normalized_content in seen_content:
            continue
        
        # 2. Heuristic Noise Filter
        # If the chunk is mostly "+ 0" or placeholders, we skip it
        if "Projects + 0" in content or content.count('!') > 5:
            continue

        unique_chunks.append(content)
        seen_content.add(normalized_content)

    # Return only the top 3 unique matches
    return "\n\n---\n\n".join(unique_chunks[:3])

# --- Execution ---
query = "What products does the industrial sector produce?"
context_for_llm = get_context(query)

print("--- CLEANED CONTEXT FOR LLM ---")
print(context_for_llm)

# Cleanup
client.close()