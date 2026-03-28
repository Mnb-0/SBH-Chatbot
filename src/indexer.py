import os
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
DB_PATH = "./qdrant_db_clean"
COLLECTION_NAME = "salem_balhamer_knowledge"
DATA_DIR = "./clean_data"

def final_clean(text):
    """
    The 'Final Boss' cleaner: Strips headers that are just single links 
    and kills specific navigation patterns that survived earlier passes.
    """
    # 1. Remove lines that are just Markdown headers containing a single link (Navigation/Menu style)
    # Example: ##### [Industrial Facilities](https://...)
    text = re.sub(r'^#+\s*\[.*?\]\(.*?\)\s*$', '', text, flags=re.MULTILINE)
    
    # 2. Kill repetitive breadcrumb-style separators if they exist
    text = re.sub(r'»', '', text)
    
    # 3. Collapse excessive whitespace again just in case
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# Initialize Client & Embeddings
client = QdrantClient(path=DB_PATH)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Splitter: 500 chars is good for bge-small to keep context dense
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# 1. Nuke and recreate collection
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 2. Process Files
all_points = []
point_id = 1

if not os.path.exists(DATA_DIR):
    print(f"Error: {DATA_DIR} does not exist.")
    exit()

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".md"):
        continue
        
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Apply the last-minute aggressive cleaning
    clean_text = final_clean(raw_text)
    
    if not clean_text:
        continue

    # Split into chunks
    chunks = text_splitter.split_text(clean_text)

    for i, chunk_text in enumerate(chunks):
        if len(chunk_text.strip()) < 30: # Slightly higher threshold for junk
            continue

        vector = embeddings.embed_query(chunk_text)
        
        all_points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "page_content": chunk_text,
                "metadata": {
                    "source": filename,
                    "chunk_id": i
                }
            }
        ))
        point_id += 1

# 3. Batch Upload
if all_points:
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"✅ Successfully indexed {len(all_points)} chunks.")
else:
    print("⚠️ No valid content found to index.")

# 4. Explicit close to prevent the msvcrt import error
client.close()