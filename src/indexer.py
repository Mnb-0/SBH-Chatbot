import os
import re
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "salem_balhamer_knowledge"
KB_FILE = "./data/Salem_Balhamer_RAG_Knowledge_Base.md"

# FIX: HuggingFace token moved out of source code.
# Set the HF_TOKEN environment variable before running this script:
#   export HF_TOKEN=your_token_here   (Linux/Mac)
#   set HF_TOKEN=your_token_here      (Windows)
#hf_token = os.environ.get("HF_TOKEN")
#if not hf_token:
#    raise EnvironmentError(
#        "HF_TOKEN environment variable is not set. "
#        "Export it before running: export HF_TOKEN=your_token_here"
#    )
#os.environ["HF_TOKEN"] = hf_token


hf_token = ""

# FIX: Sector classification now uses a keyword scoring system.
# Each sector accumulates a score based on how many of its keywords appear in the chunk.
# The sector with the highest score wins, preventing a single passing mention
# (e.g. "contracting partners") from misclassifying an otherwise General chunk.
# A minimum score threshold ensures low-confidence matches fall back to General.
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Industrial": [
        "INDUSTRIAL SECTOR", "FACTORY", "FACTORIES", "MANUFACTURING",
        "PRODUCTION PLANT", "PIPE FACTORY", "INSULATION FACTORY",
        "PLASTIC FACTORY", "RAW MATERIAL", "PRODUCTION LINE",
    ],
    "Trading": [
        "TRADING SECTOR", "SANITARY WARE", "PIPES", "FITTINGS",
        "PLASTIC PRODUCTS", "DISTRIBUTION", "SUPPLY CHAIN", "IMPORT",
        "EXPORT", "WHOLESALE", "RETAIL SALES",
    ],
    "Contracting": [
        "CONTRACTING", "HVAC", "ELECTROMECHANICAL", "CONSTRUCTION",
        "BUILDING PROJECT", "NEOM", "INFRASTRUCTURE", "CIVIL WORKS",
        "MEP", "INSTALLATION PROJECT",
    ],
    "Real Estate": [
        "REAL ESTATE", "RESIDENTIAL COMPLEX", "VILLA", "PROPERTY",
        "HOUSING", "APARTMENT", "COMPOUND", "DEVELOPMENT PROJECT",
        "LAND", "BUILDING MANAGEMENT",
    ],
    "Services": [
        "SERVICES SECTOR", "TRAINING", "INFORMATION TECHNOLOGY",
        "IT SERVICES", "FACILITY MANAGEMENT", "SUPPORT SERVICES",
        "MANPOWER", "CONSULTANCY", "MAINTENANCE SERVICES",
    ],
}

MIN_SCORE_THRESHOLD = 2


def classify_sector(chunk: str) -> str:
    """
    Scores a chunk against each sector's keyword list.
    Returns the highest-scoring sector, or 'General' if no sector
    meets the minimum threshold or if there is a tie.
    """
    chunk_upper = chunk.upper()
    scores: dict[str, int] = {sector: 0 for sector in SECTOR_KEYWORDS}

    for sector, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in chunk_upper:
                scores[sector] += 1

    best_sector = max(scores, key=lambda s: scores[s])
    best_score = scores[best_sector]

    # Check for a tie between two or more sectors
    top_scores = [s for s, score in scores.items() if score == best_score]

    if best_score < MIN_SCORE_THRESHOLD or len(top_scores) > 1:
        return "General"

    return best_sector


# --- Validate KB file exists before doing any heavy work ---
if not os.path.exists(KB_FILE):
    raise FileNotFoundError(
        f"Knowledge base file '{KB_FILE}' not found. "
        "Ensure the file is placed at the expected path before running."
    )

# --- Initialize Client & Embeddings ---
client = QdrantClient(path=DB_PATH)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# --- Nuke and recreate collection ---
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    logger.info(f"Deleted existing collection '{COLLECTION_NAME}'.")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
logger.info(f"Created fresh collection '{COLLECTION_NAME}'.")

# --- Process the Master File ---
with open(KB_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

raw_chunks = raw_text.split("\n---\n")

all_points = []
point_id = 1
sector_counts: dict[str, int] = {}

for chunk in raw_chunks:
    chunk = chunk.strip()
    if len(chunk) < 50:
        continue

    chunk_id_match = re.search(r"## (CHUNK \d+)", chunk)
    chunk_id = chunk_id_match.group(1) if chunk_id_match else f"UNKNOWN_{point_id}"

    sector = classify_sector(chunk)

    # Audit log so you can verify classifications after running
    sector_counts[sector] = sector_counts.get(sector, 0) + 1
    logger.info(f"  {chunk_id:<20} -> {sector}")

    vector = embeddings.embed_query(chunk)

    all_points.append(PointStruct(
        id=point_id,
        vector=vector,
        payload={
            "page_content": chunk,
            "metadata": {
                "source": KB_FILE,
                "chunk_id": chunk_id,
                "sector": sector,
            },
        },
    ))
    point_id += 1

# --- Batch Upload ---
if all_points:
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    logger.info(f"\nSuccessfully indexed {len(all_points)} chunks.")
    logger.info("\nSector distribution:")
    for sector, count in sorted(sector_counts.items()):
        logger.info(f"  {sector:<20} {count} chunks")
else:
    logger.warning("No valid content found to index.")

client.close()
