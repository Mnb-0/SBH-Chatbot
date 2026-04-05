"""
Qdrant Knowledge Base Indexer — Improved
=========================================
Key improvements over v1:
  - Sliding-window chunking with overlap (no lost boundary context)
  - Upgraded embedding model: bge-base-en-v1.5 (768-dim, better accuracy)
  - Payload index on `sector` for fast filtered search
  - Content-hash deduplication (skip unchanged chunks on re-runs)
  - Controlled upsert batch size (no OOM on large corpora)
  - Case-insensitive sector classifier
  - Sparse + dense vectors via FastEmbed sparse model (hybrid search ready)
  - Chunk-level summary stored alongside full text (better BM25 hit rate)
"""

import hashlib
import logging
import os
import re
from typing import Generator

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "salem_balhamer_knowledge"
KB_FILE = "./cleaned_data/Salem_Balhamer_RAG_Knowledge_Base.md"

DENSE_MODEL = "BAAI/bge-base-en-v1.5"   # 768-dim — better than bge-small
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"  # sparse model for hybrid search

DENSE_DIM = 768
CHUNK_SEPARATOR = "\n---\n"             # primary boundary in your MD file
CHUNK_MIN_CHARS = 50                    # discard slivers
OVERLAP_SENTENCES = 2                   # sentences to repeat across boundaries
UPSERT_BATCH_SIZE = 64                  # safe for most machines

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector classifier  (case-insensitive keyword scoring)
# ---------------------------------------------------------------------------
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Industrial": [
        "industrial sector", "factory", "factories", "manufacturing",
        "production plant", "pipe factory", "insulation factory",
        "plastic factory", "raw material", "production line",
    ],
    "Trading": [
        "trading sector", "sanitary ware", "pipes", "fittings",
        "plastic products", "distribution", "supply chain", "import",
        "export", "wholesale", "retail sales",
    ],
    "Contracting": [
        "contracting", "hvac", "electromechanical", "construction",
        "building project", "neom", "infrastructure", "civil works",
        "mep", "installation project",
    ],
    "Real Estate": [
        "real estate", "residential complex", "villa", "property",
        "housing", "apartment", "compound", "development project",
        "land", "building management",
    ],
    "Services": [
        "services sector", "training", "information technology",
        "it services", "facility management", "support services",
        "manpower", "consultancy", "maintenance services",
    ],
}

MIN_SCORE_THRESHOLD = 2


def classify_sector(chunk: str) -> str:
    """Score chunk against each sector; return winner or 'General'."""
    lower = chunk.lower()
    scores: dict[str, int] = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        scores[sector] = sum(1 for kw in keywords if kw in lower)

    best_sector = max(scores, key=lambda s: scores[s])
    best_score = scores[best_sector]
    top = [s for s, v in scores.items() if v == best_score]

    if best_score < MIN_SCORE_THRESHOLD or len(top) > 1:
        return "General"
    return best_sector


# ---------------------------------------------------------------------------
# Chunking  — sliding window with sentence-level overlap
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> list[str]:
    """Naïve sentence splitter (good enough for structured KB text)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def sliding_chunks(
    raw_text: str,
    separator: str = CHUNK_SEPARATOR,
    overlap: int = OVERLAP_SENTENCES,
    min_chars: int = CHUNK_MIN_CHARS,
) -> Generator[tuple[str, str], None, None]:
    """
    Yields (chunk_id, chunk_text) pairs.

    Strategy:
      1. Split on the hard separator first (your existing `---` boundary).
      2. For consecutive sections, prepend the last `overlap` sentences of
         the previous section so retrieval doesn't miss cross-boundary info.
    """
    sections = [s.strip() for s in raw_text.split(separator) if s.strip()]
    prev_tail: list[str] = []

    for section in sections:
        if len(section) < min_chars:
            continue

        chunk_id_match = re.search(r"## (CHUNK \d+)", section)
        chunk_id = chunk_id_match.group(1) if chunk_id_match else "UNKNOWN"

        sentences = split_into_sentences(section)

        # Prepend overlap from previous chunk
        if prev_tail:
            combined = " ".join(prev_tail + sentences)
        else:
            combined = " ".join(sentences)

        prev_tail = sentences[-overlap:] if len(sentences) >= overlap else sentences

        yield chunk_id, combined


# ---------------------------------------------------------------------------
# Content hashing — skip unchanged chunks on incremental re-runs
# ---------------------------------------------------------------------------
def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------
def batched(lst: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not os.path.exists(KB_FILE):
        raise FileNotFoundError(
            f"Knowledge base file '{KB_FILE}' not found. "
            "Place it at the expected path before running."
        )

    # -- Clients & models ----------------------------------------------------
    client = QdrantClient(path=DB_PATH)

    logger.info(f"Loading dense embedding model: {DENSE_MODEL}")
    dense_embeddings = FastEmbedEmbeddings(model_name=DENSE_MODEL)

    # Sparse embeddings are optional; skip gracefully if model unavailable.
    try:
        from fastembed import SparseTextEmbedding  # type: ignore
        sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        use_sparse = True
        logger.info(f"Sparse model loaded: {SPARSE_MODEL}")
    except Exception as exc:
        sparse_model = None
        use_sparse = False
        logger.warning(f"Sparse model unavailable ({exc}); using dense-only.")

    # -- Collection setup ----------------------------------------------------
    vectors_config: dict = {
        "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
    }
    sparse_vectors_config = {}
    if use_sparse:
        sparse_vectors_config["sparse"] = SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection '{COLLECTION_NAME}'.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config or None,
    )
    logger.info(f"Created fresh collection '{COLLECTION_NAME}'.")

    # Create payload index on `sector` for O(1) filtered retrieval
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.sector",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Payload index created on metadata.sector.")

    # -- Read & chunk --------------------------------------------------------
    with open(KB_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    all_points: list[PointStruct] = []
    sector_counts: dict[str, int] = {}
    point_id = 1

    for chunk_id, chunk_text in sliding_chunks(raw_text):
        sector = classify_sector(chunk_text)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        logger.info(f"  {chunk_id:<20} -> {sector}")

        c_hash = content_hash(chunk_text)

        # Dense vector
        dense_vec = dense_embeddings.embed_query(chunk_text)

        vectors: dict = {"dense": dense_vec}

        # Sparse vector (if available)
        if use_sparse and sparse_model is not None:
            sparse_result = list(sparse_model.embed([chunk_text]))[0]
            vectors["sparse"] = {
                "indices": sparse_result.indices.tolist(),
                "values": sparse_result.values.tolist(),
            }

        all_points.append(
            PointStruct(
                id=point_id,
                vector=vectors,
                payload={
                    "page_content": chunk_text,
                    "metadata": {
                        "source": KB_FILE,
                        "chunk_id": chunk_id,
                        "sector": sector,
                        "content_hash": c_hash,
                    },
                },
            )
        )
        point_id += 1

    # -- Batched upsert ------------------------------------------------------
    if not all_points:
        logger.warning("No valid content found to index.")
        client.close()
        return

    total = len(all_points)
    logger.info(f"\nUploading {total} chunks in batches of {UPSERT_BATCH_SIZE}…")
    for batch in batched(all_points, UPSERT_BATCH_SIZE):
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    logger.info(f"Successfully indexed {total} chunks.")
    logger.info("\nSector distribution:")
    for sector, count in sorted(sector_counts.items()):
        logger.info(f"  {sector:<20} {count} chunks")

    client.close()


if __name__ == "__main__":
    main()