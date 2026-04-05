"""
Bulk MD Ingestion Pipeline (pre-cleaned files)
===============================================
Assumes all 221 .md files are already clean.
The only special case is the structured KB file — everything else
is treated as a clean web page and chunked on markdown headings.

Usage:
  python ingest.py                        # index new/changed files only
  python ingest.py --force                # re-index everything
  python ingest.py --dry-run              # preview without writing
  python ingest.py --file path/to/x.md   # single file
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# Hugging Face auth — must happen before any fastembed/huggingface import
# Set HF_TOKEN in your environment:  export HF_TOKEN=hf_xxxxxxxxxxxx
# ---------------------------------------------------------------------------
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token   # used by huggingface_hub
    os.environ["HF_TOKEN"] = _hf_token                  # used by fastembed >= 0.3
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
    except ImportError:
        pass  # huggingface_hub not installed separately — env vars are enough
else:
    import warnings
    warnings.warn(
        "HF_TOKEN not set. Model downloads may be rate-limited by Hugging Face. "
        "Set it with:  export HF_TOKEN=hf_xxxxxxxxxxxx",
        stacklevel=1,
    )

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR        = Path("./cleaned_data")
DB_PATH         = "./qdrant_db"
COLLECTION_NAME = "salem_balhamer_knowledge"
MANIFEST_PATH   = Path("./cleaned_data/.ingest_manifest.json")
KB_FILENAME     = "Salem_Balhamer_RAG_Knowledge_Base.md"

DENSE_MODEL     = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL    = "prithivida/Splade_PP_en_v1"
DENSE_DIM       = 768

CHUNK_MIN_CHARS   = 80
OVERLAP_SENTENCES = 2
UPSERT_BATCH_SIZE = 64
EMBED_WORKERS     = 4
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector classifier
# ---------------------------------------------------------------------------
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Industrial":  ["industrial sector","factory","factories","manufacturing","production plant","pipe factory","insulation factory","plastic factory","raw material","production line"],
    "Trading":     ["trading sector","sanitary ware","pipes","fittings","plastic products","distribution","supply chain","import","export","wholesale","retail sales"],
    "Contracting": ["contracting","hvac","electromechanical","construction","building project","neom","infrastructure","civil works","mep","installation project"],
    "Real Estate": ["real estate","residential complex","villa","property","housing","apartment","compound","development project","land","building management"],
    "Services":    ["services sector","training","information technology","it services","facility management","support services","manpower","consultancy","maintenance services"],
}
MIN_SCORE = 2

def classify_sector(text: str) -> str:
    lower = text.lower()
    scores = {s: sum(1 for kw in kws if kw in lower) for s, kws in SECTOR_KEYWORDS.items()}
    best = max(scores, key=lambda s: scores[s])
    top  = [s for s, v in scores.items() if v == scores[best]]
    return best if scores[best] >= MIN_SCORE and len(top) == 1 else "General"


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id:      str
    text:          str
    source_file:   str
    section_title: str
    sector:        str
    content_type:  str          # "kb_section" | "web_section" | "job_listing"
    extra_payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
JOB_ROW_RE = re.compile(
    r"\[([^\]]+)\]\((https?://[^\)]+)\)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^\n|]+)"
)

def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def chunk_kb_file(text: str, source: str) -> list[Chunk]:
    """Structured KB: split on --- boundaries, carry sentence overlap."""
    chunks: list[Chunk] = []
    prev_tail: list[str] = []

    for raw in text.split("\n---\n"):
        raw = raw.strip()
        if len(raw) < CHUNK_MIN_CHARS:
            continue

        m = re.search(r"## (CHUNK \d+)", raw)
        chunk_id = m.group(1) if m else f"KB_UNKNOWN_{len(chunks)}"

        sentences = _sentences(raw)
        combined  = " ".join(prev_tail + sentences) if prev_tail else " ".join(sentences)
        prev_tail = sentences[-OVERLAP_SENTENCES:] if len(sentences) >= OVERLAP_SENTENCES else sentences

        chunks.append(Chunk(
            chunk_id=chunk_id, text=combined, source_file=source,
            section_title=chunk_id, sector=classify_sector(combined),
            content_type="kb_section",
        ))
    return chunks


def chunk_web_page(text: str, source: str) -> list[Chunk]:
    """
    Clean web page: extract job rows first, then split on headings.
    Falls back to paragraph splitting if no headings exist.
    """
    chunks: list[Chunk] = []
    stem = Path(source).stem

    # ── Job table rows → structured chunks ──────────────────────────────────
    job_matches = list(JOB_ROW_RE.finditer(text))
    for i, m in enumerate(job_matches):
        title, url, location, company, date = (g.strip() for g in m.groups())
        synth = f"Job opening: {title} at {company}, located in {location}. Posted on {date}."
        chunks.append(Chunk(
            chunk_id=f"{stem}_job_{i:03d}", text=synth, source_file=source,
            section_title="Job Listing", sector="Services",
            content_type="job_listing",
            extra_payload={"job_title": title, "location": location,
                           "company": company, "date_posted": date, "url": url},
        ))

    # Remove table so it isn't re-chunked as narrative
    if job_matches:
        text = text[: job_matches[0].start()].strip()

    # ── Heading-based narrative sections ────────────────────────────────────
    parts = HEADING_RE.split(text)
    # parts layout: [pre-heading, level, title, body, level, title, body …]

    preamble = parts[0].strip()
    if len(preamble) >= CHUNK_MIN_CHARS:
        chunks.append(Chunk(
            chunk_id=f"{stem}_preamble", text=preamble, source_file=source,
            section_title="Introduction", sector=classify_sector(preamble),
            content_type="web_section",
        ))

    sec_idx, i = 0, 1
    while i + 2 < len(parts):
        title = parts[i + 1].strip()
        body  = parts[i + 2].strip()
        i += 3
        if len(body) < CHUNK_MIN_CHARS:
            continue
        full = f"{title}\n\n{body}"
        chunks.append(Chunk(
            chunk_id=f"{stem}_sec_{sec_idx:03d}", text=full, source_file=source,
            section_title=title, sector=classify_sector(full),
            content_type="web_section",
        ))
        sec_idx += 1

    # ── Fallback: no headings → paragraph split ──────────────────────────────
    if sec_idx == 0 and not preamble:
        for idx, para in enumerate(re.split(r"\n{2,}", text)):
            para = para.strip()
            if len(para) >= CHUNK_MIN_CHARS:
                chunks.append(Chunk(
                    chunk_id=f"{stem}_para_{idx:03d}", text=para, source_file=source,
                    section_title="General", sector=classify_sector(para),
                    content_type="web_section",
                ))

    return chunks


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def load_manifest() -> dict[str, str]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}

def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed_chunk(chunk: Chunk, emb: FastEmbedEmbeddings) -> tuple[Chunk, list[float]]:
    return chunk, emb.embed_query(chunk.text)

def batched(lst: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# Qdrant setup
# ---------------------------------------------------------------------------
def build_collection(client: QdrantClient, use_sparse: bool) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config=(
            {"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
            if use_sparse else None
        ),
    )
    for field_name in ["metadata.sector", "metadata.content_type", "metadata.source_file"]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    logger.info("Collection and payload indexes ready.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    t0 = time.time()

    md_files = [Path(args.file)] if args.file else sorted(DATA_DIR.glob("**/*.md"))
    if not md_files:
        logger.error(f"No .md files found in {DATA_DIR}")
        return
    logger.info(f"Found {len(md_files)} file(s).")

    manifest   = {} if args.force else load_manifest()
    to_process = [f for f in md_files if manifest.get(str(f)) != file_hash(f)]
    skipped    = len(md_files) - len(to_process)
    logger.info(f"Skipping {skipped} unchanged  |  Processing {len(to_process)}")

    if args.dry_run:
        [logger.info(f"  {f}") for f in to_process]
        return
    if not to_process:
        logger.info("Nothing to index. Use --force to re-index everything.")
        return

    # ── Chunk ────────────────────────────────────────────────────────────────
    all_chunks: list[Chunk] = []
    for path in to_process:
        raw    = path.read_text(encoding="utf-8", errors="replace")
        is_kb  = path.name == KB_FILENAME
        chunks = chunk_kb_file(raw, str(path)) if is_kb else chunk_web_page(raw, str(path))
        logger.info(f"  [{'KB ' if is_kb else 'WEB'}] {path.name:<50} {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")

    # ── Models ───────────────────────────────────────────────────────────────
    # cache_dir keeps weights on disk — subsequent runs skip the download entirely
    model_cache = str(Path.home() / ".cache" / "fastembed")
    logger.info(f"Loading dense model: {DENSE_MODEL}  (cache: {model_cache})")
    embeddings = FastEmbedEmbeddings(
        model_name=DENSE_MODEL,
        cache_dir=model_cache,
    )

    sparse_model, use_sparse = None, False
    try:
        from fastembed import SparseTextEmbedding
        sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL,
            cache_dir=model_cache,
        )
        use_sparse = True
        logger.info(f"Sparse model loaded: {SPARSE_MODEL}")
    except Exception as e:
        logger.warning(f"Sparse model unavailable ({e}) — dense only.")

    # ── Qdrant ───────────────────────────────────────────────────────────────
    client = QdrantClient(path=DB_PATH)
    build_collection(client, use_sparse)

    # ── Parallel embedding ───────────────────────────────────────────────────
    logger.info(f"Embedding with {EMBED_WORKERS} workers…")
    embedded: list[tuple[Chunk, list[float]]] = [None] * len(all_chunks)  # type: ignore
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {pool.submit(embed_chunk, c, embeddings): i for i, c in enumerate(all_chunks)}
        done = 0
        for future in as_completed(futures):
            embedded[futures[future]] = future.result()
            done += 1
            if done % 50 == 0 or done == len(all_chunks):
                logger.info(f"  {done}/{len(all_chunks)}")

    # ── Points ───────────────────────────────────────────────────────────────
    points: list[PointStruct] = []
    for pid, (chunk, dense_vec) in enumerate(embedded, 1):
        vectors: dict = {"dense": dense_vec}
        if use_sparse and sparse_model:
            sr = list(sparse_model.embed([chunk.text]))[0]
            vectors["sparse"] = {"indices": sr.indices.tolist(), "values": sr.values.tolist()}

        points.append(PointStruct(
            id=pid,
            vector=vectors,
            payload={
                "page_content": chunk.text,
                "metadata": {
                    "source_file":   chunk.source_file,
                    "chunk_id":      chunk.chunk_id,
                    "sector":        chunk.sector,
                    "section_title": chunk.section_title,
                    "content_type":  chunk.content_type,
                    "content_hash":  hashlib.sha256(chunk.text.encode()).hexdigest()[:16],
                    **chunk.extra_payload,
                },
            },
        ))

    # ── Upsert ───────────────────────────────────────────────────────────────
    logger.info(f"Uploading {len(points)} points…")
    for batch in batched(points, UPSERT_BATCH_SIZE):
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    client.close()

    # ── Manifest + stats ─────────────────────────────────────────────────────
    for path in to_process:
        manifest[str(path)] = file_hash(path)
    save_manifest(manifest)

    sector_counts: dict[str, int] = {}
    type_counts:   dict[str, int] = {}
    for chunk, _ in embedded:
        sector_counts[chunk.sector]     = sector_counts.get(chunk.sector, 0) + 1
        type_counts[chunk.content_type] = type_counts.get(chunk.content_type, 0) + 1

    logger.info(f"\nFinished in {time.time() - t0:.1f}s — {len(points)} points indexed.\n")
    logger.info("Content types:"); [logger.info(f"  {k:<20} {v}") for k, v in sorted(type_counts.items())]
    logger.info("Sectors:");       [logger.info(f"  {k:<20} {v}") for k, v in sorted(sector_counts.items())]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force",   action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--file",    type=str, default=None)
    run(p.parse_args())