"""
Qdrant DB Test Suite
=====================
Tests the indexed collection before connecting it to an LLM.
Covers: health, retrieval quality, sector distribution, edge cases,
        Arabic content detection, and thin-chunk diagnostics.

Usage:
  python test_db.py              # full suite
  python test_db.py --quick      # health + retrieval only
  python test_db.py --diagnose   # deep diagnostics (slow)
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, NamedVector

# ---------------------------------------------------------------------------
# Match your ingest config exactly
# ---------------------------------------------------------------------------
DB_PATH         = "./qdrant_db"
COLLECTION_NAME = "salem_balhamer_knowledge"
DENSE_MODEL     = "BAAI/bge-base-en-v1.5"
MODEL_CACHE     = str(Path.home() / ".cache" / "fastembed")

EXPECTED_SECTORS      = {"Industrial", "Trading", "Contracting", "Real Estate", "Services", "General"}
EXPECTED_CONTENT_TYPES = {"kb_section", "web_section", "job_listing"}
MIN_RETRIEVAL_SCORE   = 0.30   # below this = embedding not matching meaningfully
MIN_GOOD_SCORE        = 0.50   # above this = solid hit
ARABIC_RE             = re.compile(r'[\u0600-\u06FF]')
# ---------------------------------------------------------------------------

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
INFO  = "ℹ️ "

results: list[bool] = []

def check(label: str, passed: bool, detail: str = "", warn_only: bool = False) -> bool:
    icon = PASS if passed else (WARN if warn_only else FAIL)
    print(f"  {icon}  {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"       {line}")
    results.append(passed or warn_only)
    return passed


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# Load client + embeddings once
# ---------------------------------------------------------------------------
def setup():
    client = QdrantClient(path=DB_PATH)
    print(f"\n{INFO}  Loading embedding model from cache…")
    emb = FastEmbedEmbeddings(model_name=DENSE_MODEL, cache_dir=MODEL_CACHE)
    return client, emb


def vec(emb, text: str):
    return emb.embed_query(text)


def qdrant_search(client, collection_name, query_vector, limit=5,
                  with_payload=True, with_vectors=False, query_filter=None):
    """Fixed wrapper for Qdrant client >= 1.10"""
    if isinstance(query_vector, tuple):
        vector_name, vector_data = query_vector
    else:
        vector_name = None
        vector_data = query_vector

    response = client.query_points(
        collection_name=collection_name,
        query=vector_data,
        using=vector_name,
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors,
        query_filter=query_filter,
    )
    return response.points


# ---------------------------------------------------------------------------
# 1. Collection health
# ---------------------------------------------------------------------------
def test_health(client: QdrantClient) -> dict:
    section("1 · Collection Health")

    exists = client.collection_exists(COLLECTION_NAME)
    check("Collection exists", exists)
    if not exists:
        print("\n  Collection missing — did ingest.py complete successfully?")
        raise SystemExit(1)

    info   = client.get_collection(COLLECTION_NAME)
    count  = info.points_count
    check("Collection has points", count > 0, f"{count} points found")

    vcfg = info.config.params.vectors
    if isinstance(vcfg, dict):
        dense = vcfg.get("dense")
        check("Named 'dense' vector exists", dense is not None)
        check("Dense dim = 768", dense is not None and dense.size == 768,
              f"actual size = {dense.size if dense else 'n/a'}")
        has_sparse = "sparse" in (info.config.params.sparse_vectors or {})
        check("Sparse (hybrid) vectors present", has_sparse,
              "dense-only mode — sparse model may have failed at ingest time",
              warn_only=not has_sparse)
    else:
        check("Vector dim = 768", vcfg.size == 768, f"actual = {vcfg.size} (re-run ingest with new script?)")

    return {"point_count": count}


# ---------------------------------------------------------------------------
# 2. Payload integrity
# ---------------------------------------------------------------------------
def test_payload(client: QdrantClient) -> dict:
    section("2 · Payload Integrity")

    # Scroll all points (up to 2000 — should cover 333 easily)
    all_points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2000,
        with_payload=True,
        with_vectors=False,
    )

    missing_content = [p.id for p in all_points if not p.payload.get("page_content")]
    check("All points have page_content", not missing_content,
          f"Missing in IDs: {missing_content[:10]}" if missing_content else "")

    missing_meta = [p.id for p in all_points if not p.payload.get("metadata")]
    check("All points have metadata", not missing_meta,
          f"Missing in IDs: {missing_meta[:10]}" if missing_meta else "")

    required_meta = {"chunk_id", "sector", "content_type", "source_file"}
    bad_meta = [
        p.id for p in all_points
        if not required_meta.issubset((p.payload.get("metadata") or {}).keys())
    ]
    check("All metadata has required fields", not bad_meta,
          f"Incomplete metadata in IDs: {bad_meta[:10]}" if bad_meta else "")

    bad_sectors = [
        (p.id, p.payload["metadata"].get("sector"))
        for p in all_points
        if p.payload.get("metadata", {}).get("sector") not in EXPECTED_SECTORS
    ]
    check("All sector values are valid", not bad_sectors,
          f"Unknown sectors: {bad_sectors[:5]}" if bad_sectors else "")

    bad_types = [
        (p.id, p.payload["metadata"].get("content_type"))
        for p in all_points
        if p.payload.get("metadata", {}).get("content_type") not in EXPECTED_CONTENT_TYPES
    ]
    check("All content_type values are valid", not bad_types,
          f"Unknown types: {bad_types[:5]}" if bad_types else "")

    # Duplicate chunk_id check
    chunk_ids = [p.payload["metadata"].get("chunk_id", "") for p in all_points]
    dupes = [cid for cid, n in Counter(chunk_ids).items() if n > 1 and cid]
    check("No duplicate chunk_ids", not dupes,
          f"Duplicates ({len(dupes)}): {dupes[:5]}" if dupes else "")

    # Sector distribution
    sector_counts = Counter(
        p.payload["metadata"].get("sector", "MISSING") for p in all_points
    )
    general_pct = round(100 * sector_counts.get("General", 0) / len(all_points))
    dist_str = "  ".join(f"{s}={n}" for s, n in sorted(sector_counts.items()))
    check(
        f"General sector < 60% of chunks",
        general_pct < 60,
        f"Current: {general_pct}% General  |  {dist_str}\n"
        f"{'       → Add more domain keywords to SECTOR_KEYWORDS in ingest.py' if general_pct >= 60 else ''}",
        warn_only=general_pct >= 60,
    )

    # Thin chunks
    thin = [
        (p.payload["metadata"].get("source_file", "?"), len(p.payload.get("page_content", "")))
        for p in all_points
        if len(p.payload.get("page_content", "")) < 120
    ]
    check(
        f"Thin chunks (< 120 chars) < 10% of total",
        len(thin) / len(all_points) < 0.10,
        f"{len(thin)} thin chunks ({round(100*len(thin)/len(all_points))}%)\n"
        f"Samples: {thin[:5]}",
        warn_only=len(thin) / len(all_points) >= 0.10,
    )

    # Arabic content
    arabic_chunks = [
        p.payload["metadata"].get("source_file", "?")
        for p in all_points
        if ARABIC_RE.search(p.payload.get("page_content", ""))
    ]
    check(
        "No Arabic-heavy chunks (English-only model)",
        not arabic_chunks,
        f"{len(arabic_chunks)} chunks with Arabic text:\n" +
        "\n".join(f"  {f}" for f in set(arabic_chunks)) +
        "\n  → These will embed poorly. Exclude the file or use a multilingual model.",
        warn_only=bool(arabic_chunks),
    )

    return {"points": all_points, "sector_counts": sector_counts}


# ---------------------------------------------------------------------------
# 3. Semantic retrieval quality
# ---------------------------------------------------------------------------
def test_retrieval(client: QdrantClient, emb) -> None:
    section("3 · Semantic Retrieval Quality")

    queries = [
        # (query, expected_keyword_in_top_result, description)
        ("What sectors does Salem Balhamer Group operate in?",
         None, "broad company overview"),

        ("Tell me about Salem Balhamer's contracting and construction projects",
         "contracting", "contracting sector"),

        ("What plastic and industrial manufacturing does SBH do?",
         "industrial", "industrial/manufacturing"),

        ("Are there any job openings at Salem Balhamer?",
         "job", "job listings"),

        ("What is the history and background of Salem Balhamer?",
         None, "founder / about"),

        ("What real estate projects does the group have?",
         "real", "real estate"),

        ("How does SBH approach employee training and development?",
         None, "HR / careers"),

        ("What trading products does the group sell?",
         "trading", "trading sector"),
    ]

    score_total = 0.0
    for query, kw, desc in queries:
        try:
            hits = qdrant_search(client, COLLECTION_NAME,
                
                query_vector=("dense", vec(emb, query)),
                limit=5,
                with_payload=True,
            )
        except Exception:
            hits = qdrant_search(client, COLLECTION_NAME,
                
                query_vector=vec(emb, query),
                limit=5,
                with_payload=True,
            )

        if not hits:
            check(f"[{desc}]", False, "No results returned")
            continue

        top   = hits[0]
        score = top.score
        score_total += score
        content_preview = top.payload.get("page_content", "")[:120].replace("\n", " ")
        source = Path(top.payload.get("metadata", {}).get("source_file", "?")).name

        kw_found = kw is None or any(
            kw.lower() in h.payload.get("metadata", {}).get("sector", "").lower() or
            kw.lower() in h.payload.get("metadata", {}).get("content_type", "").lower() or
            kw.lower() in h.payload.get("page_content", "").lower()
            for h in hits
        )

        passed = score >= MIN_RETRIEVAL_SCORE and kw_found
        check(
            f"[{desc}]  score={score:.3f}",
            passed,
            f"Source : {source}\n"
            f"Preview: {content_preview}…\n"
            f"{'Keyword found in top-5 ✓' if kw_found else f'⚠ keyword \"{kw}\" not found in top-5'}",
            warn_only=score >= MIN_RETRIEVAL_SCORE and not kw_found,
        )

    avg = score_total / len(queries)
    check(
        f"Average retrieval score ≥ {MIN_GOOD_SCORE}",
        avg >= MIN_GOOD_SCORE,
        f"Average = {avg:.3f}  {'(good)' if avg >= MIN_GOOD_SCORE else '→ consider a stronger embedding model'}",
        warn_only=avg < MIN_GOOD_SCORE,
    )


# ---------------------------------------------------------------------------
# 4. Content-type filtering
# ---------------------------------------------------------------------------
def test_filters(client: QdrantClient, emb) -> None:
    section("4 · Filtered Search (content_type & sector)")

    for ct in ["web_section", "job_listing", "kb_section"]:
        try:
            hits = qdrant_search(client, COLLECTION_NAME,
                
                query_vector=("dense", vec(emb, "Salem Balhamer")),
                query_filter=Filter(must=[
                    FieldCondition(key="metadata.content_type", match=MatchValue(value=ct))
                ]),
                limit=5,
                with_payload=True,
            )
        except Exception:
            hits = qdrant_search(client, COLLECTION_NAME,
                
                query_vector=vec(emb, "Salem Balhamer"),
                query_filter=Filter(must=[
                    FieldCondition(key="metadata.content_type", match=MatchValue(value=ct))
                ]),
                limit=5,
                with_payload=True,
            )

        all_match = all(h.payload["metadata"].get("content_type") == ct for h in hits)
        check(
            f"Filter content_type='{ct}' returns correct type",
            all_match or not hits,
            f"{len(hits)} results" + (" (no chunks of this type indexed)" if not hits else ""),
            warn_only=not hits,
        )

    for sector in ["Contracting", "Industrial", "Trading"]:
        try:
            hits = qdrant_search(client, COLLECTION_NAME,
                
                query_vector=("dense", vec(emb, "company activities")),
                query_filter=Filter(must=[
                    FieldCondition(key="metadata.sector", match=MatchValue(value=sector))
                ]),
                limit=5,
                with_payload=True,
            )
        except Exception:
            hits = qdrant_search(client, COLLECTION_NAME,
                
                query_vector=vec(emb, "company activities"),
                query_filter=Filter(must=[
                    FieldCondition(key="metadata.sector", match=MatchValue(value=sector))
                ]),
                limit=5,
                with_payload=True,
            )

        all_match = all(h.payload["metadata"].get("sector") == sector for h in hits)
        check(
            f"Filter sector='{sector}' returns correct sector",
            all_match or not hits,
            f"{len(hits)} results" + (" (no chunks for this sector)" if not hits else ""),
            warn_only=not hits,
        )


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------
def test_edge_cases(client: QdrantClient, emb) -> None:
    section("5 · Edge Cases")

    # Gibberish query — should return something but with a low score
    hits = qdrant_search(client, COLLECTION_NAME,
        
        query_vector=("dense", vec(emb, "xqzptlwmfkjvbn")) if True else vec(emb, "xqzptlwmfkjvbn"),
        limit=3,
        with_payload=False,
    )
    top_score = hits[0].score if hits else 0
    check("Gibberish query returns low score (< 0.5)",
          top_score < 0.50,
          f"top score = {top_score:.3f}  {'(suspiciously high — check embedding model)' if top_score >= 0.5 else ''}",
          warn_only=top_score >= 0.50)

    # Very long query — should not crash
    long_q = "Salem Balhamer " * 60
    try:
        qdrant_search(client, COLLECTION_NAME,
            
            query_vector=("dense", vec(emb, long_q)),
            limit=3,
            with_payload=False,
        )
        check("Long query (960 tokens) does not crash", True)
    except Exception as e:
        check("Long query (960 tokens) does not crash", False, str(e))

    # Exact company name
    hits = qdrant_search(client, COLLECTION_NAME,
        
        query_vector=("dense", vec(emb, "Salem Balhamer Holding Group")),
        limit=1,
        with_payload=True,
    )
    check("'Salem Balhamer Holding Group' retrieves something",
          bool(hits) and hits[0].score > 0.40,
          f"score = {hits[0].score:.3f}" if hits else "no results")


# ---------------------------------------------------------------------------
# 6. Diagnostics (--diagnose flag)
# ---------------------------------------------------------------------------
def test_diagnostics(client: QdrantClient, points: list) -> None:
    section("6 · Diagnostics")

    # Files with 0 chunks (not in DB at all — detected by absence)
    sources_in_db = set(
        Path(p.payload["metadata"].get("source_file", "")).name
        for p in points
    )
    data_dir = Path("./cleaned_data")
    if data_dir.exists():
        all_files = set(f.name for f in data_dir.glob("**/*.md"))
        missing   = all_files - sources_in_db
        check(
            "All .md files have at least 1 chunk in DB",
            not missing,
            f"{len(missing)} files produced 0 chunks:\n" +
            "\n".join(f"  {f}" for f in sorted(missing)) +
            "\n  → Check CHUNK_MIN_CHARS or file content",
            warn_only=bool(missing),
        )
    else:
        print(f"  {INFO}  cleaned_data/ not found — skipping file coverage check")

    # Per-file chunk count distribution
    chunks_per_file: Counter = Counter(
        Path(p.payload["metadata"].get("source_file", "UNKNOWN")).name
        for p in points
    )
    single_chunk_files = [f for f, n in chunks_per_file.items() if n == 1]
    print(f"\n  {INFO}  Files with only 1 chunk ({len(single_chunk_files)} files):")
    for f in sorted(single_chunk_files)[:20]:
        print(f"       {f}")
    if len(single_chunk_files) > 20:
        print(f"       … and {len(single_chunk_files) - 20} more")

    # Shortest chunks
    shortest = sorted(
        [(len(p.payload.get("page_content", "")), p.payload["metadata"].get("source_file", "?"))
         for p in points],
        key=lambda x: x[0]
    )[:10]
    print(f"\n  {INFO}  10 shortest chunks:")
    for length, src in shortest:
        print(f"       {length:>5} chars  {Path(src).name}")

    # Sector breakdown per file type
    pub_sectors = Counter(
        p.payload["metadata"].get("sector")
        for p in points
        if "Publications" in p.payload.get("metadata", {}).get("source_file", "")
    )
    print(f"\n  {INFO}  Sector distribution for Publications files: {dict(pub_sectors)}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary() -> None:
    print(f"\n{'═'*55}")
    passed = sum(results)
    total  = len(results)
    pct    = round(100 * passed / total) if total else 0
    print(f"  Result: {passed}/{total} checks passed ({pct}%)")
    if pct == 100:
        print("  🎉 DB looks healthy — ready to connect to your LLM.\n")
    elif pct >= 80:
        print("  ⚠️  Minor issues above — review warnings before connecting LLM.\n")
    else:
        print("  ❌ Significant problems — fix and re-run ingest.py.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",    action="store_true", help="Health + retrieval only")
    parser.add_argument("--diagnose", action="store_true", help="Include deep diagnostics")
    args = parser.parse_args()

    print("\n" + "═"*55)
    print("  Qdrant DB Test Suite — Salem Balhamer Chatbot")
    print("═"*55)

    client, emb = setup()

    health_data  = test_health(client)
    payload_data = test_payload(client)

    if not args.quick:
        test_retrieval(client, emb)
        test_filters(client, emb)
        test_edge_cases(client, emb)

    if args.diagnose:
        test_diagnostics(client, payload_data["points"])

    summary()
    client.close()


if __name__ == "__main__":
    main()