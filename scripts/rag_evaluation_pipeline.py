"""
===============================================================================
SALEM BALHAMER RAG CHATBOT — EVALUATION PIPELINE (GROQ API)
===============================================================================
Requires:
  pip install groq chromadb sentence-transformers pandas

Set environment variables before running:
  export GROQ_API_KEY=your_key_here   (Linux/Mac)
  set GROQ_API_KEY=your_key_here      (Windows)
===============================================================================
"""

import json
import os
import time
import pandas as pd
from datetime import datetime
from collections import Counter
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

# FIX: API key moved out of source code. Set GROQ_API_KEY as an environment variable.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set.\n"
        "  Linux/Mac: export GROQ_API_KEY=your_key_here\n"
        "  Windows:   set GROQ_API_KEY=your_key_here"
    )

# ---- FILE PATHS ----
KNOWLEDGE_BASE_PATH = "Salem_Balhamer_RAG_Knowledge_Base.md"
EVAL_SET_PATH       = "src/Salem_Balhamer_RAG_Evaluation_Set.json"
RESULTS_OUTPUT_PATH = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ---- RAG CONFIGURATION ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SEPARATOR = "\n---\n"

# FIX: TOP_K raised from 3 to 5. The production system retrieves 8 chunks;
# evaluating with only 3 under-represents what the chatbot actually sees,
# making retrieval scores artificially low and context scores misleading.
TOP_K = 5

# FIX: Minimum chunk length raised from 100 to 200 to match the production
# indexer's effective threshold (indexer skips chunks < 50 chars but in practice
# meaningful chunks are much longer). Tiny fragments skew retrieval quality scores.
MIN_CHUNK_LENGTH = 200

# ---- MODEL TAGS ----
CHATBOT_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL   = "llama-3.3-70b-versatile"

# ---- EVALUATION CONFIG ----
MAX_EVAL_QUESTIONS = None   # Set to an int (e.g. 10) to limit for quick testing

# FIX: Rate limit delay added. Groq's free tier throttles at ~30 req/min.
# Without a delay, back-to-back judge calls hit 429 errors mid-eval,
# corrupting scores for the affected questions silently.
INTER_REQUEST_DELAY_SECONDS = 2.0


# ============================================================================
# SECTION 2: GROQ CLIENT & HEALTH CHECK
# ============================================================================

groq_client = Groq(api_key=GROQ_API_KEY)


def check_groq_ready() -> None:
    """Verify Groq API key is valid and the models are reachable."""
    print("=" * 60)
    print("PRE-FLIGHT: Checking Groq API connectivity")
    print("=" * 60)

    try:
        response = groq_client.chat.completions.create(
            model=CHATBOT_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
            temperature=0.0,
        )
        reply = response.choices[0].message.content.strip()
        print(f"  Groq API reachable ✅  |  Chatbot model response: '{reply}'")
        print(f"  Chatbot model : {CHATBOT_MODEL}")
        print(f"  Judge model   : {JUDGE_MODEL}")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Groq API. Check your GROQ_API_KEY.\n  Error: {e}"
        )


# ============================================================================
# SECTION 3: LOAD KNOWLEDGE BASE & BUILD VECTOR STORE
# ============================================================================

def load_knowledge_base(path: str) -> tuple[list[str], list[str]]:
    print("\n" + "=" * 60)
    print("STEP 1: Loading Knowledge Base & Building Vector Store")
    print("=" * 60)

    with open(path, "r", encoding="utf-8") as f:
        kb_content = f.read()

    chunks, chunk_ids = [], []
    for chunk in kb_content.split(CHUNK_SEPARATOR):
        chunk = chunk.strip()
        # FIX: Uses MIN_CHUNK_LENGTH constant instead of a magic number,
        # and the threshold is raised to 200 to filter genuinely empty fragments.
        if len(chunk) > MIN_CHUNK_LENGTH:
            chunk_id = "UNKNOWN"
            for line in chunk.split("\n"):
                if line.startswith("## CHUNK"):
                    chunk_id = line.replace("## ", "").split(":")[0].strip()
                    break
            chunks.append(chunk)
            chunk_ids.append(chunk_id)

    print(f"  ✅ Loaded {len(chunks)} chunks from knowledge base")
    return chunks, chunk_ids


def build_vector_store(chunks: list[str], chunk_ids: list[str]) -> chromadb.Collection:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.Client()
    try:
        client.delete_collection("salem_balhamer_kb")
    except Exception:
        pass

    collection = client.create_collection(
        name="salem_balhamer_kb",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_id": cid} for cid in chunk_ids],
    )
    print(f"  ✅ Vector store built with {collection.count()} chunks")
    return collection


# ============================================================================
# SECTION 4: LOAD EVALUATION SET
# ============================================================================

def load_eval_set(path: str, limit: int | None = None) -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 2: Loading Evaluation Set")
    print("=" * 60)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["evaluation_set"]
    if limit:
        questions = questions[:limit]

    print(f"  ✅ Loaded {len(questions)} evaluation questions")
    dist = Counter(q["query_type"] for q in questions)
    print(f"  Query type distribution: {dict(dist)}")
    return questions


# ============================================================================
# SECTION 5: RETRIEVAL
# ============================================================================

def retrieve(question: str, collection: chromadb.Collection, top_k: int = TOP_K) -> dict:
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


# ============================================================================
# SECTION 6: GENERATION
# ============================================================================

SYSTEM_PROMPT = (
    "You are a helpful customer service chatbot for Salem Balhamer Holding Group. "
    "Answer questions based ONLY on the provided context. "
    "If the answer is not in the context, say so honestly. "
    "Do not make up information. Be concise and professional."
)


def generate_answer(question: str, contexts: list[str], retries: int = 3) -> str:
    """Generate an answer using the Groq API."""
    context_text = "\n\n---\n\n".join(contexts)
    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the provided context:"
    )

    for attempt in range(1, retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model=CHATBOT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"\n  ⚠️  Generation error (attempt {attempt}/{retries}): {e}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)


# ============================================================================
# SECTION 7: RUN RAG PIPELINE
# ============================================================================

def run_pipeline(eval_questions: list[dict], collection: chromadb.Collection) -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 3: Running RAG Pipeline on All Questions")
    print("=" * 60)

    results = []
    total = len(eval_questions)

    for i, q in enumerate(eval_questions):
        print(f"\r  Generating {i+1}/{total}: {q['id']}...", end="", flush=True)
        try:
            retrieval         = retrieve(q["question"], collection)
            contexts          = retrieval["documents"]
            retrieved_ids     = [m.get("chunk_id", "UNKNOWN") for m in retrieval["metadatas"]]
            answer            = generate_answer(q["question"], contexts)

            results.append({
                "id":                  q["id"],
                "query_type":          q["query_type"],
                "question":            q["question"],
                "expected_answer":     q["expected_answer"],
                "generated_answer":    answer,
                "retrieved_contexts":  contexts,
                "retrieved_chunk_ids": retrieved_ids,
                "expected_chunks":     q.get("source_chunks", []),
                "key_facts":           q.get("key_facts", []),
                "distances":           retrieval["distances"],
                "evaluation_focus":    q.get("evaluation_focus", []),
            })

        except Exception as e:
            print(f"\n  ⚠️  Error on {q['id']}: {e}")
            results.append({
                "id": q["id"], "query_type": q["query_type"],
                "question": q["question"], "expected_answer": q["expected_answer"],
                "generated_answer": f"ERROR: {e}",
                "retrieved_contexts": [], "retrieved_chunk_ids": [],
                "expected_chunks": [], "key_facts": [], "distances": [],
                "evaluation_focus": [],
            })

        # FIX: Rate limit delay applied after each generation call to avoid
        # hitting Groq's request-per-minute limit mid-eval.
        time.sleep(INTER_REQUEST_DELAY_SECONDS)

    print(f"\n  ✅ RAG pipeline complete.")
    return results


# ============================================================================
# SECTION 8: LLM-AS-JUDGE
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a RAG chatbot.
Score the generated answer on FIVE dimensions using a 1-5 scale:
1. FAITHFULNESS       — does the answer stay faithful to the retrieved context?
2. ANSWER_RELEVANCE   — does the answer directly address the question?
3. COMPLETENESS       — does the answer cover all important aspects?
4. CORRECTNESS        — does the answer match the expected answer?
5. CONTEXT_RELEVANCE  — is the retrieved context relevant to the question?

Respond ONLY with valid JSON in this exact format (no extra text, no markdown):
{
  "faithfulness": 5,
  "answer_relevance": 5,
  "completeness": 5,
  "correctness": 5,
  "context_relevance": 5,
  "hallucination_detected": false,
  "key_facts_covered": ["fact1"],
  "key_facts_missed": ["fact2"],
  "brief_reasoning": "one sentence explanation"
}"""

_EMPTY_SCORES = {
    "faithfulness": 0, "answer_relevance": 0, "completeness": 0,
    "correctness": 0, "context_relevance": 0, "hallucination_detected": True,
    "key_facts_covered": [], "key_facts_missed": [], "brief_reasoning": "",
}


def judge_answer(
    question: str,
    expected: str,
    generated: str,
    contexts: list[str],
    key_facts: list[str],
    retries: int = 3,
) -> dict:
    """Ask the judge model to score one Q&A pair. Returns a dict of scores."""

    context_text = "\n---\n".join(contexts[:3])
    user_prompt  = (
        f"QUESTION: {question}\n\n"
        f"EXPECTED ANSWER:\n{expected}\n\n"
        f"GENERATED ANSWER:\n{generated}\n\n"
        f"RETRIEVED CONTEXT:\n{context_text}\n\n"
        f"KEY FACTS TO CHECK: {json.dumps(key_facts)}\n\n"
        "Respond with JSON only:"
    )

    for attempt in range(1, retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)

        except json.JSONDecodeError as e:
            if attempt == retries:
                scores = dict(_EMPTY_SCORES)
                scores["brief_reasoning"] = f"JSON parse error: {e}"
                return scores
            time.sleep(1)

        except Exception as e:
            if attempt == retries:
                scores = dict(_EMPTY_SCORES)
                scores["brief_reasoning"] = f"Error: {e}"
                return scores
            wait = 2 ** attempt
            print(f"\n  ⚠️  Judge error (attempt {attempt}/{retries}): {e}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)


def run_judge(results: list[dict]) -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 4: LLM-as-Judge Evaluation (Groq API)")
    print("=" * 60)

    judge_scores = []
    total = len(results)

    for i, r in enumerate(results):
        print(f"\n[{i+1}/{total}] ID: {r['id']} | Model: {CHATBOT_MODEL}")
        print(f"Q: {r['question']}")
        print(f"A: {r['generated_answer']}\n")
        print("Judging...", end="", flush=True)

        scores = judge_answer(
            question=r["question"],
            expected=r["expected_answer"],
            generated=r["generated_answer"],
            contexts=r["retrieved_contexts"],
            key_facts=r["key_facts"],
        )

        print(f"\rBENCHMARKS: Faithfulness: {scores.get('faithfulness', 0)}/5 | "
              f"Relevance: {scores.get('answer_relevance', 0)}/5 | "
              f"Completeness: {scores.get('completeness', 0)}/5 | "
              f"Correctness: {scores.get('correctness', 0)}/5 | "
              f"Context: {scores.get('context_relevance', 0)}/5")
        print("-" * 60)

        scores["id"]         = r["id"]
        scores["query_type"] = r["query_type"]
        judge_scores.append(scores)

        # FIX: Rate limit delay between judge calls for the same reason as generation.
        time.sleep(INTER_REQUEST_DELAY_SECONDS)

    print(f"\n  ✅ Judging complete.")
    return judge_scores


# ============================================================================
# SECTION 9: EXPORT RESULTS
# ============================================================================

def export_results(results: list[dict], judge_scores: list[dict], output_path: str) -> None:
    scores_df = pd.DataFrame(judge_scores)
    output_df = pd.DataFrame(results)

    export_df = output_df.merge(scores_df, on=["id", "query_type"], how="left")

    for col in ["retrieved_chunk_ids", "key_facts_covered", "key_facts_missed"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x
            )

    export_df.to_csv(output_path, index=False)
    print(f"\n  ✅ Results saved to: {output_path}")

    numeric_cols = ["faithfulness", "answer_relevance", "completeness",
                    "correctness", "context_relevance"]
    available = [c for c in numeric_cols if c in scores_df.columns]
    if available:
        print("\n" + "=" * 60)
        print("SUMMARY — Average Scores by Query Type")
        print("=" * 60)
        summary = scores_df.groupby("query_type")[available].mean().round(2)
        print(summary.to_string())
        print(f"\nOverall averages:")
        print(scores_df[available].mean().round(2).to_string())


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    check_groq_ready()

    chunks, chunk_ids = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    collection = build_vector_store(chunks, chunk_ids)

    eval_questions = load_eval_set(EVAL_SET_PATH, limit=MAX_EVAL_QUESTIONS)

    results = run_pipeline(eval_questions, collection)

    judge_scores = run_judge(results)

    export_results(results, judge_scores, RESULTS_OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("ALL DONE ✅")
    print("=" * 60)