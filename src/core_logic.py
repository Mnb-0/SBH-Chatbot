import logging
import os
import json
from typing import Any
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

api_key = os.environ.get("GROQ_API_KEY")

# FIX: Single source of truth for baseline facts, shared by the generator and auditor
# in main.py. Updating facts here propagates everywhere automatically.
BASELINE_FACTS = """
    BASELINE CORPORATE FACTS:
    - Company Name: Salem Balhamer Holding Group
    - Founder & Chairman: Sheikh Salem Ahmed Balhamer
    - CEO: Ahmed Salem Balhamer
    - Headquarters: Balhamer Business Gate, Floor 20, Al Ashriah Street, Dammam, KSA
    - Phone: +966 138127397
    - Established: 1979 (Trading), 2013 (Holding Group)
    """
    
    
AUDIT_MIN_LENGTH = 100


def get_rag_components(api_key):
    """Initializes the vector store and dual-model setup."""
    if not api_key:
        raise ValueError("API Key is missing. Check your environment variables.")

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", providers=["CPUExecutionProvider"]
    )

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qdrant_db"
    )
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database folder '{db_path}' not found! Run indexer.py first."
        )

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings, collection_name="salem_balhamer_knowledge", path=db_path
    )

    generator_llm = ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.3, groq_api_key=api_key
    )
    critic_llm = ChatGroq(
        model="llama-3.3-70b-versatile", temperature=0, groq_api_key=api_key
    )

    return vectorstore, generator_llm, critic_llm


def get_sector_filter(query, llm):
    """Routes the query with strict definitions to prevent misclassification."""
    routing_prompt = """
    Categorize the following query into ONE of these sectors: 
    [Real Estate, Trading, Contracting, Industrial, Services, General]
    
    CRITICAL RULES:
    - Leadership (CEO, Chairman, Executives), company history, mission, or the "Holding Group" as a whole MUST map to -> General
    - Factories, manufacturing, or specific production plants map to -> Industrial
    - Construction, building projects, or NEOM map to -> Contracting
    - Residential complexes, villas, or properties map to -> Real Estate
    - Plastic products, pipes, or insulation sales map to -> Trading
    - Training programs, IT services, or support functions map to -> Services
    - If the query does not clearly match any specific sector, map to -> General
    
    Output ONLY the exact category name from the list above. Nothing else.
    """
    # FIX: Replaced bare except with explicit Exception logging so routing failures
    # are visible rather than silently degrading every response to "General".
    try:
        response = llm.invoke(
            [HumanMessage(content=routing_prompt + f"\nQuery: {query}")]
        )
        category = response.content.strip()
        valid_sectors = [
            "Real Estate",
            "Trading",
            "Contracting",
            "Industrial",
            "Services",
            "General",
        ]
        if category not in valid_sectors:
            logger.warning(f"Router returned unexpected sector '{category}' for query '{query}'. Falling back to General.")
            return "General"
        return category
    except Exception as e:
        logger.error(f"Sector routing failed for query '{query}': {e}")
        return "General"


def get_soft_search_results(vectorstore, query, target_sector, k=8):
    """Retrieves context, heavily favoring the routed sector."""
    search_kwargs: dict[str, Any] = {"k": k}
    if target_sector and target_sector != "General":
        search_kwargs["filter"] = Filter(
            should=[
                FieldCondition(
                    key="metadata.sector", match=MatchValue(value=target_sector)
                )
            ]
        )
    return vectorstore.similarity_search(query, **search_kwargs)


def verify_response(critic_llm, context, answer):
    """The Critic Pass: Neutral fact-check against provided context."""
    # FIX: Replaced "Hostile Auditor" framing with a neutral fact-checker.
    # The previous prompt was calibrated to find any excuse to reject, causing
    # frequent false positives that triggered self_correct unnecessarily,
    # adding latency and often producing worse, over-hedged replies.
    verification_prompt = f"""
    ### ROLE: Neutral Fact-Checker
    Your job is to verify whether the ANSWER contains claims that directly contradict 
    or are completely absent from the CONTEXT. 
    
    RULES:
    1. Only flag clear fabrications — specific names, dates, or numbers that are NOT present in the CONTEXT.
    2. Reasonable synthesis or summarisation of content that IS in the CONTEXT is acceptable.
    3. Do NOT flag an answer simply because the CONTEXT is incomplete; only flag active contradictions or inventions.
    
    Respond ONLY in valid JSON format matching this schema exactly:
    {{
        "is_hallucinated": boolean,
        "reason": "Specific explanation of which fact was fabricated, or 'Answer is consistent with context' if valid."
    }}

    CONTEXT: {context}
    ANSWER: {answer}
    """
    response = critic_llm.invoke(
        [
            SystemMessage(
                content="You are a strict JSON-output machine. No conversational filler."
            ),
            HumanMessage(content=verification_prompt),
        ]
    )

    try:
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except json.JSONDecodeError:
        return {
            "is_hallucinated": True,
            "reason": "Auditor failed to output valid JSON. Flagged as hallucination for safety.",
        }


def self_correct(
    generator_llm, context, original_query, flawed_answer, critic_feedback
):
    """The Correction Pass."""
    correction_prompt = f"""
    ### ROLE: Revision Assistant
    Your previous answer was flagged for inaccuracies by the Auditor.
    
    AUDITOR FEEDBACK: {critic_feedback}
    ORIGINAL CONTEXT: {context}
    
    TASK: Rewrite the answer to the query: '{original_query}'. 
    STRICT RULE: Remove any information not explicitly found in the context.
    """
    response = generator_llm.invoke([SystemMessage(content=correction_prompt)])
    return response.content.strip()
