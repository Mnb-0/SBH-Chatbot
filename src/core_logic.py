import os
import json
from typing import Any
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def get_rag_components(api_key):
    """Initializes the vector store and dual-model setup."""
    if not api_key:
        raise ValueError("API Key is missing. Check your environment variables.")

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", providers=["CPUExecutionProvider"]
    )

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qdrant_db_clean"
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
    
    Output ONLY the exact category name. Nothing else.
    """
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
        ]
        return category if category in valid_sectors else "General"
    except:
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
    """The Critic Pass: Forces a structured JSON evaluation of the answer."""
    verification_prompt = f"""
    ### ROLE: Strict Corporate Auditor
    Compare the ANSWER against the CONTEXT.
    Do not penalize for missing information, only for incorrect information that contradicts the CONTEXT. 
    Respond ONLY in valid JSON format matching this schema exactly:
    {{
        "is_hallucinated": boolean,
        "reason": "Short explanation of the finding."
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
