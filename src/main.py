import os
import re
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables explicitly for local parity
load_dotenv()

# FIX: Unified import path to match app.py and pulled in AUDIT_MIN_LENGTH
from core_logic import (
    BASELINE_FACTS,
    AUDIT_MIN_LENGTH,
    MAX_HISTORY_TURNS,
    get_rag_components,
    get_sector_filter,
    get_soft_search_results,
    self_correct,
    verify_response,
)

# --- Casual & Identity Short-Circuit (Ported from app.py) ---
IDENTITY_REPLY = (
    "I am the Salem Balhamer Holding Group's corporate intelligence assistant. "
    "I can answer questions about SBH's business sectors — including Real Estate, Trading, "
    "Contracting, Industrial, and Services — as well as its leadership, history, and strategy. "
    "How can I help you today?"
)

_CASUAL_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings|salaam|marhaba)[\s!?.]*$"
    r"|^\s*(thanks|thank you|thx|ty)[\s!?.]*$"
    r"|\b(who|what)\s+(are|is)\s+(you|this|ur)\b"
    r"|\bwhat\s+(can|do)\s+you\b"
    r"|\byour\s+(name|purpose|role|job|function)\b"
    r"|\bintroduce\s+yourself\b"
    r"|\bwhat\s+are\s+you\b"
    r"|\btell me about yourself\b"
    r"|\bwhat can you do\b",
    re.IGNORECASE,
)

_GREETING_REPLIES = {
    "hi": "Hello! How can I help you with Salem Balhamer Holding Group today?",
    "hello": "Hello! How can I help you with Salem Balhamer Holding Group today?",
    "hey": "Hey there! Feel free to ask me anything about SBH.",
    "thanks": "You're welcome! Let me know if there's anything else I can help with.",
    "thank you": "You're welcome! Let me know if there's anything else I can help with.",
}


def get_casual_reply(text: str) -> str | None:
    if not _CASUAL_PATTERNS.search(text):
        return None
    lower = text.strip().lower().rstrip("!?. ")
    return _GREETING_REPLIES.get(lower, IDENTITY_REPLY)


# --- Pydantic Models for API Contract ---
class HistoryItem(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []


class ChatResponse(BaseModel):
    reply: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is missing.")
    try:
        vectorstore, gen_llm, critic_llm = get_rag_components(api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to load system components: {e}") from e
    app.state.vectorstore = vectorstore
    app.state.gen_llm = gen_llm
    app.state.critic_llm = critic_llm
    yield


app = FastAPI(
    title="Salem Balhamer API",
    description="RAG chat API for SBH corporate intelligence (POST /api/chat).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, req: Request):
    prompt = payload.message
    vectorstore = req.app.state.vectorstore
    gen_llm = req.app.state.gen_llm
    critic_llm = req.app.state.critic_llm

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # 1. Short-circuit greetings/identity (Ported from app.py)
        casual_reply = get_casual_reply(prompt)
        if casual_reply:
            return ChatResponse(reply=casual_reply)

        # 2. Routing
        sector = get_sector_filter(prompt, gen_llm)

        # 3. Retrieval
        docs = get_soft_search_results(vectorstore, prompt, sector)
        context = "\n\n".join([d.page_content for d in docs])

        # 4. Generation (System Prompt ported directly from app.py)
        system_prompt = f"""
            You are a knowledgeable, friendly assistant for Salem Balhamer Holding Group.
            You talk like a real person — warm, clear, and natural — not like a formal corporate brochure.
            Keep answers focused and easy to read. Use short paragraphs. Avoid bullet-point overload unless a list genuinely helps.

            STRICT RULES:
            1. Only use information from the retrieved context and baseline facts below. Never invent or guess.
            2. Match the user's language — if they write in Arabic, reply fully in Arabic.
            3. If you only have partial information, share what you know and honestly say you don't have the full picture.
            4. Don't quote specific numbers, financials, or headcounts unless they appear explicitly in the context.
            5. If someone asks you to compare SBH with competitors or criticise the company, politely steer the conversation back and suggest they reach out to the team directly.
            6. Never open with hollow phrases like "Certainly!", "Of course!", or "Great question!". Just answer.
            7. When your answer is incomplete or the person might need more help, politely suggest they reach out to the support team using the contact information provided in the baseline facts.
            8. If the context and baseline facts have absolutely nothing relevant to the query, output ONLY the single word: UNANSWERABLE

            {BASELINE_FACTS}

            RETRIEVED CONTEXT: {context}
        """

        messages = [SystemMessage(content=system_prompt)]
        for item in payload.history[-MAX_HISTORY_TURNS:]:
            if item.role == "user":
                messages.append(HumanMessage(content=item.text))
            else:
                messages.append(AIMessage(content=item.text))
        messages.append(HumanMessage(content=prompt))

        llm_response = gen_llm.invoke(messages)
        initial_res = llm_response.content.strip()

        full_audit_context = f"{BASELINE_FACTS}\n{context}"

        # 5. Audit & Correction Logic (Ported exactly from app.py)
        if "UNANSWERABLE" in initial_res.upper():
            final_reply = (
                "That's outside what I currently have on record. "
                "For the most accurate information, please reach out to the SBH team directly at "
                "+966 138127397 or visit us at Balhamer Business Gate, Dammam."
            )
        elif len(initial_res) < AUDIT_MIN_LENGTH:
            # Skip the 70B auditor to save latency and tokens
            final_reply = initial_res
        else:
            audit_result = verify_response(critic_llm, full_audit_context, initial_res)

            if audit_result.get("is_hallucinated", False):
                reason = audit_result.get("reason", "Unknown hallucination")
                final_reply = self_correct(
                    gen_llm, full_audit_context, prompt, initial_res, reason
                )
            else:
                final_reply = initial_res

        return ChatResponse(reply=final_reply)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
