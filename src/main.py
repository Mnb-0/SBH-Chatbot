import os
import re
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from og.core_logic import (
    BASELINE_FACTS,
    get_rag_components,
    get_sector_filter,
    get_soft_search_results,
    self_correct,
    verify_response,
)

# --- Identity short-circuit ---
IDENTITY_REPLY = (
    "I am the Salem Balhamer Holding Group's corporate intelligence assistant. "
    "I can answer questions about SBH's business sectors — including Real Estate, Trading, "
    "Contracting, Industrial, and Services — as well as its leadership, history, and strategy. "
    "How can I help you today?"
)

_IDENTITY_PATTERNS = re.compile(
    r"\b(who|what)\s+(are|is)\s+(you|this|ur)\b"
    r"|\bwhat\s+(can|do)\s+you\b"
    r"|\byour\s+(name|purpose|role|job|function)\b"
    r"|\bintroduce\s+yourself\b"
    r"|\bwhat\s+are\s+you\b",
    re.IGNORECASE,
)


def is_identity_question(text: str) -> bool:
    return bool(_IDENTITY_PATTERNS.search(text))


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


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Salem Balhamer API", "chat": "/api/chat", "docs": "/docs"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, req: Request):
    prompt = payload.message
    vectorstore = req.app.state.vectorstore
    gen_llm = req.app.state.gen_llm
    critic_llm = req.app.state.critic_llm

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Short-circuit identity questions before hitting the RAG pipeline
        if is_identity_question(prompt):
            return ChatResponse(reply=IDENTITY_REPLY)

        # 1. Routing
        sector = get_sector_filter(prompt, critic_llm)

        # 2. Retrieval
        docs = get_soft_search_results(vectorstore, prompt, sector)
        context = "\n\n".join([d.page_content for d in docs])

        # 3. Generation
        # FIX: BASELINE_FACTS is now a single source of truth imported from core_logic.
        # Both the generator and auditor use the same constant, so they can never go out of sync.
        system_prompt = f"""
            ### ROLE: Corporate Analyst
            Rules: Use ONLY the provided context and baseline facts to answer the user's query. You are permitted to synthesize related themes if the text contains relevant corporate strategy.
            
            CRITICAL THRESHOLD: If the context and baseline facts are completely irrelevant and offer absolutely no foundational information to address the query, you must output EXACTLY the word "UNANSWERABLE" and nothing else. Do not apologize.
            
            {BASELINE_FACTS}
            
            RETRIEVED CONTEXT: {context}
            """

        # FIX: Conversation history is now built into the message list so the LLM
        # has full context of the conversation and can give coherent follow-up answers.
        messages = [SystemMessage(content=system_prompt)]
        for item in payload.history:
            if item.role == "user":
                messages.append(HumanMessage(content=item.text))
            else:
                messages.append(AIMessage(content=item.text))
        messages.append(HumanMessage(content=prompt))

        llm_response = gen_llm.invoke(messages)
        initial_res = llm_response.content.strip()

        # 4. Audit & Short-Circuit
        if "UNANSWERABLE" in initial_res.upper():
            final_reply = "I lack the corporate documentation to answer this question. The current database does not contain this information."
        else:
            # Combine baseline facts with retrieved context so the auditor
            # doesn't falsely flag known corporate facts as hallucinations.
            full_audit_context = f"{BASELINE_FACTS}\n{context}"

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
