import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import your existing logic
from core_logic import (
    get_rag_components, 
    get_sector_filter, 
    get_soft_search_results, 
    verify_response, 
    self_correct
)

# --- Pydantic Models for API Contract ---
class HistoryItem(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []

class ChatResponse(BaseModel):
    reply: str

# --- Initialize App & Components ---
app = FastAPI(title="Salem Balhamer API")

# Allow frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load components globally so they don't re-initialize on every request
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is missing.")

try:
    vectorstore, gen_llm, critic_llm = get_rag_components(api_key)
except Exception as e:
    raise RuntimeError(f"Failed to load system components: {e}")

from langchain_core.messages import SystemMessage, HumanMessage

# --- The API Endpoint ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    prompt = request.message
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # 1. Routing
        sector = get_sector_filter(prompt, critic_llm)

        # 2. Retrieval
        docs = get_soft_search_results(vectorstore, prompt, sector)
        context = "\n\n".join([d.page_content for d in docs])

        # 3. Generation
        system_prompt = f"""
        ### ROLE: Corporate Analyst
        Rules: Use ONLY the provided context to answer the user's query. You are permitted to synthesize related themes (e.g., summarizing strategic goals to answer a "mission" question) if the text contains relevant corporate strategy.
        
        CRITICAL THRESHOLD: If the context is completely irrelevant and offers absolutely no foundational information to address the query, you must output EXACTLY the word "UNANSWERABLE" and nothing else. Do not apologize.
        
        CONTEXT: {context}
        """
        
        llm_response = gen_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        raw_content = getattr(llm_response, "content", llm_response)

        if isinstance(raw_content, str):
            initial_res = raw_content.strip()
        elif isinstance(raw_content, list):
            initial_res = " ".join(
                part if isinstance(part, str)
                else str(part.get("text", "")) if isinstance(part, dict)
                else str(part)
                for part in raw_content
            ).strip()
        else:
            initial_res = str(raw_content).strip()

        # 4. Audit & Short-Circuit
        if "UNANSWERABLE" in initial_res.upper():
            final_reply = "I lack the corporate documentation to answer this question. The current database does not contain this information."
        else:
            audit_result = verify_response(critic_llm, context, initial_res)
            
            if audit_result.get("is_hallucinated", False):
                reason = audit_result.get("reason", "Unknown hallucination")
                final_reply = self_correct(gen_llm, context, prompt, initial_res, reason)
            else:
                final_reply = initial_res

        # Return exact JSON shape required by API.md
        return {"reply": final_reply}

    except Exception as e:
        # Catch errors so the server doesn't crash, return 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )