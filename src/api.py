from fastapi import FastAPI, HTTPException
from models.chat import ChatRequest, ChatResponse
from services.chat_service import generate_reply

app = FastAPI()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(data: ChatRequest):
    try:
        if not data.message:
            raise HTTPException(status_code=400, detail="Message is required")

        reply = generate_reply(data.message, data.history)

        if not reply or not isinstance(reply, str):
            raise HTTPException(status_code=500, detail="Invalid reply format")

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))