# schemas.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    The request body from Next.js
    """
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    """
    The response body sent back to Next.js
    """
    response: str
    detected_intent: str
    retrieved_context: str