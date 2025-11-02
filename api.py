# api.py
import logging
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError

from schemas import ChatRequest, ChatResponse
from config import Settings, get_settings
from services import ChatService

# Create an API router
router = APIRouter()

# --- Dependency Injection ---
# This is the key to good FastAPI design.
# We use @lru_cache to create a *single instance* of the service
# and re-use it for every request. This is efficient.
@lru_cache
def get_chat_service(settings: Settings = Depends(get_settings)) -> ChatService:
    """
    Dependency to create and cache a single instance of the ChatService.
    """
    return ChatService(settings=settings)
# -----------------------------

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    The main chat endpoint for your Next.js frontend.
    """
    try:
        response_text, intent, context = chat_service.get_chat_response(request)
        
        return ChatResponse(
            response=response_text,
            detected_intent=intent,
            retrieved_context=context
        )
        
    except OpenAIError as e:
        logging.error(f"An error occurred with the AI service: {e}")
        raise HTTPException(
            status_code=503, # 503 Service Unavailable
            detail=f"The AI service is currently unavailable. Please try again later."
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, # 500 Internal Server Error
            detail="An unexpected server error occurred."
        )