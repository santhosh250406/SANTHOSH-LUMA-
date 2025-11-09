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
# Use a cached single instance of ChatService for performance
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
    Main chat endpoint for the frontend.
    Uses the ChatService to handle NLU + RAG processing and generate a response.
    """
    try:
        # --- Core processing ---
        # request.message → user text
        response_text, intent, context = chat_service.get_chat_response(request)
        
        # --- Return structured output ---
        return ChatResponse(
            response=response_text,
            detected_intent=intent,
            retrieved_context=context
        )

    # --- Error Handling ---
    except OpenAIError as e:
        logging.error(f"AI service error: {e}")
        raise HTTPException(
            status_code=503,
            detail="The AI service is currently unavailable. Please try again later."
        )
    except Exception as e:
        logging.error(f"Unexpected server error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        )
