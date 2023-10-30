"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from hydramoe_api.core.chat_service import ChatService
from hydramoe_api import schemas
router = APIRouter()


@router.post("/chat", status_code=200)
async def chat_endpoint(request: schemas.ChatRequest):
    try:
        chat_service = ChatService()
        chat_service.start_service(request)
        return StreamingResponse(chat_service.stream_results(), media_type="text/plain")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise