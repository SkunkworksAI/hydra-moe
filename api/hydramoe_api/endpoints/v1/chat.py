"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from hydramoe_api.core.chat_service import ChatService

from hydramoe_api import schemas

router = APIRouter()

# Update your FastAPI route
@router.post("/chat", status_code=200)
async def chat_endpoint(request: schemas.ChatRequest):
    chat_service = ChatService()
    stream_response = chat_service.submit_query(request)
    return StreamingResponse(stream_response, media_type="text/plain")