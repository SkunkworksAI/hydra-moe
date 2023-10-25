"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter, Depends

from hydramoe_api import schemas

router = APIRouter()

@router.post("/chat", status_code=200, response_model=schemas.ChatBase)
def chat_endpoint(
    request: schemas.ChatRequest
):
    # Fetch chat history from Redis
    logger.info(request)

    response_model = schemas.ChatBase(
        query=request.query,
        result="Test Result",
        session_id=request.session_id
    )
    logger.info(response_model)
    return response_model