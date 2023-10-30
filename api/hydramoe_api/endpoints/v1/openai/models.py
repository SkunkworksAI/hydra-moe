"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from hydramoe_api.core.chat_service import ChatService
from hydramoe_api.schemas.openai import *
router = APIRouter()

served_model = None # Get from model-service TODO

@router.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)
