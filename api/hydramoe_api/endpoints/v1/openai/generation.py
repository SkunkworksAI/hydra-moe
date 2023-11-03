"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from hydramoe_api.core.chat_service import ChatService
from hydramoe_api.schemas.openai import *
router = APIRouter()


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.
    """
    logger.info(f"Received chat completion request: {request}")
    pass 
                                
                                
@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.
    """
    logger.info(f"Received completion request: {request}")
    pass 