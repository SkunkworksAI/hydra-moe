"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from hydramoe_api.core.chat_service import ChatService
import time
from hydramoe_api import schemas

router = APIRouter()

# Update your FastAPI route
@router.post("/chat", status_code=200)
async def chat_endpoint(request: schemas.ChatRequest):
    chat_service = ChatService()
    stream_response = chat_service.submit_query(request)
    return StreamingResponse(stream_response, media_type="text/event-stream")

def fake_data_streamer():
    for i in range(10):
        yield b'some fake data\n\n'
        time.sleep(0.5)


@router.get('/streamtest')
async def main():
    return StreamingResponse(fake_data_streamer(), media_type='text/event-stream')