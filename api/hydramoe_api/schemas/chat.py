
"""
This module defines the Pydantic models for the Chat entity. These models are used for data validation,
serialization and deserialization.
"""

from pydantic import BaseModel
from typing import List


class ChatBase(BaseModel):
    """
    A base model for chat responses that includes the common attributes.
    """

    query: str
    result: str
    session_id: str
    # sources: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    session_id: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int


class ChatPair(BaseModel):
    human: str
    ai: str


class ChatHistory(BaseModel):
    history: List[ChatPair]
    session_id: str
    user_id: str


class ChatSession(BaseModel):
    session_id: str
    user_id: str


class ChatHistoryRequest(BaseModel):
    session_id: str