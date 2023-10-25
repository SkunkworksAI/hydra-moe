"""
This module creates an API router for the FastAPI application and includes routes from the auth module.
"""

import hydramoe_api.endpoints.v1.chat as chat
from fastapi import APIRouter

api_router = APIRouter()
"""
An instance of APIRouter to which we will include the routes from the auth module.
"""


api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
"""
Includes the routes defined in the chat module under the prefix "/chat".
"""