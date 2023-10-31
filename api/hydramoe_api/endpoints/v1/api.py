"""
This module creates an API router for the FastAPI application.
"""

import hydramoe_api.endpoints.v1.chat as chat
import hydramoe_api.endpoints.v1.model as model
from fastapi import APIRouter

api_router = APIRouter()
"""
An instance of APIRouter to which we will include the routes from the auth module.
"""

api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
"""
Includes the routes defined in the chat module under the prefix "/chat".
"""

api_router.include_router(model.router,  tags=["model"])
"""
Includes the routes defined in the mode module.
"""