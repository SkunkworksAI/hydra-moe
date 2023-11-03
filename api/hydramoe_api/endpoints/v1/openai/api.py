"""
This module creates an API router for the FastAPI Openai Swap In API. 
"""

import hydramoe_api.endpoints.v1.openai.generation as gen
import hydramoe_api.endpoints.v1.openai.models as models
from fastapi import APIRouter

api_router = APIRouter()
"""
An instance of APIRouter to which we will include the routes from the auth module.
"""


api_router.include_router(gen.router, tags=["gen"])
"""
Includes the routes defined in the generation module.
"""


api_router.include_router(models.router, tags=["models"])
"""
Includes the routes defined in the models module.
"""