"""
This module defines API endpoints related to model info
"""
from loguru import logger
from fastapi import APIRouter
import time
from hydramoe_api.core.model import ModelService
router = APIRouter()
model_info_manager = ModelService()

@router.get("/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    latest_model_info = model_info_manager.get_latest_model_info()
    if latest_model_info == {}: 
        time.sleep(1) # debounce if the info has not yet been received
        latest_model_info = model_info_manager.get_latest_model_info()    
    logger.info(latest_model_info)
    return latest_model_info
