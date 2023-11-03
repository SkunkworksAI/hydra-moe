from hydramoe_api.settings import settings
from hydramoe_api.app import app
from hydramoe_api.logging import setup_app_logging
from loguru import logger
import uvicorn

logger.info(settings)



if __name__ == "__main__":
    setup_app_logging(config=settings)

    logger.info("Starting uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.LOG_LEVEL)