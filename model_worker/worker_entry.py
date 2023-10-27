from model_worker import ModelWorker
from loguru import logger

if __name__ == "__main__":
    logger.info("Initializing Model Worker...")
    model_worker = ModelWorker()
    model_worker.listen_for_model_requests()