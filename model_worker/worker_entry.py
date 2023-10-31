from model_worker import ModelWorker
from loguru import logger
import urllib3, socket
from urllib3.connection import HTTPConnection
from streaming_strategy import CompletionStreamingStrategy
import argparse

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])

parser = argparse.ArgumentParser(description="Start the Model Worker with an optional configuration file.")
parser.add_argument("--config", type=str, help="Path to the configuration YAML file.")

args = parser.parse_args()
config_path = args.config

if __name__ == "__main__":
    logger.info("Initializing Model Worker...")
    model_worker = ModelWorker(CompletionStreamingStrategy(), config_path=config_path if config_path else None)
    model_worker.start_listeners()