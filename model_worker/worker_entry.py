from model_worker import ModelWorker
from loguru import logger
import urllib3, socket
from urllib3.connection import HTTPConnection
from streaming_strategy import CompletionStreamingStrategy

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])

if __name__ == "__main__":
    logger.info("Initializing Model Worker...")
    model_worker = ModelWorker(CompletionStreamingStrategy())
    model_worker.listen_for_inference_requests()