from transformers import AutoModelForCausalLM, AutoTokenizer
import pika 
import torch 
from dataclasses import dataclass
from loguru import logger
import urllib3, socket
from urllib3.connection import HTTPConnection

def connect_to_broker():
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue='model_request_queue')
    return channel

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])

cluster_nums = range(32)  
checkpoint_dirs = [
{
    "adapter_dir": f"HydraLM/mistralic-expert-{str(cluster)}",
    "adapter_name": f"{str(cluster)}"
}
for cluster in cluster_nums
]

@dataclass 
class ModelConfig:
    base_model_path: str = "SkunkworksAI/Mistralic-7B-1"
    # expert_model_paths: dict = checkpoint_dirs

class ModelWorker:
    _instance = None
    model_pool = []
    request_queue = []

    def __new__(cls, config=ModelConfig()):
        if cls._instance is None:
            logger.info("Initializing Model Worker")
            cls._instance = super(ModelWorker, cls).__new__(cls)
            cls._instance.init_models(config)
            logger.info("ModelWorker Initialized")
        return cls._instance
    
    def init_base_model(self, config: ModelConfig): 
        # Initialize base model
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_path, trust_remote_code=True, torch_dtype="auto", device_map={"": 0}, resume_download=True)
        self.base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True, torch_dtype="auto",device_map={"": 0}, resume_download=True)
        self.base_tokenizer.bos_token_id = 1
        
    def init_models(self, config):
        torch.set_default_device('cuda')
        self.init_base_model(config)


    def listen_for_model_requests(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel = connection.channel()

        channel.queue_declare(queue='model_request_queue')

        def callback(ch, method, properties, body):
            print("Received model request.")
            # Serialize and send back model and tokenizer here

        channel.basic_consume(queue='model_request_queue',
                              on_message_callback=callback,
                              auto_ack=True)

        logger.info('ModelService is waiting for model requests. To exit press CTRL+C')
        channel.start_consuming()
    
    def handle_model_request(self, ch, method, properties, body):
        # Handle incoming requests for the model
        logger.info(f"Received request: {body.decode()}")

        # Send back the base model (or expert model based on request)
        # For demonstration, just sending back a string
        response = "Base model here!"
        
        # Publish the response to RabbitMQ
        self.channel.basic_publish(exchange='',
                                   routing_key=properties.reply_to,
                                   properties=pika.BasicProperties(
                                       correlation_id=properties.correlation_id
                                   ),
                                   body=response)

