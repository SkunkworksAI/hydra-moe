from transformers import AutoModelForCausalLM, AutoTokenizer
import config as Config
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
import torch
from loguru import logger
from uuid import uuid4
from threading import Event
from pathlib import Path
import json


class ModelWorker:
    """Singleton class to manage language model and RabbitMQ interactions for inference.

    This class initializes and holds the language model, listens to RabbitMQ for inference
    requests, and publishes the results back to another RabbitMQ queue.
    """

    _instance = None

    def __new__(cls, inference_strategy, config: Config.ModelConfig = None) -> 'ModelWorker':
        """Create a new instance or return the existing instance."""
        if cls._instance is None:
            logger.info("Initializing Model Worker")
            
            # If config is None, try to load it from YAML
            if config is None:
                yaml_path = '/configs/inference_config.yaml'
                
                if Path(yaml_path).exists():
                    config: Config.ModelConfig = Config.build_model_config_from_yaml(yaml_path)
                else:
                    raise FileNotFoundError(f"Could not find the configuration YAML file at {yaml_path}. Please ensure the file exists.")
            
            cls._instance = super(ModelWorker, cls).__new__(cls)
            cls._instance.init_models(config)
            cls._instance.inference_strategy = inference_strategy  
            logger.info("ModelWorker Initialized")
        return cls._instance






    def init_base_model(self, config: Config.ModelConfig) -> None:
        """Initialize the base language model and tokenizer.

        Args:
            config (ModelConfig): Configuration for the model.
        """
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map={"": 0},
            resume_download=True,
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map={"": 0},
            resume_download=True,
        )
        self.base_tokenizer.bos_token_id = 1

    def init_models(self, config):
        torch.set_default_device("cuda")
        self.init_base_model(config)

    def publish_to_stream(self, channel, message: str, stream_complete: Event, session_id: str) -> None:
        """Publish the model's output to a RabbitMQ stream.
        
        Args:
            channel: RabbitMQ channel.
            message (str): Message to publish.
            stream_complete (Event): Event to indicate if the stream is complete.
            session_id (str): Unique session identifier.
        """
        data = json.dumps({"token": message, "session_id": session_id})
        logger.info(f"Publishing stream: {data}")
        channel.basic_publish(
            exchange="", routing_key="inference_results_stream", body=data
        )
        if stream_complete.is_set():
            data = json.dumps({"token": "STREAM_COMPLETE", "session_id": session_id})
            logger.info(f"Ending stream:  STREAM_COMPLETE")
            channel.basic_publish(
                exchange="", routing_key="inference_results_stream", body=data
            )

    def listen_for_inference_requests(self) -> None:
        """Listen to a RabbitMQ queue for incoming inference requests and process them."""
        connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
        channel = connection.channel()

        channel.queue_declare(
            queue="inference_requests_stream",
            durable=True,
            arguments={"x-queue-type": "stream"},
        )

        channel.queue_declare(
            queue="inference_results_stream",
            durable=True,
            arguments={"x-queue-type": "stream"},
        )
        channel.basic_qos(prefetch_count=1)
        # channel.basic_qos(prefetch_count=1000)

        def on_inference_request(ch: BlockingChannel, method: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
            data = json.loads(body.decode())
            message = data["query"]
            session_id = data["session_id"]
            self.inference_strategy.perform_inference(
                message,
                session_id,
                self.base_model,
                self.base_tokenizer,
                lambda msg, stream_complete: self.publish_to_stream(
                    ch, msg, stream_complete, session_id
                ),  # Pass the session_id to publish_to_stream
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(
            queue="inference_requests_stream",
            on_message_callback=on_inference_request,
            auto_ack=False,
        )

        logger.info("Waiting for inference requests...")
        channel.start_consuming()
