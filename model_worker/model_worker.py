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
from threading import Thread
from tokenizer import check_length

class ModelWorker:
    """Singleton class to manage language model and RabbitMQ interactions for inference.

    This class initializes and holds the language model, listens to RabbitMQ for inference
    requests, and publishes the results back to another RabbitMQ queue.
    """

    _instance = None

    def __new__(cls, inference_strategy, config_path: str = None) -> 'ModelWorker':
        """Create a new instance or return the existing instance."""
        if cls._instance is None:
            logger.info("Initializing Model Worker")
            
            # If config is None, try to load it from YAML
            if config_path is None:
                config_path = '/configs/inference_config.yaml'
                
                if Path(config_path).exists():
                    config: Config.ModelConfig = Config.build_model_config_from_yaml(config_path)
                else:
                    raise FileNotFoundError(f"Could not find the configuration YAML file at {config_path}. Please ensure the file exists.")
            
            cls._instance = super(ModelWorker, cls).__new__(cls)
            cls._instance.init_models(config)
            cls._instance.inference_strategy = inference_strategy  
            logger.info("ModelWorker Initialized")
        return cls._instance

    def start_listeners(self):
            """Start threads to listen for both inference and info requests."""
            # Thread for inference requests
            inference_thread = Thread(target=self.listen_for_inference_requests)
            inference_thread.start()

            # Thread for info requests
            info_thread = Thread(target=self.listen_for_info_requests)
            info_thread.start()

            # Optionally, you can add logging to confirm that the threads have started
            logger.info("Started threads for listening to inference and info requests.")

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
        self.config = config


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
            logger.info(f"Received inference request: {data}")
            message = data["query"]
            session_id = data["session_id"]
            max_tokens = data['max_tokens']

            error_response = check_length(self.base_tokenizer, max_tokens, self.config.max_new_tokens, message)

            if error_response:
                # Publish the error to the stream
                error_message = error_response.get('message', 'An unknown error occurred.')
                error_json = json.dumps({"error": error_message})
                ch.basic_publish(
                    exchange="", routing_key="inference_results_stream", body=error_json
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            self.inference_strategy.perform_inference(
                message,
                session_id,
                self.base_model,
                self.base_tokenizer,
                lambda msg, stream_complete: self.publish_to_stream(
                    ch, msg, stream_complete, session_id
                ),  
                max_tokens,
                temperature=data['temperature'],
                top_p=data['top_p'],
                top_k=data['top_k'],
                repetition_penalty=data['repetition_penalty']
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(
            queue="inference_requests_stream",
            on_message_callback=on_inference_request,
            auto_ack=False,
        )
        self.publish_model_info(channel)
        logger.info("Waiting for inference requests...")
        channel.start_consuming()

    def publish_model_info(self, channel) -> None:
        """Publish model info to a RabbitMQ stream."""
        channel.queue_declare(
            queue="model_info",
            durable=True,
        )
        
        logger.info(self.base_model.config)
        base_model_config = self.base_model.config.to_dict() 

        model_info = {
            "model_path": self.config.base_model_path,
            "tokenizer_name": self.base_tokenizer.__class__.__name__,
            "max_new_tokens": self.config.max_new_tokens,
            "lora_alpha": self.config.lora_alpha,
            "model_kind": self.base_model.config.model_type,
            "architectures": base_model_config.get("architectures"),
            "bos_token_id": base_model_config.get("bos_token_id"),
            "eos_token_id": base_model_config.get("eos_token_id"),
            "hidden_act": base_model_config.get("hidden_act"),
            "hidden_size": base_model_config.get("hidden_size"),
            "initializer_range": base_model_config.get("initializer_range"),
            "intermediate_size": base_model_config.get("intermediate_size"),
            "max_position_embeddings": base_model_config.get("max_position_embeddings"),
            "num_attention_heads": base_model_config.get("num_attention_heads"),
            "num_hidden_layers": base_model_config.get("num_hidden_layers"),
            "num_key_value_heads": base_model_config.get("num_key_value_heads"),
            "rms_norm_eps": base_model_config.get("rms_norm_eps"),
            "rope_theta": base_model_config.get("rope_theta"),
            "sliding_window": base_model_config.get("sliding_window"),
            "tie_word_embeddings": base_model_config.get("tie_word_embeddings"),
            "torch_dtype": base_model_config.get("torch_dtype"),
            "transformers_version": base_model_config.get("transformers_version"),
            "use_cache": base_model_config.get("use_cache"),
            "vocab_size": base_model_config.get("vocab_size")
        }
        model_info_json = json.dumps(model_info)
        channel.basic_publish(
            exchange="", routing_key="model_info", body=model_info_json
        )
        logger.info(f"Published model info: {model_info}")
        
    def listen_for_info_requests(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel = connection.channel()

        channel.queue_declare(
            queue='model_info_requests',
            durable=True
        )

        def on_request(ch, method, properties, body):
            # Assuming you publish the model info using the existing `publish_model_info` method
            self.publish_model_info(channel)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(
            queue='model_info_requests',
            on_message_callback=on_request,
            auto_ack=False
        )

        logger.info("Waiting for model info requests...")
        channel.start_consuming()