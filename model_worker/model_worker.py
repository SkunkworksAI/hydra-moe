from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ModelConfig
import pika
import torch
from loguru import logger
from uuid import uuid4
from threading import Event
import json

class ModelWorker:
    _instance = None

    def __new__(cls, inference_strategy, config=ModelConfig()):
        if cls._instance is None:
            logger.info("Initializing Model Worker")
            cls._instance = super(ModelWorker, cls).__new__(cls)
            cls._instance.init_models(config)
            cls._instance.inference_strategy = inference_strategy  
            logger.info("ModelWorker Initialized")
        return cls._instance

    def init_base_model(self, config: ModelConfig):
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_path, trust_remote_code=True, torch_dtype="auto", device_map={"": 0}, resume_download=True)
        self.base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True, torch_dtype="auto", device_map={"": 0}, resume_download=True)
        self.base_tokenizer.bos_token_id = 1

    def init_models(self, config):
        torch.set_default_device('cuda')
        self.init_base_model(config)

    def publish_to_stream(self, channel, message, stream_complete, session_id):
            data = json.dumps({'token': message, 'session_id': session_id}) 
            logger.info(f"Publishing stream: {data}" )
            channel.basic_publish(
                exchange='',
                routing_key='inference_results_stream',
                body=data
            )
            if stream_complete.is_set():
                data = json.dumps({'token': "STREAM_COMPLETE", 'session_id': session_id})
                logger.info(f"Ending stream:  STREAM_COMPLETE" )
                channel.basic_publish(
                    exchange='',
                    routing_key='inference_results_stream',
                    body=data
                )
 

    def listen_for_inference_requests(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel = connection.channel()

        channel.queue_declare(
            queue='inference_requests_stream',
            durable=True,
            arguments={"x-queue-type": "stream"}
        )

        channel.queue_declare(
            queue='inference_results_stream',
            durable=True,
            arguments={"x-queue-type": "stream"}
        )
        channel.basic_qos(prefetch_count=1)
        # channel.basic_qos(prefetch_count=1000)


        def on_inference_request(ch, method, properties, body):
                data = json.loads(body.decode())
                message = data['query']
                session_id = data['session_id']
                self.inference_strategy.perform_inference(
                    message,
                    session_id, 
                    self.base_model,
                    self.base_tokenizer,
                    lambda msg, stream_complete: self.publish_to_stream(ch, msg, stream_complete, session_id)  # Pass the session_id to publish_to_stream
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(
            queue='inference_requests_stream',
            on_message_callback=on_inference_request,
            auto_ack=False 
        )

        logger.info('Waiting for inference requests...')
        channel.start_consuming()