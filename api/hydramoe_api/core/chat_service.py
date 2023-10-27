from uuid import uuid4
from loguru import logger 
import hydramoe_api.schemas as Schemas
import pika 
from pika.adapters.blocking_connection import BlockingChannel
from threading import Thread

class ChatService:       
    def __init__(self):
        self.buffer = []
        
        
    def listen_for_inference_results(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel: BlockingChannel = connection.channel()

        stream_name = 'inference_results_stream'
        channel.queue_declare(
            queue=stream_name,
            durable=True,
            arguments={"x-queue-type": "stream"}
        )

        consumer_name = f"consumer_{uuid4()}"
        channel.basic_qos(prefetch_count=1000)
        
        def on_inference_result(ch, method, properties, body):
            logger.info("Received inference results")
            token = body.decode()
            logger.info(token)
            self.buffer.append(body.decode())

        channel.basic_consume(
            queue=stream_name,
            on_message_callback=on_inference_result,
            consumer_tag=consumer_name
        )

        logger.info('Waiting for inference results...')
        channel.start_consuming()
 
            
    def submit_query(self, request: Schemas.ChatRequest):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel = connection.channel()

        channel.queue_declare(queue='inference_requests_stream',
                            durable=True,
                            arguments={"x-queue-type": "stream"})

        channel.basic_publish(
            exchange='',
            routing_key='inference_requests_stream',
            body=request.query
        )
        logger.info("Client query sent.")
        connection.close()
        
    def start_service(self, request):
        listen_thread = Thread(target=self.listen_for_inference_results)
        submit_thread = Thread(target=self.submit_query, args=(request,))
        
        listen_thread.start()
        submit_thread.start()

    def stream_results(self):
        while True:
            if self.buffer:
                yield self.buffer.pop(0)