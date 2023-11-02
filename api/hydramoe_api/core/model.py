import pika
import json
from threading import Thread
from loguru import logger 

class ModelService:
    def __init__(self):
        self.latest_model_info = {}
        self.start_consumer()
        self.request_model_info()

    def start_consumer(self):
        thread = Thread(target=self._consume_model_info)
        thread.start()

    def _consume_model_info(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel = connection.channel()
        channel.queue_declare(
            queue='model_info',
            durable=True
        )

        def callback(ch, method, properties, body):
            logger.info("Received model info")
            self.latest_model_info = json.loads(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=1)

        channel.basic_consume(
            queue='model_info',
            on_message_callback=callback,
            auto_ack=False
        )
        channel.start_consuming()

    def request_model_info(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel = connection.channel()

        channel.queue_declare(
            queue='model_info_requests',
            durable=True
        )

        channel.basic_publish(
            exchange='',
            routing_key='model_info_requests',
            body='Request for model info'
        )

        connection.close()

    def get_latest_model_info(self):
        if self.latest_model_info == {}:
            self.request_model_info()
        
        return self.latest_model_info