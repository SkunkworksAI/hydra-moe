from uuid import uuid4
import torch
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from loguru import logger 
from pydantic import BaseModel, Field
from typing import Any
from threading import Event, Thread
import hydramoe_api.schemas as Schemas
import pika 


torch.set_default_device('cuda')
class ChatService:
    def __init__(self):
        self.init_model()
        
    def init_model(self):

        self.model, self.tokenizer = self.request_model()
        
    def request_model(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))  # Connect to RabbitMQ
        channel = connection.channel()

        # Declare a queue for sending model requests
        channel.queue_declare(queue='model_request_queue')

        # Declare a temporary queue for receiving the model
        result = channel.queue_declare(queue='', exclusive=True)
        callback_queue = result.method.queue

        # Generate a correlation ID
        corr_id = str(uuid4())

        # Send the request for the model
        channel.basic_publish(
            exchange='',
            routing_key='model_request_queue',
            properties=pika.BasicProperties(
                reply_to=callback_queue,
                correlation_id=corr_id,
            ),
            body='Requesting model'
        )
        print("Model request sent.")

        # Function to handle responses
        def on_response(ch, method, properties, body):
            if corr_id == properties.correlation_id:
                # Here you can deserialize the received model and set it to `self.model`
                # For demonstration, just printing it.
                print("Received model:", body.decode())
                connection.close()

        # Wait for a response
        channel.basic_consume(
            queue=callback_queue,
            on_message_callback=on_response,
            auto_ack=True
        )
        print('Waiting for model...')
        channel.start_consuming()

    def convert_to_text(self, msg):
        system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        text = system_prompt
        text += f"### Instruction:\n{msg}\n\n"
        text += "### Response:\n"
        return text

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_token_ids = [0]
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def bot(self, message, conversation_id):
        stop = self.StopOnTokens()
        messages = self.convert_to_text(message)

        input_ids = self.tokenizer(messages, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        
        max_new_tokens = 1536  
        temperature = 0.01
        top_p = 0.9
        top_k = 0
        repetition_penalty = 1.1
        
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop]),
            eos_token_id=self.tokenizer.eos_token_id
        )

        model_output=  ""
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        for new_text in streamer:
            model_output += new_text
            logger.info(f"Model output: {new_text}")
            yield new_text

            
    def submit_query(self, request: Schemas.ChatRequest, request_id):
        return self.bot(request.query, request_id)
    
