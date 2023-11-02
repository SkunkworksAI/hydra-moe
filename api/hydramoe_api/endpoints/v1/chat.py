"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from hydramoe_api import schemas
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union


import asyncio
from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import httpx
import json
import uuid
router = APIRouter()

app = FastAPI()

# List of servers
servers = ["http://localhost:8000"]#, "http://localhost:8002"]  # Update this with your actual servers
server_status = {server: False for server in servers}  # False means the server is not in use

class ChatCompletionRequest(BaseModel):
    user: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    messages: List[dict]

# A dictionary to store the queues for each user
user_queues = {}

async def handle_request(body, server, user):
    async with httpx.AsyncClient() as client:
        # Forward the request to the server
        response = await client.post(f"{server}/completions", json=json.loads(body))

        # Mark the server as not in use
        server_status[server] = False

        # Write the response to the user's queue
        await user_queues[user].put(response.json())

        return response.json()

def consume_queue():
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
    channel = connection.channel()

    # Declare a queue
    channel.queue_declare(queue='requests')

    while True:
        # Consume one message from the queue
        method_frame, header_frame, body = channel.basic_get('requests')

        if method_frame:
            # Extract the user from the request
            request = json.loads(body)
            user = request['user']

            # Find an available server
            for server in servers:
                if not server_status[server]:
                    # Mark the server as in use
                    server_status[server] = True

                    # Handle the request in a new task
                    asyncio.create_task(handle_request(body, server, user))

                    # Acknowledge the message
                    channel.basic_ack(method_frame.delivery_tag)

                    break

# Start the queue consumer when the application starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(consume_queue())

@app.post("/completions")
async def handle_chat_completion_request(request: ChatCompletionRequest):
    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
        channel = connection.channel()

        # Declare a queue
        channel.queue_declare(queue='requests')

        # Create a unique queue for this user
        user_queues[request.user] = asyncio.Queue()

        # Publish the request to the queue
        channel.basic_publish(exchange='', routing_key='requests', body=json.dumps(request.dict()))

        # Create a generator that yields the responses as they are received from the server
        async def stream_responses():
            while True:
                response = await user_queues[request.user].get()
                yield response

        return StreamingResponse(stream_responses())
    except Exception as e:
        return {"error": str(e)}

# TIMEOUT_KEEP_ALIVE = 5  # seconds
# served_model = None
# engine = None


# def create_error_response(status_code: HTTPStatus,
#                           message: str) -> JSONResponse:
#     return JSONResponse(schemas.ErrorResponse(message=message,
#                                       type="invalid_request_error").dict(),
#                         status_code=status_code.value)



# async def check_model(request) -> Optional[JSONResponse]:
#     if request.model == served_model:
#         return
#     ret = create_error_response(
#         HTTPStatus.NOT_FOUND,
#         f"The model `{request.model}` does not exist.",
#     )
#     return ret



# async def check_length(
#     request: Union[schemas.ChatCompletionRequest, schemas.CompletionRequest],
#     prompt: Optional[str] = None,
#     prompt_ids: Optional[List[int]] = None
# ) -> Tuple[List[int], Optional[JSONResponse]]:
#     assert (not (prompt is None and prompt_ids is None)
#             and not (prompt is not None and prompt_ids is not None)
#             ), "Either prompt or prompt_ids should be provided."
#     if prompt_ids is not None:
#         input_ids = prompt_ids
#     else:
#         input_ids = tokenizer(prompt).input_ids
#     token_num = len(input_ids)

#     if request.max_tokens is None:
#         request.max_tokens = max_model_len - token_num
#     if token_num + request.max_tokens > max_model_len:
#         return input_ids, create_error_response(
#             HTTPStatus.BAD_REQUEST,
#             f"This model's maximum context length is {max_model_len} tokens. "
#             f"However, you requested {request.max_tokens + token_num} tokens "
#             f"({token_num} in the messages, "
#             f"{request.max_tokens} in the completion). "
#             f"Please reduce the length of the messages or completion.",
#         )
#     else:
#         return input_ids, None


# @router.post("/chat", status_code=200)
# async def chat_endpoint(request: schemas.ChatRequest):
#     try:
#         chat_service = ChatService()
#         chat_service.start_service(request)
#         return StreamingResponse(chat_service.stream_results(request.session_id), media_type="text/plain")
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
#         raise


# def convert_request(request: schemas.ChatCompletionRequest, return_all=True, return_both=False):
#     if isinstance(request.messages, list):
#         if return_all:
#             return '\n'.join([msg['content'] for msg in request.messages])
#         elif return_both:
#             return '\n'.join([msg['content'] for msg in request.messages]), request.messages[-1]['content']
#         else:
#             return request.messages[-1]['content']
#     return request.messages

# @router.post("/completions")
# async def handle_chat_completion_request(request:schemas.ChatCompletionRequest):
#     try:
        
#         #endpoint receives HTTPS request
#         #Chatservice adds request to queue
#         #queue routes (sends HTTP request to appropriate endpoint) to available machines based on list of IPs and Ports and availability
#         #StreamingResponse received by the machine is sent back to user that has called it (based on users id



#         chat_service = ChatService()

#         # Convert ChatCompletionRequest to normal request (query)
#         query = convert_request(request)
#         chat_request = schemas.ChatRequest(
#             query=query,
#             session_id=request.user,
#             api_key= request.api_key,
#             model=request.model,
#             temperature=request.temperature,
#             max_tokens=request.max_tokens
#         )

#         # Send converted request in appropriate format. 
#         chat_service.start_service(chat_request)
#         # Make a new stream_results which does it in the ChatResponse format
#         response = chat_service.stream_results_oai(request.user)

#         # Check if the response is empty
#         if not response:
#             # If it's empty, send a default message
#             response = "Sorry, I couldn't find an answer to your request."

#         return StreamingResponse(response, media_type="text/plain")
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
#         raise

