"""
This module defines API endpoints related to chat such as fetching, creating, and updating chat sessions
"""
from loguru import logger
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from hydramoe_api.core.chat_service import ChatService
from hydramoe_api import schemas
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

router = APIRouter()

# try:
#     import fastchat
#     from fastchat.conversation import Conversation, SeparatorStyle
#     from fastchat.model.model_adapter import get_conversation_template
#     _fastchat_available = True
# except ImportError:
#     _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds

served_model = None
engine = None


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(schemas.ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)



async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


# async def get_gen_prompt(request) -> str:
#     if not _fastchat_available:
#         raise ModuleNotFoundError(
#             "fastchat is not installed. Please install fastchat to use "
#             "the chat completion and conversation APIs: `$ pip install fschat`"
#         )
#     if version.parse(fastchat.__version__) < version.parse("0.2.23"):
#         raise ImportError(
#             f"fastchat version is low. Current version: {fastchat.__version__} "
#             "Please upgrade fastchat to use: `$ pip install -U fschat`")

#     conv = get_conversation_template(request.model)
#     conv = Conversation(
#         name=conv.name,
#         system_template=conv.system_template,
#         system_message=conv.system_message,
#         roles=conv.roles,
#         messages=list(conv.messages),  # prevent in-place modification
#         offset=conv.offset,
#         sep_style=SeparatorStyle(conv.sep_style),
#         sep=conv.sep,
#         sep2=conv.sep2,
#         stop_str=conv.stop_str,
#         stop_token_ids=conv.stop_token_ids,
#     )

#     if isinstance(request.messages, str):
#         prompt = request.messages
#     else:
#         for message in request.messages:
#             msg_role = message["role"]
#             if msg_role == "system":
#                 conv.system_message = message["content"]
#             elif msg_role == "user":
#                 conv.append_message(conv.roles[0], message["content"])
#             elif msg_role == "assistant":
#                 conv.append_message(conv.roles[1], message["content"])
#             else:
#                 raise ValueError(f"Unknown role: {msg_role}")

#         # Add a blank message for the assistant.
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#     return prompt


async def check_length(
    request: Union[schemas.ChatCompletionRequest, schemas.CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    if prompt_ids is not None:
        input_ids = prompt_ids
    else:
        input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None

@router.post("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
         schemas.ModelCard(id=served_model,
                  root=served_model,
                  permission=[schemas.ModelPermission()])
    ]
    return schemas.ModelList(data=model_cards)


# def create_logprobs(token_ids: List[int],
#                     id_logprobs: List[Dict[int, float]],
#                     initial_text_offset: int = 0) -> LogProbs:
#     """Create OpenAI-style logprobs."""
#     logprobs = LogProbs()
#     last_token_len = 0
#     for token_id, id_logprob in zip(token_ids, id_logprobs):
#         token = tokenizer.convert_ids_to_tokens(token_id)
#         logprobs.tokens.append(token)
#         logprobs.token_logprobs.append(id_logprob[token_id])
#         if len(logprobs.text_offset) == 0:
#             logprobs.text_offset.append(initial_text_offset)
#         else:
#             logprobs.text_offset.append(logprobs.text_offset[-1] +
#                                         last_token_len)
#         last_token_len = len(token)

#         logprobs.top_logprobs.append({
#             tokenizer.convert_ids_to_tokens(i): p
#             for i, p in id_logprob.items()
#         })
#     return logprobs







@router.post("/chat", status_code=200)
async def chat_endpoint(request: schemas.ChatRequest):
    try:
        chat_service = ChatService()
        chat_service.start_service(request)
        return StreamingResponse(chat_service.stream_results(request.session_id), media_type="text/plain")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise





def convert_request(request: schemas.ChatCompletionRequest, return_all=True, return_both=False):
    if isinstance(request.messages, list):
        if return_all:
            return '\n'.join([msg['content'] for msg in request.messages])
        elif return_both:
            return '\n'.join([msg['content'] for msg in request.messages]), request.messages[-1]['content']
        else:
            return request.messages[-1]['content']
    return request.messages

@router.post("/completions")
async def handle_chat_completion_request(request:schemas.ChatCompletionRequest):
    try:
        chat_service = ChatService()

        # Convert ChatCompletionRequest to normal request (query)
        query = convert_request(request)
        chat_request = schemas.ChatRequest(
            query=query,
            session_id=request.user,
            api_key= request.api_key,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Send converted request in appropriate format. 
        chat_service.start_service(chat_request)

        # Make a new stream_results which does it in the ChatResponse format
        return StreamingResponse(chat_service.stream_results_oai(request.user), media_type="text/plain")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


# @router.post("/completions")
# async def handle_chat_completion_request(request:schemas.ChatCompletionRequest):
#     """
#     This function handles the chat completion request. It takes a ChatCompletionRequest
#     as input and returns a response.
#     """
#     # Validate the request
# #     error_response = await check_model(request)
# #     if error_response is not None:
# #         return error_response

# # =    prompt = await get_gen_prompt(request)

# #     token_ids, error_response = await check_length(request, prompt=prompt)
# #     if error_response is not None:
# #         return error_response

#     try:
#         chat_service = ChatService()

#         #Convert ChatCompletionRequest to normal request.


#         chat_service.start_service(request)

#         #make a new stream_results which does it in the ChatResponse format
#         return StreamingResponse(chat_service.stream_results(request.session_id), media_type="text/plain")
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
        # raise
