from typing import Optional, Tuple, List
from http import HTTPStatus
from pydantic import BaseModel
from loguru import logger 

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None

def create_error_response(status_code: HTTPStatus,
                          message: str) -> ErrorResponse:
    return ErrorResponse(message=message,type="invalid_request_error").model_dump()

def check_length(
    tokenizer,
    max_new_tokens: int, 
    max_model_len: int,
    prompt: str,
) -> ErrorResponse:
    
    input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Token num: {token_num}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    logger.info(f"Max model len: {max_model_len}")
    if token_num + max_new_tokens > max_model_len:
        logger.warning("Token input and max_new_tokens exceed max_model_len")
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"""This model's maximum context length is {max_model_len} tokens. "
            However, you requested {max_new_tokens + token_num} tokens 
            ({token_num} in the messages, 
            {max_new_tokens} in the completion). 
            Please reduce the length of the messages or completion."""
        )
    else:
        return None