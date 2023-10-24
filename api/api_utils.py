import httpx
import json
import textwrap
from vllm import SamplingParams
from typing import List 

def get_default_sample_params() -> SamplingParams:
    default_params = {
    "n": 1,
    "best_of": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1.0,
    "early_stopping": False,
    "stop": ["string"],
    "stop_token_ids": [0],
    "ignore_eos": False,
    "max_tokens": 1000,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True
    }

    # Initialize a default instance of SamplingParams
    return SamplingParams(**default_params)
    

def wrap_and_print(text, width=100):
    paragraphs = text.split('\n')
    wrapped_paragraphs = [textwrap.fill(para, width=width) for para in paragraphs]
    return '\n'.join(wrapped_paragraphs)

async def post_request_async(sampling_params: SamplingParams, model:str, messages: List[str]):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "n": sampling_params.n,
        "max_tokens": sampling_params.max_tokens,
        "stop": sampling_params.stop,
        "stream": True,
        "presence_penalty": sampling_params.presence_penalty,
        "frequency_penalty": sampling_params.frequency_penalty,
        "user": "User",
        "best_of": sampling_params.best_of,
        "top_k": sampling_params.top_k,
        "ignore_eos": sampling_params.ignore_eos,
        "use_beam_search": sampling_params.use_beam_search,
        "stop_token_ids": sampling_params.stop_token_ids,
        "skip_special_tokens": sampling_params.skip_special_tokens
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()  # Check if the request was successful

        if response.status_code == 200:
            return json.loads(response.text)

    except httpx.HTTPError as e:
        print(f"An error occurred while making the request: {e}")
        return None
    