import httpx
import json
from typing import Dict, List, Literal, Optional, Union
import time
from pydantic import BaseModel, Field
import uuid 
import re 
def random_uuid() -> str:
    return str(uuid.uuid4().hex)

url = 'http://127.0.0.1:8000/api/v1/chat/completions'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

sessionID = "484320011"

messages = []

while True: 
    textIn = input("\n\nEnter: ")
    messages.append({"role": "user", "content": textIn})
    payload = {
        'model': 'string',
        'messages': messages,
        'temperature': 0.7,
        'top_p': 1.0,
        'n': 1,
        'max_tokens': 1000,
        'stop': [],
        'stream': False,
        'presence_penalty': 0.0,
        'frequency_penalty': 0.0,
        'logit_bias': None,
        'api_key': 'string',
        'user': sessionID
    }

    # with httpx.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
    #     if response.status_code == 200:
    #         for chunk in response.iter_text():
    #             print(chunk, end='', flush=True)   
    #     else:
    #         print(f"Failed: {response.status_code}")

    class DeltaMessage(BaseModel):
        role: Optional[str] = None
        content: Optional[str] = None


    class ChatCompletionResponseStreamChoice(BaseModel):
        index: int
        delta: DeltaMessage
        finish_reason: Optional[Literal["stop", "length"]] = None


    class ChatCompletionStreamResponse(BaseModel):
        id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
        object: str = "chat.completion.chunk"
        created: int = Field(default_factory=lambda: int(time.time()))
        model: str
        choices: List[ChatCompletionResponseStreamChoice]
    bot = ""
    with httpx.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
        if response.status_code == 200:
            for chunk in response.iter_text():
                # print(f"Chunk: {chunk}")  # Debug line
                if chunk:  # Check if chunk is not empty
                    try: 
                        chunk = re.sub('^data: ', '', chunk)
                    except:
                        pass
                    response_obj = ChatCompletionStreamResponse.parse_raw(chunk)
                    for choice in response_obj.choices:
                        print(choice.delta.content, end='', flush=True)
                        bot += choice.delta.content

                else:
                    print("Received an empty chunk.")
        else:
            print(f"Failed: {response.status_code}")
    messages.append({"role": "bot", "content": bot})

    # with httpx.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
    #     if response.status_code == 200:
    #         for chunk in response.iter_text():
    #             print(f"Chunk: {chunk}")  # Debug line

    #             # Parse the response into the ChatCompletionStreamResponse object
    #             response_obj = ChatCompletionStreamResponse.parse_raw(chunk)
    #             # Print the content of each choice
    #             for choice in response_obj.choices:
    #                 print(choice.delta.content, end='', flush=True)
    #     else:
    #         print(f"Failed: {response.status_code}")