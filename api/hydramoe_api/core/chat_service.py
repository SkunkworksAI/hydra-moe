from uuid import uuid4
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from loguru import logger 
from pydantic import BaseModel, Field
from typing import Any
from threading import Event, Thread

class ChatKwargs(BaseModel):
    input_ids: Any = Field(..., description="Expert IDs for the model")
    max_new_tokens: int = Field(..., description="Maximum number of new tokens to be generated")
    temperature: float = Field(..., description="Temperature for randomness")
    do_sample: bool = Field(..., description="Whether to do sampling or not")
    top_p: float = Field(..., description="Top p for nucleus sampling")
    top_k: int = Field(..., description="Top k for top-k sampling")
    repetition_penalty: float = Field(..., description="Penalty for repeated tokens")
    stopping_criteria: Any = Field(..., description="Criteria for stopping the model")
    eos_token_id: int = Field(..., description="End-of-sentence token ID")


class ChatService:

    def __init__(self):
        self.init_model()
        
    def init_model(self):
        torch.set_default_device('cuda')
        self.model = AutoModelForCausalLM.from_pretrained("SkunkworksAI/Mistralic-7B-1")
        self.tokenizer = AutoTokenizer.from_pretrained("SkunkworksAI/Mistralic-7B-1")
        self.tokenizer.bos_token_id = 1

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
            yield new_text
        return model_output

            
    def submit_query(self, request):
        conversation_id = str(uuid4())
        return self.bot(request.query, request.session_id)
    
