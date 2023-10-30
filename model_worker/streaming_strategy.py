import torch
from transformers import StoppingCriteria, TextIteratorStreamer, StoppingCriteriaList
from threading import Thread, Event
from loguru import logger
from inference_strategy import InferenceStrategy

class CompletionStreamingStrategy(InferenceStrategy):

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_token_ids = [0]
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def convert_to_text(self, msg):
        system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        text = system_prompt
        text += f"### Instruction:\n{msg}\n\n"
        text += "### Response:\n"
        return text

    def perform_inference(self, message, conversation_id, model, tokenizer, publish_function):
        stop = self.StopOnTokens()
        
        messages = self.convert_to_text(message)
        
        input_ids = tokenizer(messages, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        
        max_new_tokens = 1536  
        temperature = 0.01
        top_p = 0.9
        top_k = 0
        repetition_penalty = 1.1
        
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        model_output =  ""
        
        stream_complete = Event()

        def generate_and_signal_complete():
            model.generate(**generate_kwargs)
            stream_complete.set()


        t = Thread(target=generate_and_signal_complete)
        # t = Thread(target=model.generate, kwargs=generate_kwargs)

        t.start()
        # t.join()
        
        # Listen for new tokens and publish
        for new_text in streamer:
            model_output += new_text
            logger.info(f"Model output: {new_text}")
            publish_function(new_text, stream_complete)  
            