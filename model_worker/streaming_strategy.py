import torch
from transformers import StoppingCriteria, TextIteratorStreamer, StoppingCriteriaList
from threading import Thread, Event
from loguru import logger
from inference_strategy import InferenceStrategy
from typing import Callable


class CompletionStreamingStrategy(InferenceStrategy):
    """CompletionStreamingStrategy is an implementation of the InferenceStrategy for streaming-based inference.

    This strategy uses a thread to perform the generation task asynchronously and an Event to signal
    the completion of the streaming. It utilizes a specific stopping criterion and a text streamer.
    """

    class StopOnTokens(StoppingCriteria):
        """Inner class for defining custom stopping criteria based on token IDs."""

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            """Stop the generation if any of the stop_token_ids are found.
            Args:
                input_ids (torch.LongTensor): Tensor of token IDs.
                scores (torch.FloatTensor): Tensor of token scores.
            Returns:
                bool: True if stopping criteria met, False otherwise.
            """
            stop_token_ids = [0]
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def convert_to_text(self, msg: str) -> str:
        """Convert the given message to a formatted text prompt for the model.
        Args:
            msg (str): Instruction message.
        Returns:
            str: Formatted text for model input.
        """
        system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        text = system_prompt
        text += f"### Instruction:\n{msg}\n\n"
        text += "### Response:\n"
        return text

    def perform_inference(
        self,
        message: str,
        conversation_id: str,
        model,
        tokenizer: Callable,
        publish_function: Callable,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: float,
        repetition_penalty: float,
    ) -> None:
        """Perform the inference operation using a streaming strategy.

        Args:
            message (str): The message to be processed.
            conversation_id (str): The ID of the conversation.
            model (AutoModel): The language model.
            tokenizer (Callable): The tokenizer function.
            publish_function (Callable): Function to publish the model output.
        """
        stop = self.StopOnTokens()

        messages = self.convert_to_text(message)

        input_ids = tokenizer(messages, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )

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
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info(f"Inference inputs : {generate_kwargs}")
        model_output = ""

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
