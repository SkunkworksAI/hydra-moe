from abc import ABC, abstractmethod

class InferenceStrategy(ABC):

    @abstractmethod
    def perform_inference(self, message, conversation_id, model, tokenizer, publish_function):
        pass
