from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

class ModelService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.init_model()
        return cls._instance

    def init_model(self):
        torch.set_default_device('cuda')
        self.model = AutoModelForCausalLM.from_pretrained("SkunkworksAI/Mistralic-7B-1")
        self.tokenizer = AutoTokenizer.from_pretrained("SkunkworksAI/Mistralic-7B-1")
        self.tokenizer.bos_token_id = 1

    def get_model(self):
        return self.model, self.tokenizer
