from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch 
from dataclasses import dataclass
from loguru import logger
import argparse

# ### Globals 
# import base.args 
# import base.model_utils
# import base.moe_utils 
# IGNORE_INDEX = -100
# model = None
# tokenizer = None
# centroids = {}
# kmeans_centroids = {}
# generation_args = None
# ### 

cluster_nums = range(32)  
checkpoint_dirs = [
{
    "adapter_dir": f"HydraLM/mistralic-expert-{str(cluster)}",
    "adapter_name": f"{str(cluster)}"
}
for cluster in cluster_nums
]

@dataclass 
class ModelConfig:
    base_model_path: str = "SkunkworksAI/Mistralic-7B-1"
    # expert_model_paths: dict = checkpoint_dirs

class Worker:
    model_service: "ModelService" = None

class ModelService:
    _instance = None
    model_pool = []
    request_queue = []
     

    def __new__(cls, config=ModelConfig()):
        if cls._instance is None:
            logger.info("Initializing ModelService")
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.init_models(config)
        return cls._instance
    
    def init_base_model(self, config: ModelConfig): 
        # Initialize base model
        
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_path, trust_remote_code=True, torch_dtype="auto", device_map={"": 0})
        self.base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True, torch_dtype="auto",device_map={"": 0})
        self.base_tokenizer.bos_token_id = 1
        
    def init_models(self, config):
        torch.set_default_device('cuda')
        self.init_base_model(config)


    def get_base_model(self):
        return self.base_model, self.base_tokenizer

