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

class ModelService:
    _instance = None
    
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
    #     hfparser = transformers.HfArgumentParser((
    #     args.ModelArguments, args.DataArguments, args.TrainingArguments, args.GenerationArguments
    # ))
    #     model_args, data_args, training_args, generation_args, extra_args = \
    #     hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    #     training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    #     args = argparse.Namespace(
    #         **vars(model_args), **vars(data_args), **vars(training_args)
    #     )
        # print(args)
        torch.set_default_device('cuda')
        self.init_base_model(config)

        
        # # Initialize Mixture of Experts models
        # self.expert_models = {}
        # self.expert_tokenizers = {}
        # for cluster_id, path in config['expert_model_paths'].items():
        #     self.expert_models[cluster_id] = AutoModelForCausalLM.from_pretrained(path)
        #     self.expert_tokenizers[cluster_id] = AutoTokenizer.from_pretrained(path)
        #     self.expert_tokenizers[cluster_id].bos_token_id = 1

    def get_model(self):
        return self.base_model, self.base_tokenizer

    def get_expert_model(self, cluster_id):
        if cluster_id not in self.expert_models:
            # Handle this case, perhaps by logging or raising an exception
            return None, None
        return self.expert_models[cluster_id], self.expert_tokenizers[cluster_id]
