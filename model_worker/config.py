from dataclasses import dataclass
import yaml
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
    base_model_path: str
    max_new_tokens: int   
    lora_r: int       
    lora_alpha: int    
    bits: int  
    # expert_model_paths: dict = checkpoint_dirs

def build_model_config_from_yaml(yaml_path: str = '/configs/inference_config.yaml') -> ModelConfig:
    """Creates a ModelConfig object from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file, defaults to '/configs/inference_config.yaml'.

    Returns:
        ModelConfig: A new ModelConfig object populated from the YAML file.
    """

    # Read YAML file
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Create and return a new ModelConfig object
    return ModelConfig(
        base_model_path=config_data.get('model_name_or_path'),
        max_new_tokens=config_data.get('max_new_tokens'),
        lora_r=config_data.get('lora_r'),
        lora_alpha=config_data.get('lora_alpha'),
        bits=config_data.get('bits')
    )