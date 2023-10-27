from dataclasses import dataclass

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
