import wandb
import argparse
import yaml
import shutil
from subprocess import call
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Wandb sweep id for decentralized sweeping. If not provided, a new sweep will be created.",
    )

    parser.add_argument(
        "--gpu",
        type=list,
        default=None,
        help="List of CUDA device ids to use for training. If not provided, all available GPUs will be used.",
    )

    parser.add_argument(
        "--sweep_config",
        type=str,
        default="configs/finetuning/ft_sweep_config.yaml",
        help="Path to sweep config yaml file. Ignored if sweep_id is provided.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="AblateIt-Sweeps",
        help="Wandb project name.",
    )

    parser.add_argument(
        "--default_training_config",
        type=str,
        default="configs/finetuning/default_ft_config.yaml",
        help="Path to default training args yaml file.",
    )


class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            config = Config(config_dict)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def finetune_sweep():
    args = get_args()
    sweep_id = args.sweep_id

    if not sweep_id:
        sweep_config = yaml.safe_load(open(args.sweep_config))["wandb_args"]
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print("Sweep ID: ", sweep_id)
        with open("sweep_id.txt", "w") as file:
            file.write(sweep_id)

    def run_finetune():
        train_config = load_config(args.default_training_config)
        wandb.init(entity=args.entity)
        wandb_config = dict(wandb.config)

        cluster_n = wandb_config.pop("cluster")
        model_name = f"expert-{cluster_n}"
        wandb.run.name = model_name
        split_name = f"config{cluster_n}"

        train_config["run_name"] = model_name
        train_config["output_dir"] = f"experts/{model_name}"
        train_config["hub_model_id"] = f"HydraLM/{model_name}"
        train_config["wandb_project"] = args.project
        train_config["wandb_entity"] = args.entity
        train_config["wandb_run_name"] = model_name
        train_config["split"] = split_name

        cuda_device_declaration = (
            "export CUDA_VISIBLE_DEVICES=" + ",".join([str(x) for x in args.gpu]) + "; "
            if args.gpu
            else ""
        )

        command = cuda_device_declaration + "python finetuner.py "
        for key, value in vars(train_config).items():
            command += f"--{key} {value} "
        print(f"Command:\n{command}")
        call(command, shell=True)

    if args.sweep_id is not None:
        # Run the sweep
        wandb.agent(sweep_id, run_finetune, project=args.project, entity=args.entity)

if __name__ == "__main__":
    finetune_sweep()








def finetuner_runner(config_file):
    config = load_config(config_file)
    model_name = config.model_name_or_path.split("/")[1]
    if "/" in config.dataset:
        dataset_name = config.dataset.split("/")[1]
    else:
        dataset_name = config.dataset
    config.output_dir = f"{config.output_dir}_{model_name}_{dataset_name}"
    config.hub_model_id = f"{config.hub_model_id}/{model_name}_{dataset_name}"
    command = "python finetuner.py "
    for key, value in vars(config).items():
        command += f"--{key} {value} "
    print(f"Command:\n{command.split(' ')}")
    subprocess.run(command, shell=True)