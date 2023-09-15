from utils import AttributeDict
import argparse
import os
import subprocess
import yaml


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


def inference_runner(config_file):
    config = load_config(config_file)
    model_name = config.model_name_or_path.split("/")[1]
    if "/" in config.dataset:
        dataset_name = config.dataset.split("/")[1]
    else:
        dataset_name = config.dataset
    config.output_dir = f"{config.output_dir}_{model_name}_{dataset_name}"
    command = "python moe.py "
    for key, value in vars(config).items():
        command += f"--{key} {value} "
    print(f"Command:\n{command.split(' ')}")
    subprocess.run(command, shell=True)


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

def webui_runner(config_file):
    config = load_config(config_file)
    model_name = config.model_name_or_path.split("/")[1]
    if "/" in config.dataset:
        dataset_name = config.dataset.split("/")[1]
    else:
        dataset_name = config.dataset
    config.output_dir = f"{config.output_dir}_{model_name}_{dataset_name}"
    command = "python server.py "
    for key, value in vars(config).items():
        command += f"--{key} {value} "
    print(f"Command:\n{command.split(' ')}")
    subprocess.run(command, shell=True)

def main():
    parser = argparse.ArgumentParser(description="MoE")
    parser.add_argument("--finetune", action="store_true", help="Finetune? T/F")
    parser.add_argument("--inference", action="store_true", help="Inference? T/F")
    parser.add_argument("--webui", action="store_true", help="Webui? T/F")
    parser.add_argument(
        "--config", type=str, required=False, help="Path to YAML config file"
    )
    args = parser.parse_args()

    if not args.config:
        if args.finetune:
            config_file = "configs/default_ft_config.yaml"
        elif args.inference:
            config_file = "configs/inference_config.yaml"
        elif args.webui:
            config_file = "configs/inference_config.yaml"
    else:
        config_file = args.config

    if args.finetune:
        finetuner_runner(config_file)
    if args.inference:
        inference_runner(config_file)
    if args.webui:
        webui_runner(config_file)


if __name__ == "__main__":
    main()
