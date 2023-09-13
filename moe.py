# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from args import *
from utils import *
from moe_utils import *
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import yaml

from moe_utils import get_inference_model

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
)

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
model = None
tokenizer = None
centroids = {}
kmeans_centroids = {}

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
            config = Config(config_dict)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def inference():

    parser = argparse.ArgumentParser(
            prog="moe.py",
            description="run MoE inference")

    parser.add_argument('--config_file', type=str, default="configs/inference_config.yaml", help="path of config file. e.g. configs/inference_config.yaml")
    args, unknown_args = parser.parse_known_args()

    if args.config_file:
        # Load config from file
        config = load_config(args.config_file)
    else:
        # Parse remaining arguments as key-value pairs
        config = {}
        print(unknown_args)
        for arg in unknown_args:
            key, value = arg.split("=")
            config[key] = value
        config = Config(config)


    cluster_nums = range(32)
    checkpoint_dirs = [
        {
            "adapter_dir": f"HydraLM/Nous-Hermes-llama-2-7b_7b_cluster{str(cluster).zfill(3) if cluster >= 10 else str(cluster).zfill(2)}_partitioned_v3_standardized_{str(cluster).zfill(3) if cluster >= 10 else str(cluster).zfill(2)}",
            "adapter_name": f"{str(cluster)}"
        }
        for cluster in cluster_nums
    ]

    #Load PEFT adapters to model
    model, tokenizer = get_inference_model(config, checkpoint_dirs)
    #base_model, base_tokenizer = get_base_inference_model(args, checkpoint_dirs)

    model.config.use_cache = False
    #base_model.config.use_cache = False
    print('loaded model')
    #set_seed(args.seed)

    # Verifying the datatypes and parameter counts before training.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    logger.info("*** Predict ***")

    def generate_prompt(instruction, input=None):
        prompt = f"### Instruction:\n{instruction}\n\n"
        if input:
            prompt += f"### Input:\n{input}\n\n"
        return prompt + "### Response:\n"

    def generate_output(instruction, model, alphas, tokenizer, generation_args, count = 320):
        prompt = generate_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        print(f'Updating alphas to {alphas}')
        model.update_alphas(alphas)


        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                max_length=count,
                max_new_tokens = count,
                #do_sample=config.do_sample,
                #num_beams=generation_args.num_beams,
                #temperature=generation_args.temperature,
                #top_k=generation_args.top_k,
                #top_p=generation_args.top_p,
                #repetition_penalty=generation_args.repetition_penalty,
                #length_penalty=generation_args.length_penalty,
                #no_repeat_ngram_size=generation_args.no_repeat_ngram_size,
                num_return_sequences=1,
            )
        output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
        return output


    def generate_base_output(instruction, model, alphas, tokenizer, generation_args, count = 320):
        prompt = generate_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                max_length=count,
                max_new_tokens = count,
                do_sample=generation_args.do_sample,
                num_beams=generation_args.num_beams,
                temperature=generation_args.temperature,
                top_k=generation_args.top_k,
                top_p=generation_args.top_p,
                repetition_penalty=generation_args.repetition_penalty,
                length_penalty=generation_args.length_penalty,
                no_repeat_ngram_size=generation_args.no_repeat_ngram_size,
                num_return_sequences=1,
            )
        output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
        return output
    # load_kmeans()
    # load_centroid()
    load_gating32()
    while True:
        # Get user input
        instruction = input("Enter your instruction: ")
        methodIn = input("Enter your mode: transformer, centroid, combined ")
        alphaIn = 16
        expertsK = 3

        #methods: combined, transformer, multi, kmeans, centroid
        weights = get_weights(instruction, methodIn)
        # weights = get_weights(instruction, "transformer")

        alphas = mult_weights_by_alpha(weights, int(alphaIn), int(expertsK) )
        # alphas = mult_weights_by_alpha(weights, training_args.lora_alpha)

        print("Predicted Weights:")
        [print(k, v) for k, v in weights.items()]

        output = generate_output(instruction, model, alphas, tokenizer, generation_args)
        print("MoE Model:")
        print(output)

        #output_base = generate_base_output(instruction, base_model, alphas, base_tokenizer, generation_args)
        #print("Base Model:")
        #print(output_base)

        continue_prompt = input("Do you want to continue? (yes/no): ")
        if continue_prompt.lower() != "yes":
            break

if __name__ == "__main__":
    inference()
