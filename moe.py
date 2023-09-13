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
generation_args = None

def initialize_model():
    global model, tokenizer, base_model, base_tokenizer, generation_args
    
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    cluster_nums = range(32)  
    checkpoint_dirs = [
        {
            "adapter_dir": f"HydraLM/Nous-Hermes-llama-2-7b_7b_cluster{str(cluster).zfill(3) if cluster >= 10 else str(cluster).zfill(2)}_partitioned_v3_standardized_{str(cluster).zfill(3) if cluster >= 10 else str(cluster).zfill(2)}",
            "adapter_name": f"{str(cluster)}"
        }
        for cluster in cluster_nums
    ]
    #Load PEFT adapters to model
    print(checkpoint_dirs)
    print(args)
    model, tokenizer = get_inference_model(args, checkpoint_dirs)
    base_model, base_tokenizer = get_base_inference_model(args, checkpoint_dirs)
    
    model.config.use_cache = False
    base_model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

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
    
    # load_kmeans()
    # load_centroid()
    load_gating32()

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

def inference():

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

        output_base = generate_base_output(instruction, base_model, alphas, base_tokenizer, generation_args)
        print("Base Model:")
        print(output_base)

        continue_prompt = input("Do you want to continue? (yes/no): ")
        if continue_prompt.lower() != "yes":
            break
  


if __name__ == "__main__":
    initialize_model()
    inference()
