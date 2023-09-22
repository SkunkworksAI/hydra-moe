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
import torch.nn.functional as F
import json
import pickle

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from sentence_transformers import SentenceTransformer

import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,

)
from transformers import BertTokenizer, AutoModel, AutoTokenizer, BertForSequenceClassification,AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from .peft_model import PeftModel

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import subprocess
import torch.hub

cluster_nums = range(32)
# model = None
model_class = None
tokenizer = None
centroids = None
embedding_model = None
kmeans_centroids = {}
#load tfidf vectorizer
root_dir = os.path.abspath(os.pardir)  
gte = None


ROUTER_FILES_PATH = 'router_files'


def load_gating32():
    global centroids
    centroids_pickle_path = os.path.join(ROUTER_FILES_PATH, 'cluster_centers.pkl')
    with open(centroids_pickle_path, 'rb') as f:
        centroids_array = pickle.load(f)
    centroids = {f"cluster_{i}": centroids_array[i] for i in range(centroids_array.shape[0])}
    return centroids


def chunk_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def chunk_and_get_embeddings(model, data_list, batch_size):
    chunks = chunk_list(data_list, batch_size)
    embeddings = []
    for chunk in chunks:
        emb = model.encode(chunk)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings

def select_adapter_centroid32(instruction, centroids, embedding_model):
    instruction_embedding = chunk_and_get_embeddings(embedding_model, [instruction], 1)

    similarities = []
    adapter_names = []

    for name, centroid in centroids.items():
        similarity = cosine_similarity(instruction_embedding, centroid.reshape(1, -1))
        similarities.append(similarity)
        adapter_names.append(name)

    return similarities, adapter_names


def select_adapter_classifier(instruction):
    global model_class
    global tokenizer

    if model_class is None:
        model_path = os.path.join(root_dir, 'hydra-moe', ROUTER_FILES_PATH, 'model')
        model_class = AutoModelForSequenceClassification.from_pretrained('HydraLM/e5-large-32-32000')
        model_class = model_class.to('cuda')


    if torch.cuda.is_available():
        model_class = model_class.to('cuda')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')


    encoding = tokenizer.encode_plus(
        "query: " + instruction, # e5 requires "query: " to be added to the beginning for classification
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    encoding = {key: val.to(model_class.device) for key, val in encoding.items()}
    outputs = model_class(**encoding)
    probs = F.softmax(outputs.logits, dim=1)
    _, predicted_indices = torch.max(outputs.logits, dim=1)
    # cluster_nums = range(32)  # adjust this to match the number of clusters in your data
    # adapter_names = [
    #    f"cluster_{str(cluster).zfill(3) if cluster >= 10 else str(cluster).zfill(2)}" for cluster in cluster_nums
    # ]
    adapter_names = [
       f"{str(cluster)}" for cluster in cluster_nums
    ]

    predicted_class = adapter_names[predicted_indices.item()]

    return probs, adapter_names

def combine_all_predictions(transformer_pred, transformer_conf, centroid_pred, kmeans_pred, multi_pred, embeds_pred, transformer_weight=2):
   
    adapter_names = [
       f"{str(cluster)}" for cluster in cluster_nums
        ]
    votes = [0] * len(adapter_names)

    votes[transformer_pred] += transformer_weight
    votes[centroid_pred] += 1
    votes[kmeans_pred] += 1
    votes[multi_pred] += 1
    votes[embeds_pred] += 1

    ranked_classes = sorted(zip(adapter_names, votes), key=lambda x: x[1], reverse=True)

    return votes.index(max(votes)), ranked_classes

def get_weights(instruction, method="centroids"):
    global centroids
    global gte
    global embedding_model
    if centroids is None:
            print("LOADING CENTROIDS")
            load_gating32()

    if embedding_model is None:
        embedding_model = SentenceTransformer('all-mpnet-base-v2', device="cuda")

    def normalize(probs):
        total = sum(probs)
        return [prob / total for prob in probs]

    def normalize_neg(probs):
        min_val = min(probs)
        shifted_probs = [prob - min_val for prob in probs]  
        total = sum(shifted_probs)
        return [prob / total for prob in shifted_probs]  

    if method == "centroid":
        probs, adapter_names = select_adapter_centroid32(instruction, centroids, embedding_model)
        d = {}
        probs = normalize(probs)

        print(probs)
        for i, name in enumerate(adapter_names):
            if '_' in name:
                name = name.split('_')[1]
            if name.startswith('0') and len(name) > 1:
                name = name[1:]

            d[name] = probs[i][0][0]

    elif method == "transformer":
        probs, adapter_names = select_adapter_classifier(instruction)
        d = {adapter_names[i]: probs[0][i] for i in range(len(probs[0]))}
        print(d)

    elif method == "combined":
        probs, adapter_names = select_adapter_centroid32(instruction, centroids, embedding_model)
        probs = normalize(probs)
        d = {}

        for i, name in enumerate(adapter_names):
            if '_' in name:
                name = name.split('_')[1]
            if name.startswith('0') and len(name) > 1:
                name = name[1:]
            d[name] = probs[i][0][0]

        probs, adapter_names  = select_adapter_classifier(instruction)

        for i, name in enumerate(adapter_names):
            d[name] += probs[0][i]

        _sum = sum([i for i in d.values()])
        for k, v in d.items():
            d[k] = v / _sum
    else:
        raise ValueError("Invalid method. Choose from 'centroid', 'kmeans', 'multi', 'transformer', 'combined'")

    return d

def mult_weights_by_alpha(weights: dict, alpha, k=3):
    top_k_values = sorted(weights.values(), reverse=True)[:k]

    for key in weights:
        if weights[key] not in top_k_values:
            weights[key] = 0

    max_weight = max(weights.values())
    min_weight = min(weights.values())
    for key in weights:
        weights[key] = (weights[key] - min_weight) / (max_weight - min_weight)
    for key in weights:
        weights[key] *= alpha

    return weights
