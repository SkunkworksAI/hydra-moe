from transformers import BertTokenizer, AutoModel, AutoTokenizer, BertForSequenceClassification,AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from utils import *
from moe_utils import *
import json
from pathlib import Path, PosixPath
import numpy as np
from peft_model import PeftModel
#from moe.alpha import PeftModel
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from sentence_transformers import SentenceTransformer

# model = None
model_class = None
tokenizer = None
embedding_model = None
#load tfidf vectorizer
ROOT_DIR = Path('.').resolve().parent

def get_inference_model(
    config, checkpoint_dirs: dict[str, str], base_model=None):
    """
    Load the adaptors from the `checkpoint_dirs` onto the base model

    If `base_model` is provided, will load on to `base_model`,
    otherwise will call `get_base_inference_model` to load the base
    """
    if base_model is None:
        base_model, base_tokenizer = get_base_inference_model(config)

    if checkpoint_dirs is not None:
        print("Loading Experts.")
        for checkpoint in checkpoint_dirs:
            checkpoint_dir = checkpoint["adapter_dir"]
            adapter_name = checkpoint["adapter_name"]
            print(f"Loading Expert #{adapter_name} from {checkpoint_dir}.")
            if checkpoint == checkpoint_dirs[0]:
                # Load the first adapter with from_pretrained
                model = PeftModel.from_pretrained(base_model, checkpoint_dir, adapter_name=adapter_name)
            else:
                # Load the remaining adapters with load_adapter
                model.load_adapter(checkpoint_dir, adapter_name=adapter_name)
    """

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    """

    return model


def get_base_inference_model(config):

    n_gpus = torch.cuda.device_count()
    #dmax_memory = f'{config.max_memory}MB'
    #max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        #max_memory = {'': max_memory[local_rank]}

    print(f'loading base model {config.model_name_or_path}...')

    compute_dtype = torch.float16
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        load_in_4bit=config.bits == 4,
        load_in_8bit=config.bits == 8,
        device_map=device_map,
        #max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=config.bits == 4,
            load_in_8bit=config.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_quant_type=config.quant_type,
        ),

        torch_dtype=torch_dtype,
    )
    if compute_dtype == torch.float16 and config.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    # why?
    # model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in config.model_name_or_path else None, # Needed for HF name change
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })

    #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    """

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    """

    return model, tokenizer

def load_centroids(centroids_pickle_path: PosixPath) -> dict:
    """Loads the centroids for the Experts from a pickle file
    The pkl file shoudl contain a dict like {adapter_name: adapter_centroid}

    Args:
        centroids_pickle_path (posixpath): path to the centroids file

    Returns:
        dict  {adapter_name: adapter_centroid}
    """
    centroids_pickle_path = ROOT_DIR/'hydra-moe/gating_v2/cluster_centers.pkl'
    with open(centroids_pickle_path, 'rb') as f:
        centroids_array = pickle.load(f)

    # TODO: we should probably change the format of `cluster_centers.pkl`
    # so that we have each centroid mapped to an adapter name
    # instead of having to "guess" here which centroid goes with which adapter

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
        model_path = os.path.join(ROOT_DIR, 'hydra-moe', 'gating_v2', 'model')

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
    # adapter_names = [
    #    f"cluster_{str(cluster).zfill(3) if cluster >= 10 else str(cluster).zfill(2)}" for cluster in cluster_nums
    # ]

    #TODO: adapter names should come from file

    adapter_names = [
       f"{str(cluster)}" for cluster in cluster_nums  # TODO: need to change this to pull the "right" set of adapters
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

def get_weights(instruction, method="combined", centroids_pickle_path: PosixPath = None):
    global centroids
    global embedding_model
    if centroids is None:
        print("LOADING CENTROIDS")
        centroids = load_centroids(centroids_pickle_path)

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
        # print(probs)

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
    # print(d)
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
    # print(weights)
    return weights
