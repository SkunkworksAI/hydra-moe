import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlamaTokenizer,
    BitsAndBytesConfig,
)

from .peft_model import PeftModel

def get_inference_model(config, checkpoint_dirs):

    n_gpus = torch.cuda.device_count()
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

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

    if checkpoint_dirs is not None:
        print("Loading Experts.")
        for checkpoint in checkpoint_dirs:
            checkpoint_dir = checkpoint["adapter_dir"]
            adapter_name = checkpoint["adapter_name"]
            print(f"Loading Expert #{adapter_name} from {checkpoint_dir}.")
            if checkpoint == checkpoint_dirs[0]:
                # Load the first adapter with from_pretrained
                model = PeftModel.from_pretrained(model, checkpoint_dir, adapter_name=adapter_name)
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
    
    return model, tokenizer



def get_base_inference_model(config, checkpoint_dirs):

    n_gpus = torch.cuda.device_count()
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    print(f'loading base model {config.model_name_or_path}...')

    compute_dtype = torch.float16
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        load_in_4bit=config.bits == 4,
        load_in_8bit=config.bits == 8,
        device_map=device_map,
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
    
    return model, tokenizer