import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

from peft import LoraModel
from peft.tuners.lora import LoraLayer, Linear, Linear4bit, Linear8bitLt, Embedding
from peft.import_utils import is_bnb_4bit_available, is_bnb_available

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

class AlphaLoraModel(LoraModel):
    #Extends LoraModel to provide support for inference with multiple adapters at a time.

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}
        
    def _update_alphas(self, alphas_dict):
        lora_config = list(self.peft_config.items())[0][1]
        self._check_quantization_dependency()

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:

            if not self._check_target_module_exists(lora_config, key):
                continue

            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, AlphaLoraLayer):
                target._update_alphas(alphas_dict)

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = AlphaLinear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = AlphaLinear4bit(
                adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = AlphaEmbedding(
                adapter_name, in_features, out_features, **embedding_kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(
                        target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = AlphaLinear(adapter_name, in_features,
                                out_features, bias=bias, **kwargs)

        return new_module

    def enable_adapter_layers(self):
        return NotImplemented

    def _get_active_adapter(self) -> str:
        return NotImplemented

    def disable_adapter_layers(self):
        return NotImplemented

    def set_adapter(self, adapter_name):
        return NotImplemented

    def merge_adapter(self):
        return NotImplemented

    def unmerge_adapter(self):
        return NotImplemented

    def delete_adapter(self, adapter_name):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        key_list = [key for key, _ in self.model.named_modules()
                    if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, AlphaLoraLayer):
                for attr in [
                    "r",
                    "lora_alpha",
                    "scaling",
                    "lora_A",
                    "lora_B",
                    "lora_embedding_A",
                    "lora_embedding_B",
                    "lora_dropout",
                ]:
                    if adapter_name in getattr(target, attr):
                        getattr(target, attr).pop(adapter_name)
                if target.active_adapter == adapter_name:
                    resetting_active_adapter = list(self.peft_config.keys())[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
                    )
                    target.active_adapter = resetting_active_adapter

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)


class AlphaLoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super(LoraLayer, self).__init__(self, in_features, out_features, **kwargs)

    def _update_alphas(self, new_alphas: dict):
        # dont update entire layer, only alphas
        self.lora_alpha = new_alphas
        adapters = list(self.lora_A.keys())
        for adapter_name in adapters:
            self.scaling[adapter_name] = self.lora_alpha[adapter_name] / self.r[adapter_name]


class AlphaLinear(Linear):
    # hydra-moe-alpha implemented in a linear layer.

    def _update_alphas(self, new_alphas: dict):
        # dont update entire layer, only alphas
        self.lora_alpha = new_alphas
        adapters = list(self.lora_A.keys())
        for adapter_name in adapters:
            self.scaling[adapter_name] = self.lora_alpha[adapter_name] / self.r[adapter_name]


    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        result = F.linear(x, transpose(
            self.weight, self.fan_in_fan_out), bias=self.bias)
        adapters = list(self.lora_A.keys())
        x = x.to(self.lora_A[adapters[0]].weight.dtype)
        for adapter_name in adapters:
            result += (
                self.lora_B[adapter_name](
                    self.lora_A[adapter_name](
                        self.lora_dropout[adapter_name](x))
                )
                * self.scaling[adapter_name]
            )
        result = result.to(previous_dtype)

        return result


class AlphaEmbedding(Embedding):
    # hydra-moe-alpha implemented in an embedding layer.

    def _update_alphas(self, new_alphas: dict):
        # dont update entire layer, only alphas
        self.lora_alpha = new_alphas
        adapters = list(self.lora_A.keys())
        for adapter_name in adapters:
            self.scaling[adapter_name] = self.lora_alpha[adapter_name] / self.r[adapter_name]


    def forward(self, x: torch.Tensor):
        result = nn.Embedding.forward(self, x)
        adapters = list(self.lora_A.keys())
        for adapter_name in adapters:
            after_A = F.embedding(
                x,
                self.lora_embedding_A[adapter_name].T,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @
                       self.lora_embedding_B[adapter_name].T) * self.scaling[adapter_name]

        return result


class AlphaLinear8bitLt(Linear8bitLt):
    # hydra-moe-alpha implemented in a dense layer.

    def __init__(self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs):
        super(Linear8bitLt, self).__init__(*kwargs[0])

    def _update_alphas(self, new_alphas: dict):
        # dont update entire layer, only alphas
        self.lora_alpha = new_alphas
        adapters = list(self.lora_A.keys())
        for adapter_name in adapters:
            self.scaling[adapter_name] = self.lora_alpha[adapter_name] / self.r[adapter_name]


    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters:
            return result
        
        else:
            adapters = list(self.lora_A.keys())

            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()

                for adapter_name in adapters:
                    result += self.lora_B[adapter_name](
                        self.lora_A[adapter_name](
                            self.lora_dropout[adapter_name](x))
                    ).to(expected_dtype) \
                    * self.scaling[adapter_name] 
            else:

                for adapter_name in adapters:
                    result += self.lora_B[adapter_name](
                            self.lora_A[adapter_name](
                                self.lora_dropout[adapter_name](x))
                    ) * self.scaling[adapter_name] 
        return result


class AlphaLinear4bit(Linear4bit):
    # hydra-moe-alpha implemented in a dense layer.

    def __init__(self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs):
        super(Linear4bit, self).__init__(*kwargs[0])

    def _update_alphas(self, new_alphas: dict):
        # dont update entire layer, only alphas
        self.lora_alpha = new_alphas
        adapters = list(self.lora_A.keys())
        for adapter_name in adapters:
            self.scaling[adapter_name] = self.lora_alpha[adapter_name] / self.r[adapter_name]

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters:
            return result
        
        else:
            result = result.clone()
            adapters = list(self.lora_A.keys())

            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                x = x.to(self.lora_A[adapters[0]].weight.dtype)

                for adapter_name in adapters:
                    result += self.lora_B[adapter_name](
                        self.lora_A[adapter_name](
                            self.lora_dropout[adapter_name](x))
                    ).to(expected_dtype) \
                    * self.scaling[adapter_name] 
            else:

                for adapter_name in adapters:
                    result += self.lora_B[adapter_name](
                            self.lora_A[adapter_name](
                                self.lora_dropout[adapter_name](x))
                    ) * self.scaling[adapter_name] 

        return result



class AlphaLinear4bit(bnb.nn.Linear4bit, LoraLayer):
# Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        bnb.nn.Linear4bit.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            compute_dtype=kwargs.get("compute_dtype", torch.float32),
            compress_statistics=kwargs.get(
                "compress_statistics", True),
            quant_type=kwargs.get("quant_type", "nf4"),
        )
        LoraLayer.__init__(
            self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.preprocessed = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha,
                            lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name


    def trim_task_vector(self, adapter_name, k, matrix):
        magnitude = matrix.abs()
        threshold = torch.kthvalue(magnitude.view(-1), int((1 - k / 100) * magnitude.numel())).values
        matrix[torch.abs(matrix) < threshold] = 0
        return matrix

    def elect_sign_vector(self, adapter_name, matrix):
        total_positive = (matrix[matrix > 0]).sum()
        total_negative = (matrix[matrix < 0]).abs().sum()
        matrix = torch.where(total_positive > total_negative, matrix.abs(), -matrix.abs())
        return matrix

    def disjoint_merge(self, adapter_name, matrix):
        elected_sign = torch.sign(matrix.sum())
        same_sign_weights = matrix[torch.sign(matrix) == elected_sign]
        mean = same_sign_weights.mean()
        matrix = torch.where(torch.sign(matrix) == elected_sign, mean, 0)
        return matrix
    def preprocess_weights(self, k):
        if not self.preprocessed:
            for adapter_name in self.lora_A.keys():
                self.lora_A[adapter_name].weight.data = self.trim_task_vector(adapter_name, k, self.lora_A[adapter_name].weight.data)
                self.lora_A[adapter_name].weight.data = self.elect_sign_vector(adapter_name, self.lora_A[adapter_name].weight.data)
                self.lora_A[adapter_name].weight.data = self.disjoint_merge(adapter_name, self.lora_A[adapter_name].weight.data)
                
                self.lora_B[adapter_name].weight.data = self.trim_task_vector(adapter_name, k, self.lora_B[adapter_name].weight.data)
                self.lora_B[adapter_name].weight.data = self.elect_sign_vector(adapter_name, self.lora_B[adapter_name].weight.data)
                self.lora_B[adapter_name].weight.data = self.disjoint_merge(adapter_name, self.lora_B[adapter_name].weight.data)
            
            self.preprocessed = True
    def forward(self, x: torch.Tensor):
        result = super().forward(x)
        # if not self.preprocessed:
        #     self.preprocess_weights(25)
        if not self.disable_adapters:
            adapters = list(self.lora_A.keys())

            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()

                for adapter_name in adapters:
                    if self.scaling[adapter_name] != 0:
                        
                        temp_result = self.lora_B[adapter_name](
                            self.lora_A[adapter_name](x)
                        ).to(expected_dtype) * self.scaling[adapter_name]
                        result = torch.add(result, temp_result)
            else:
                for adapter_name in adapters:
                    if self.scaling[adapter_name] != 0:
                        
                        temp_result = self.lora_B[adapter_name](
                            self.lora_A[adapter_name](x)
                        ).to(expected_dtype) * self.scaling[adapter_name]
                        result = torch.add(result, temp_result)

        return result
