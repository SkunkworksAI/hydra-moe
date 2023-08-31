## Hydra-MoE: Open-Source Mixture of Experts

<p align="center">
🤗 <a href="https://huggingface.co/HydraLM" target="_blank">HF Repo</a> • 🐦 <a href="https://twitter.com/skunkworks_ai" target="_blank">Twitter</a> • 👋 Join our <a href="discord.gg/bNKsa8gE2y" target="_blank">Discord</a> <br>
</p>

## Mission
Develop an effective Open Source Mixture of Experts (MoE) architecture to enable OSS AI to achieve SOTA (GPT-4 level) performance.

## Current Status
Several MoE architectures developed. Currently training, optimizing and evaluating. We are also exploring more MoE arch designs based on insights gathered from our collective experiments.

## Description

Skunkworks OSS team introduces HydraLM, a collection of innovative Mixture of Experts (MoE) architectures that utilize LoRA/QLoRA experts to scale and augment the performance of base language models. The central aim of this research is to transmute any base language model into an advanced,lightweight, efficient MoE framework, employing swappable QLoRA Expert Adapters, with the objective of reaching performance levels that rival state-of-the-art models. The use of techniques such as Low Rank Adaptation (LoRAs) and 4-bit quantized finetuning (QLoRA) has allowed for the finetuning of open source large language models such as LLaMA and Falcon on domain-specific instruction following datasets for a fraction of the cost of a full finetune with minimal/no degradation. LoRAs are leveraged by current SOTA open-source models. We leverage LoRAs/QLoRAs to build highly efficient and scalable MoE architectures.

## Architectures

As part of the initiative, the team is exploring and has developed several architecture designs to identify the most performant MoE architecture. Based on our exploration, we have developed several MoEs that have achieved promising early results.

1. Swappable-QLoRA experts MoE (c-BTM variant)
   - Classifier-based dynamic swapping QLoRA Experts trained on unsupervised domain clusters
3. E2E Gating Swappable-QLoRA experts MoE (c-BTM variant)
   - Same as above, but with more advanced Gating/Routing + Adapter Merging + End to End Training
4. Switch-like MoE-QLoRAs (Switch Transformer variant)
   - Switch Transformer adapted to leverage a pre-trained base and LoRAs as finetuned experts
5. Switch-Llama (Switch Transformer variant)
   - Vanilla Switch Transformer w/ pretrained Llama2 base

In our POCs, these architectures have displayed potential for scaling any base model. These architectures are currently being trained on further data and will be scaled to larger sizes. We have also discovered several insights that we aim to publish for the community.

## Datasets
Skunkworks community is collectively standardizing every high quality public and private Instruct data source to craft a unified source of data for our MoE initiative and any open-source project. The collective size of the datasets exceeds 90 separate sources and over 1T tokens.
We are collaborating with several academic and open source groups to expand the datasets further.
We are also crafting new instruct datasets from high quality raw data sources.

Full list of (public) datasets: https://docs.google.com/spreadsheets/d/1tkRZ8T7LZ0ojDUzNFWMKcrPiQJ8Glh-GIFfB7Bksb60/edit?usp=sharing
Standardized datasets are hosted here: https://huggingface.co/HydraLM

## Compute Requirements

Our MoE initiative is bottlenecked by compute. We are currently seeking supporters to provide us access to reliable continuous compute for the next 3-6 months. Our target compute objective is 64x H100s. 


## Some related works
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- DEMix Layers: Disentangling Domains for Modular Language Modeling
- Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models
- Scaling Expert Language Models with Unsupervised Domain Discovery
- Soft Merging of Experts with Adaptive Routing
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- AdapterFusion: Non-Destructive Task Composition for Transfer Learning
- Nearest Neighbor Zero-Shot Inference 
- Eliciting and Understanding Cross-Task Skills with Task-Level Mixture-of-Experts
- Mixture-of-Supernets: Improving Weight-Sharing Supernet Training with Architecture-Routed Mixture-of-Experts
- Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints
- AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning


