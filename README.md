## Hydra-MoE: A new class of Open-Source Mixture of Experts

<p align="center">
🤗 <a href="https://huggingface.co/HydraLM" target="_blank">HF Repo</a> • 🐦 <a href="https://twitter.com/skunkworks_ai" target="_blank">Twitter</a> • ⚡ <a href="https://github.com/hydrallm" target="_blank">Github</a> • 👋 Join our <a href="discord.gg/bNKsa8gE2y" target="_blank">Discord</a> <br>
</p>

## Mission
A scalable, efficient and Open Source Mixture of Experts (MoE) architecture to enable OSS AI to achieve SOTA (GPT-4 level) performance.

## Description

Skunkworks OSS team introduces Hydra-MoE, a collection of innovative Mixture of Experts (MoE) architectures that utilize LoRA/QLoRA experts to scale and augment the performance of base language models. The central aim of this research is to transmute any base language model into an advanced, lightweight, efficient MoE framework, employing swappable QLoRA Expert Adapters, with the objective of achieving performance levels that rival state-of-the-art models. We use low-cost training approaches such as instruct finetuning and (low rank adaptations to retain the benefits of the MoE architecture while producing a model that can run on commodity hardware. 

## Project Status

This project has completed POC stage and is currently training, optimizing, evaluating and scaling several MoE architectures developed by our team.

## Goals

Our objectives for this project are:
- Enhance Llama-2 (or any base model) capabilities closer to GPT-4 level performance through MoE architecture
- Serve as an exhaustive open source attempt to validate MoE techniques for large language models 
- Release open source an effective MoE model and framework for scaling capabilities of LLMs

## Our Architectures

We have designed and developed several MoE architectures which have achieved promising early results.

1. Hydra-α: Swappable-QLoRA experts MoE
   - c-BTM Inspired
   - K-means clustering for unsupervised domain discovery from instruct dataset
   - QLoRA finetuned experts on discovered clusters
   - Dynamic swapping of QLoRA Experts at inference through similarity and classifier based methods
   - Weighted-average merging of experts with base model at inference
     
2. Hydra-β: End-2-End MoE-QLoRA
   - An Extension to Hydra-α with better Gating, merging and an E2E training method
   - Expert merging through TIES merging and other methods
   - End to End Training of gating/routing functions with exotic loss/reward functions

3. Hydra-γ: Switch-MoE-QLoRA
   - Switch Transformer Inspired
   - adapted to leverage a pre-trained base and LoRAs as finetuned experts


In our POCs, these architectures have displayed potential for scaling any base model. These architectures are currently being trained on further data and will be scaled to larger sizes. We have also discovered several insights that we aim to publish for the community.

## Initial Expert Domains

We are training experts for the following domains:

- Math 
- Science (Physics, Chemistry, Biology..)
- Reasoning
- Coding
- Writing
- Truthfulness
- Structured data handling
- Tool integration
- Role-Playing

## Datasets
Skunkworks community is collecting, consolidating and standardizing high quality public and private Instruct data sources to craft a unified source of data for our MoE initiative and any open-source project. The collective size of the datasets exceeds 90 separate sources covering all domains listed above.

We are collaborating with several academic and open source groups (including Alignment Lab and members of Nous) to expand datasets further. We are also crafting new instruct datasets from high quality raw data sources.

Full list of (public) datasets: https://docs.google.com/spreadsheets/d/1tkRZ8T7LZ0ojDUzNFWMKcrPiQJ8Glh-GIFfB7Bksb60/edit?usp=sharing

## Roadmap

Over the next weeks, we are exploring new architectures as well as optimizing, training and scaling existing performant architectures.
Our objective is to achieve SOTA performance within the next months by scaling a 70B base model with our most performant MoE arch. We fully open-source all datasets, code and models.

## Compute Requirements

Skunkworks team is currently seeking supporters/sponsors to provide us access to reliable continuous compute for the next 3-6 months. Our target compute objective is 64x GPUs (preferably H100s/A100s). 
This compute will help us scale our MoE experiments and the most performant MoE architecture to GPT-4 level performance.


## Collaborations

The Skunkworks community is comprised of hundreds of the top minds in OSS AI. The core team collaborates directly with individuals, open-source groups (Alignment Lab), academics and companies to accelerate the development of our data pipeline and MoE architectures.
We welcome open source contributions to HydraLM! Please join our Discord for details.


## Background on MoE
There is a tradeoff in dense language models between capabilities and inference time.

<p align="center" width="100%">
<a ><img src="imgs/triangle_of_success.png" alt="WizardLM" style="width: 50%; min-width: 50%; display: block; margin: auto;"></a>
</p>

To speed up inference, there is data or model parallelism. Alternatively, if we want to solve this problem without decreasing “efficiency”, i.e. how much GPU time you’re spending per inference, we would need a way to decouple inference FLOPs from model capabilities. Mixture of Experts are a class of sparse language models designed to perform this tradeoff at the cost of additional memory. In an MoE model, the linear layers of the model are replaced by a Gating Mechanism that routes to N experts.

<p align="center" width="100%">
<a ><img src="imgs/moe.png" alt="WizardLM" style="width: 50%; min-width: 50%; display: block; margin: auto;"></a>
</p>

The Gating network’s job is to compute probabilities for each expert, per token. Then, at each token, k=1, 2, or 3 experts are selected and used for the MoE layer, with the results concatenated in the end. In practice, the tokens are split up (almost) evenly between the experts, and processed in sequence during the forward pass. By increasing the number of experts, the amount of memory that the model uses goes up, but the cost of inference stays exactly the same, since we haven’t changed the amount of computation done by the model during each foward pass. MoE is an architecture that allows us to make the tradeoff we wanted to make. But do the results of MoE models stack up to dense transformers? Given that they still have more parameters in total, can we expect better performance from MoE mdoels with a similar number of inference FLOPs? In the original MoE paper from 2017, the authors use RNNs. But since RNNs are old, we will compare the results from the Switch Transformer paper from 2021, where the Google authors use the T5 architecture.

<p align="center" width="100%">
<a ><img src="imgs/graph.png" alt="WizardLM" style="width: 50%; min-width: 50%; display: block; margin: auto;"></a>
</p>

Based on the results, MoE models not only outperform dense models in terms of inference FLOPs, they also punch above their weight class: Switch-Base achieves lower perplexity on C4 than T5-Large while using 30% the FLOPs and 10x the parameters, while Switch-Large is competitive with T5-XXL with 6% the FLOPs and 2.5x the parameters.

There are several other MoE architectures that achieve varying results but consistently display the ability to scale LMs including Branch-Train-Merge (BTM), Clustered-BTM (c-BTM), GLaM... All papers are listed below in related works.

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


