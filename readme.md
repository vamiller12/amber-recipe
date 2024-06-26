<h1 align="center">Amber: a 7B English language model with the LLaMA architecture.</h1>

<div align="center">
   <img src="amber_logo.png" alt="amber logo" width="200"><br><br>
</div>

---

<p align="center">
   <a href="https://github.com/LLM360/Analysis360/blob/dev/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license"></a>
</p>

# Overview

All LLM360 models are trained and released to make access to LLM training knowledge accessible to all. This repo contains the complete training process and details we used to train Amber. 

## Reproduce Amber
The repo is organized into subfolders by function. 

To reproduce the entire training process, to proper order is:
1. Begin by [pretraining the model](./reproduce-amber) 
2. Determine the models performance through [evaluations and benchmarks](./evaluations)
3. Improve the base model with chat specific functionality via [finetuning](./finetuning)
4. Interact with model by [downloading Amber for inference](./inference)

## Repository Organization

Contains examples are organized in folders by topic:
| Subfolder | Description |
|---|---|
[reproduce amber](./reproduce-amber)|Instructions to fully reproduce Amber from data prep to trained model
[finetuning](./finetuning)|Scripts to finetune Amber for chat, SFT, and DPO alignment options
[inference](./inference)|Scripts to deploy Amber for inference locally
[evaluations and benchmarks](./evaluations)|Scripts to evaluation Amber and compare against LLM360's results

## About Amber
Amber is an 7B English language model with the LLaMA architecture.

## Training Details

| Hyperparameters      | Hyperparameter      | Value | Data Mix      | Subset      | Tokens (Billion) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|| Total Parameters      | 6.7B       || Arxiv      | 30.00       |
|| Hidden Size   | 4096        || Book   | 28.86        |
|| Intermediate Size (MLPs)   | 11008        || C4   | 197.67        |
|| Number of Attention Heads   | 32        || Refined-Web   | 665.01        |
|| Number of Hidden Layers  | 32        || StarCoder   | 291.92        |
|| RMSNorm ɛ  | 1e^-6        || StackExchange   | 21.75        |
|| Max Seq Length   | 2048        || Wikipedia   | 23.90        |
|| Vocab Size | 32000 || Total | 1259.13 |



## About LLM360
LLM360 is an initiative for comprehensive and fully open-sourced LLMs, 
where all training details, model checkpoints, intermediate results, and 
additional analyses are made available to the community. Our goal is to advance 
the field by inviting the community to deepen the understanding of LLMs 
together. As the first step of the project LLM360, we release all intermediate 
model checkpoints, our fully-prepared pre-training dataset, all source code and
configurations, and training details. We are
committed to continually pushing the boundaries of LLMs through this open-source 
effort.

Get access now at [LLM360 site](https://www.llm360.ai/)

# Citation

**BibTeX:**

```bibtex
@misc{liu2023llm360,
      title={LLM360: Towards Fully Transparent Open-Source LLMs}, 
      author={Zhengzhong Liu and Aurick Qiao and Willie Neiswanger and Hongyi Wang and Bowen Tan and Tianhua Tao and Junbo Li and Yuqi Wang and Suqi Sun and Omkar Pangarkar and Richard Fan and Yi Gu and Victor Miller and Yonghao Zhuang and Guowei He and Haonan Li and Fajri Koto and Liping Tang and Nikhil Ranjan and Zhiqiang Shen and Xuguang Ren and Roberto Iriondo and Cun Mu and Zhiting Hu and Mark Schulze and Preslav Nakov and Tim Baldwin and Eric P. Xing},
      year={2023},
      eprint={2312.06550},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
