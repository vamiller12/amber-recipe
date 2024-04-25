
<h1 align="center">Finetuning Amber</h1>

---

<p align="center">
• 🤗 <a href="https://huggingface.co/LLM360/AmberChat">[AmberChat Download]</a>
• 🤗 <a href="https://huggingface.co/LLM360/AmberSafe">[AmberSafe Download]</a> 
</p>

# Overview
Extending a base model's capability to use case requires is often vital for adoption. This repo contains instructions for instruction tuning Amber to AmberChat. AmberChat can be further tuned to make output safer using the instructions in AmberSafe.

## How to use this repo
Start here:
1. Follow the instructions in [AmberChat](./reproduce-amberchat) to transform Amber into AmberChat
2. Once AmberChat is complete, use DPO to improve the AmberChat's output following the instructions to train [AmberSafe](./reproduce-ambersafe)

# About AmberChat
[AmberChat](https://huggingface.co/LLM360/AmberChat) is an instruction following model finetuned from [LLM360/Amber](https://huggingface.co/LLM360/Amber).

## Model Description
- **Model type:** Language model with the same architecture as LLaMA-7B
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Resources for more information:**
  - [Metrics](https://github.com/LLM360/Analysis360)
  - [Fully processed Amber pretraining data](https://huggingface.co/datasets/LLM360/AmberDatasets)

## DataMix
| Subset      | Number of rows |  License   |
| ----------- | ----------- | ----------- |
| WizardLM/WizardLM_evol_instruct_V2_196k      | 143k       |  |
| anon8231489123/ShareGPT_Vicuna_unfiltered   | 90k        | cc0-1.0 |
| Total | 233k |  |

## Hyperparameters
| Training Hyperparameters      | Value | | Model Hyperparameters      | Value |
| ----------- | ----------- | ----------- |----------- | ----------- |
| learning_rate      | 2e-5       || Total Parameters      | 6.7B       |
| num_train_epochs  |  3        || Hidden Size   | 4096        |
| per_device_train_batch_size   | 2        || Intermediate Size (MLPs)   | 11008        |
| gradient_accumulation_steps  | 16        || Number of Attention Heads   | 32        |
| warmup_ratio | 0.04      || Number of Hidden Lyaers  | 32        |
| model_max_length | 2048     || RMSNorm ɛ  | 1e^-6        |
|  |      || Max Seq Length   | 2048        |
|  |      || Vocab Size | 32000 |




# About AmberSafe
[AmberSafe](https://huggingface.co/LLM360/AmberSafe) is a safety-finetuned instruction model using [LLM360/AmberChat](https://huggingface.co/LLM360/AmberChat) as the base. It's an aligned version of the model finetuned with Direct Preference Optimization (DPO).

## Model Description
- **Model type:** Language model with the same architecture as LLaMA-7B
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Resources for more information:**
  - [Metrics](https://github.com/LLM360/Analysis360)
  - [Fully processed Amber pretraining data](https://huggingface.co/datasets/LLM360/AmberDatasets)

## DataMix
| Subset      | Number of rows |  License   |
| ----------- | ----------- | ----------- |
| [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)    | 330k        | cc-by-nc-4.0 |
| Total | 330k |  |



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
