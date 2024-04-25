## Finetuning Methods
AmberChat is finetuned with [FastChat](https://github.com/lm-sys/FastChat), an open platform for training, serving, and evaluating large language model based chatbots developed by 
[LMSYS](https://lmsys.org/). We will present a brief walkthrough to replicate the finetuning process of AmberChat.
### Requirements
Follow the instructions [here](https://github.com/lm-sys/FastChat/tree/main?tab=readme-ov-file#method-2-from-source) to install FastChat. Make sure to install from source and include the extra requirements `[train]`:
```bash
pip3 install -e ".[model_worker,webui,train]"
```
Note that you first need to install [PyTorch](https://pytorch.org/get-started/locally/). Depending on your environment and setup, you may need to get a specific version for packages like `torch`, `accelerate`, `flash-attn`, `transformers`, etc.
### Data Mixtures
AmberChat is finetuned with the [evol-instruct](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k) dataset from WizardLM. To get a full dataset mixture, we also need to merge the original [ShareGPT-90k](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) dataset with this one. You can try the script [here](amberchat/prepare_evol_datamix.py) to create a combined datasets. It will store the combined dataset in a json file called `evol-mix.json` under your current directory.
### Chunking
The combined dataset may include long conversations that exceed Amber's maximum sequence length. Since Amber's context length is 2048, we need to split the original conversations in the dataset into chunks of 2048 tokens. We'll make use of a tool from FastChat to split our dataset:
```bash
python3 -m fastchat.data.split_long_conversation --in evol-mix.json --out evol-mix-split.json --model-name LLM360/Amber --max-length 2048
``` 
It will create a json file `evol-mix-split.json` under the current directory.
### Instruction Finetuning
Now we are ready to run the instruction finetuning job with the processed dataset that we prepared in the steps above. We follow the guidance provided from FastChat's [instructions](https://github.com/lm-sys/FastChat/tree/main?tab=readme-ov-file#fine-tuning) and refer to some example [scripts](https://github.com/lm-sys/FastChat/blob/main/scripts/train_vicuna_7b.sh) for vicuna finetuning. We run the finetuning tasks on a node with 8 × 80GB A100 GPUs distributed by FSDP. Flash attention is also enabled for better memory efficiency. The hyperparameters are listed in this [section](#hyperparameters).

You can try to adapt our [training script](amberchat/instruction_finetune.sh) to fit your own setup. To make it work on your environment, there are a few things that needs to be updated. For example, `DATA_PATH` should be the correct path of your input dataset. You'll also need to replace `TRAIN_SCRIPT_PATH` with the local path of the training script `FastChat/fastchat/train/train_mem.py` on your machine. If you would like to turn off the flash attention, try using `train.py` instead of `train_mem.py`. And if you want to disable FSDP, just remove any related options like `--fsdp`. Also remember to tune the hyperparameters like batch size to fit the GPU memory.

After the finetuning is done, you will find the checkpoints under `OUTPUT_DIR`, which can be loaded using `transformers`.
