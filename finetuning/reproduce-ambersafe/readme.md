## Finetuning Methods
[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO) is a stable, efficient, and computationally lightweight approach for LLM alighment finetuning. It eliminates the need for reward model fitting, extensive sampling, and hyperparameter tuning.

We make use of the [DPO github repo](https://github.com/eric-mitchell/direct-preference-optimization) which includes a reference implementation of the DPO algorithm for training language models from preference data. You can clone the repo with the following script:
```bash
git clone https://github.com/eric-mitchell/direct-preference-optimization.git
```

### Data Processing
[PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) is a preference dataset consisting of 30k+ expert comparison data. For each question in the dataset, it gives two responses with annoteted information regarding their corresponding helpfulness and harmlessness. Since we focus on safety, we will need to filter out the rows with the same halmlessness annotation. For example, if both responses for a question are considered not safe, then we will simply discard this data point.

We follow the instruction provided [here](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#adding-new-datasets) to integrate the PKU-SafeRLHF dataset into the DPO repository. In general, we need to convert the orignal dataset into a specified format that can be used for DPO finetuning.

To start with, let's add a new function into [preference_datasets.py](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py) to load, filter, and transform the PKU-SafeRLHF dataset. Feel free to reuse our script [here](ambersafe/get_saferlhf.py). You may just copy the code of `get_saferlhf` function into `preference_datasets.py`.

The next step is to register the function so that we can load our custom preference dataset from the command line tool. All it takes is to add a new `elif` branch before this [line](https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/preference_datasets.py#L171). You can refer to our implementation below:
```python
...
elif name == "saferlhf":
    data = get_saferlhf(split, silent=silent, cache_dir=cache_dir)
...
```

### SFT
Now that we are ready with the dataset, we can proceed to the training part. DPO usually consists of two stages, supervised fine-tuning (SFT) and direct preference optimization (DPO). The SFT stage essentially ensures that the preference data we train on is in-distribution for our policy before we actually do the learning from preferences part. For each response pair in the dataset, we will only take the safe one for SFT.

You may follow the steps in the [instructions](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#running-sft) and [examples](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#step-2-run-sft) to get an overview of how to run the SFT and how to adapt different options for your setup. It provides a brief guidance on loading custom model and dataset and setting up FSDP. We also provide a script [here](ambersafe/sft.sh) for reference. Make sure to replace `TRAIN_SCRIPT_PATH` with the local path of `direct-preference-optimization/train.py` on your machine. You may also find it helpful to tune parameters like batch size, learning rate, and number of epochs, etc., to get better performance on your environment.

After the training is done, the checkpoint will be saved to `.cache` under your current working directory. The script usually generates a series of `policy.pt` files that can be loaded in PyTorch.

### DPO
Now that we get the SFT checkpoint, we are ready to make use our preference dataset for DPO training. You can learn how to run DPO training by refering to materials listed in the github repository. Some useful resources include [Running DPO](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#running-dpo) and a quick walkthrough [here](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#step-3-run-dpo). 

Most arguments of the DPO command are similar to that of the SFT script so you can tune the options the same way as before. There are two major differences compared with the SFT command. The first one is the loss. You will need to set the `loss` to `dpo` and also choose a reasonable value (e.g., `0.1`) of hyperparameter `loss.beta`. The second difference is that you need to load the SFT checkpoint using `model.archive`.

Again, we do provide a reference [script](ambersafe/dpo.sh) for DPO training. Feel free to reuse our script and change any arguments (e.g., `TRAIN_SCRIPT_PATH` for the local path of `direct-preference-optimization/train.py`, and `SFT_CKPT_PATH` for the path of the SFT checkpoint `policy.pt`) as you need.

Make sure to monitor the `reward/accuracies` and `reward/margins` as the training goes to ensure that the DPO is stably progressing.

You can look for the checkpoint under `.cache` after the DPO training is done.
