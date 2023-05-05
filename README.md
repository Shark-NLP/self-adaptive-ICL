# Self-adaptive In-context Learning
This repository contains the source code for Self-adaptive In-context Learning, which is proposed in our paper [“Self-adaptive In-context Learning”](https://arxiv.org/abs/2212.10375). If you want to use our method easily, you can use [OpenICL](https://github.com/Shark-NLP/OpenICL), a toolkit for In-context learning. You can also quickly repeat our experiments using our [script](https://github.com/Shark-NLP/OpenICL/blob/main/examples/research_projects/self-adaptive_in-context_learning.ipynb) in it.

## Contents
* [Setup](#setup)
* [Reproduce](#reproduce)
* [Usage](#usage)
* [Modules](#modules)
* [Add a New Task](#add-a-new-task)
* [Citation](#citation)

## Setup
All required packages can be found in ``requirements.txt``. 
You can install them in a new environment with 
```shell
conda create -n adaptive python=3.8
conda activate adaptive

# The following line to be replaced depending on your cuda version.
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

accelerate config # ignore if you don't need multi-gpu
```

Setup WandB for tracking the training status in `scripts/run_xxx.sh`:
```shell
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_TEAM_NAME

root=YOUR_PROJECT_PATH
```

### Reproduce
```shell
bash ./scripts/run_mdl.sh
```

### Usage
Given an index dataset (by default the training set) and an test dataset (by default the test set), we include scripts to run five ICL method under `scripts/`:
- `run_mdl.sh`: based on mdl;
- `run_topk.sh`: based on the similarity of the sentence transfromer embedding;
- `run_random.sh`: random selected in-context examples;
- `run_local_e.sh`: based on entropy;
- `run_prompting.sh`: inference without in-context example;

The config files can be found in `configs/`.

## Modules
1. `prerank.py`: retrieve examples from training set with topk, random
2. `retriever.py`: continue to select and rank examples based on the result of prerank.py.
3. `ppl_inferencer.py`: inference based on the retrived in-context examples. 

## Add a New Task
Change the task by modify `task_name` argument, and the current available tasks are `sst5, mrpc, qnli, mnli, cmsqa, swag, webqs, geoquery, nl2bash, mtop, break, smcalflow`.
It's easy to add a new task with this repo. You can take the following steps:
1. Define a dataset wrapper under `src/dataset_readers/dataset_wrapper` to set the text fields.
2. Add a task template in `src/datasets/instructions.py`
3. Add a metric method in `src/metrics/eval_datasets.py`

## Citation
If you find our work helpful, please cite us:
```
@ARTICLE{2022arXiv221210375W,
       author = {{Wu}, Zhiyong and {Wang}, Yaoxiang and {Ye}, Jiacheng and {Kong}, Lingpeng},
        title = "{Self-adaptive In-context Learning}",
         year = 2022,
       eprint = {2212.10375},
 primaryClass = {cs.CL},
 archivePrefix={arXiv},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221210375W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
