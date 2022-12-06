# Progressive Prompts

This repo includes an original implementation of Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis and Amjad Almahairi. ["Progressive Prompts: Continual Learning for Language Models without Forgetting"](https://openreview.net/pdf?id=UJTgQBc91_), 2022.

### Table of contents
* [Introduction](#star2-introduction)
* [What's in this repository](#question-whats-in-this-repository)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 


## :star2: Introduction
We introduce **Progressive Prompts** – a novel Continual Learning (CL) approach for language models. Our
method is inspired by progressive networks ([A. Rusu et al., NeurIPS 2017](https://arxiv.org/pdf/1606.04671.pdf)), but is significantly more memory-efficient. In Progressive Prompts, we learn a separate set of virtual tokens, or ***prompt*** ([B. Lester et al., EMNLP 2021](https://arxiv.org/pdf/2104.08691.pdf)), for each incoming task and sequentially concatenate it with previously learned prompts. 

Our method can: 

1) **alleviate catastrophic forgetting**; since it preserves the knowledge acquired by previous prompts, and 
2) **transfer knowledge to future tasks**; since new prompts are sequentially concatenated with all prior prompts.

![Progressive Prompts schematics](/images/illustration.png)
Figure: *Illustrating our proposed method **Progressive Prompts** and contrasting it with a simple
adaptation of progressive networks using prompt tuning. In the simple adaptation of progressive
networks we learn a separate prompt and repeat the frozen input embeddings for each new task.
This setup requires repeating input tokens for each task. In Progressive Prompts we use the same
input and progressively append new prompt for each new task. Prior task prompts are not modified
by the addition of new prompts.*

## :question: What's in this repository

This is our code structure:

```
|_T5_codebase/
      |_t5_dataset.py --> T5 Dataset class for reading and processing datasets
      |_train_t5_cl.py --> Model class for T5 with prompt tuning and continual learning functions
      |_t5_continual.py --> Code to run continual learning experiments with T5
      
|_BERT_codebase/
      |_dataset_utils.py --> BERT Dataset class for reading and processing datasets
      |_model_utils.py --> Model class for BERT with prompt tuning and fine-tuning functions
      |_continual_learning_utils.py --> Continual Learner class for Progressive Prompts (with BERT)
      |_continual_learning_one_head.py --> Continual Learner class for regularization-based CL approaches for BERT 
      |_train_cl2.py --> Code to run continual learning experiments with BERT
```


## :wrench: Installation

Our implementation is based on PyTorch and HuggingFace (transformers + datasets). 

Requirements:
* Python 3.8.5
* Pytorch 1.10.0
* transformers 4.20.0
* datasets 2.3.2
* tqdm, sklearn, numpy, pandas

Step-by-step instructions to get you running Progressive Prompts:

### 1) Clone this repository to your local machine:

```console
git clone https://github.com/arazd/ProgressivePrompts    
```  

A folder called ```ProgressivePrompts``` with all the codebase should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://docs.conda.io/en/latest/miniconda.html).

To run Progressive Prompts code on GPU, make sure that you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. In our implementation, we used and CUDA 11.0.

You can re-create our conda enviroment from ```environment.yaml``` file:

```console
cd ProgressivePrompts
conda env create -f environment.yaml
```

Your conda should start downloading and extracting packages. This can take ~15-20 minutes.

### 3) Activate the environment:

Your environment should be called ```nlp```, and you can activate now it to run the scripts:

```console
conda activate nlp
```

## :zap: How to run 

```bash
python train_t5_cl.py --task_list cb --prefix_len 10 --save_name cb_k1000_10epochs_progTEST --select_k_per_class 1000 --num_epochs 10
```
<!--
The configuration keys are as follows:
| Argument |   Default     |  Description |
|----------|:-------------:|------:   |
| col 1 is |  left-aligned | $1600    |
| col 2 is |    centered   |   $12.   |
| col 3 is | right-aligned |    $1    |
-->

**Note**: if you have any questions about the paper or code, please contact Anastasia Razdaibiedina (anastasia.razdaibiedina[at]mail.utoronto.ca) or open an issue. 
