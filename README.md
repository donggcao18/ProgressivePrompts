# Progressive Prompts

**Our work on Progressive Prompts is accepted to ICLR 2023!** 🎉

This repo includes an original implementation of Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis and Amjad Almahairi. ["Progressive Prompts: Continual Learning for Language Models"](https://arxiv.org/abs/2301.12314), ICLR 2023.



## :question: What's in this repository

This is our code structure:

```
|_T5_codebase/
      |_t5_dataset.py --> T5 Dataset class for reading and processing datasets
      |_t5_continual.py --> Model class for T5 with prompt tuning and continual learning functions
      |_train_t5_cl.py --> Code to run continual learning experiments with T5
      
|_BERT_codebase/
      |_dataset_utils.py --> BERT Dataset class for reading and processing datasets
      |_model_utils.py --> Model class for BERT with prompt tuning and fine-tuning functions
      |_continual_learning_utils.py --> Continual Learner class for Progressive Prompts (with BERT)
      |_continual_learning_one_head.py --> Continual Learner class for regularization-based CL approaches for BERT 
      |_train_cl2.py --> Code to run continual learning experiments with BERT
      
|_datasets/src/data/ --> CL datasets from Zhang et. al., 2015
      |_amazon --> Amazon reviews (zip archive, since dataset is not available through HuggingFace datasets)
      (the rest of datasets can be either accessed through HuggingFace or downloaded by instructions below)
```

**Note**: we access most of the datasets for our experiments through HuggingFace datasets, including CL datasets from Zhang et. al., 2015. Since only one CL datasets from Zhang et. al. is not available on HuggingFace - Amazon Reviews, we uploaded its archived train / test data to ```datasets/src/data/amazon/```. To access the rest of CL datasets (Yelp, Yahoo, AG, DbPedia), you can either use their HuggingFace names in our training script or download them from [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq) to ```datasets/src/data/```.

## :wrench: Installation


```sh
conda create -y -n nlp python=3.10.12
conda activate nlp 
cd ProgressivePrompts
pip install -r requirements.txt
```


## :zap: How to run 

For example, to run Progressive Prompts with T5-large on four tasks (IMDb, CB, SST-2 and DbPedia):
```bash
cd T5_codebase
bash run_t5.sh
```

In the example above, we froze weights and trained a prompt of size 10 (per task) for 10 epochs. We also limited data to 1000 samples per class. 
For other arguments and their descriptions, please check ```T5_codebase/train_t5_cl.py``` file.




## :raising_hand: Questions
If you have any questions about the paper or code, please contact Anastasia Razdaibiedina (anastasia.razdaibiedina[at]mail.utoronto.ca) or open an issue. 

## :books: Citation
If you use our code in your research, please cite our work:
```bibtex
@inproceedings{razdaibiedina2023progressive,
   title={Progressive Prompts: Continual Learning for Language Models},
   author={Razdaibiedina, Anastasia and Mao, Yuning and Hou, Rui and Khabsa, Madian and Lewis, Mike and Almahairi, Amjad},
   booktitle={International Conference on Learning Representations},
   year={2023}
}
```

<!--
@article{razdaibiedina2023progressive,
  title={Progressive Prompts: Continual Learning for Language Models},
  author={Razdaibiedina, Anastasia and Mao, Yuning and Hou, Rui and Khabsa, Madian and Lewis, Mike and Almahairi, Amjad},
  journal={arXiv preprint arXiv:2301.12314},
  year={2023}
}
-->
