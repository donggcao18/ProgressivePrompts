import os
import pandas as pd
import numpy as np
import hashlib

from typing import Dict

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets

from task_info import TASK_SPECS, HF_SPLIT_MAP, INSTRUCTION_POOL, TRAIN_ONLY_TASKS, TASK_LIST, INSTRUCTION_SPLIT_POLICY

"""
(check) find official dataset and their exact collumn name
""" 
class T5Dataset:
    def __init__(self, tokenizer, task):
        """
        Dataset class for T5 model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): T5 model tokenizer to use.
        """
        
        self.tokenizer = tokenizer
        self.task = task
        self.task_list = ['CodeTrans', 
                        'CodeSearchNet',
                        'BFP', 
                        'CONCODE',
                        'TheVault_Csharp',
                        'KodCode',
                        'RunBugRun',
                        'CoST']
        
        self.text_key = {'CONCODE': 'nl',
                           'CodeTrans': 'java',
                           'CodeSearchNet': 'code',   
                           'BFP': 'buggy',
                           'TheVault_Csharp': 'code',      
                           'KodCode': 'question',           
                           'RunBugRun': 'buggy_code',
                           'CoST': 'lang1'}               
        self.label_key = {'CONCODE': 'code',
                            'CodeTrans': 'cs',
                            'CodeSearchNet': 'docstring',
                            'BFP': 'fixed',
                            'TheVault_Csharp': 'docstring',    
                            'KodCode': 'solution',                
                            'RunBugRun': 'fixed_code',
                            'CoST': 'lang2'}              
        
        self.max_input_length = {'CodeTrans': 320,
                                 'CodeSearchNet': 256,
                                 'BFP': 130,
                                 'CONCODE': 320,
                                 'TheVault_Csharp': 256,        # TODO: update if needed
                                 'KodCode': 256,                 # TODO: update if needed
                                 'RunBugRun': 256,
                                 'CoST': 256}               # TODO: update if needed

        self.train_only_tasks = {
            'KodCode': {'val': 5000, 'test': 5000},
            'RunBugRun': {'val': 972,  'test': 1000},
        }


    def _split_train_only(self, dataset, task, split, split_seed=42):

        sizes = self.train_only_tasks[task]
        test_size  = sizes['test']
        val_size   = sizes['val']

        # Step 1: carve out test from the full dataset
        tmp = dataset.train_test_split(test_size=test_size, seed=split_seed)
        test_ds = tmp['test']

        # Step 2: carve out val from the remainder (test never included)
        tmp2 = tmp['train'].train_test_split(test_size=val_size, seed=split_seed)
        train_ds = tmp2['train']
        val_ds   = tmp2['test']

        mapping = {'train': train_ds, 'validation': val_ds, 'test': test_ds}
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}' for train-only task '{task}'")
        return mapping[split]


    def select_subset_ds(self, ds, k=2000, seed=0):
        np.random.seed(seed)
        num_samples = min(k, ds.shape[0])
        idx_total = np.random.choice(np.arange(ds.shape[0]), num_samples, replace=False)
        return ds.select(idx_total)

    @staticmethod
    def _to_string(value):
        if value is None:
            return ""
        return str(value)

    def _get_candidate_instruction_pool(self, task, split_name):
        task_type = TASK_SPECS[task]['task_type']
        pool = INSTRUCTION_POOL.get(task_type, [])
        if not pool:
            raise ValueError(f"No instruction templates defined for task_type '{task_type}'")

        policy = INSTRUCTION_SPLIT_POLICY.get(split_name, INSTRUCTION_SPLIT_POLICY['train'])
        if policy['pool_scope'] == 'full':
            return pool

        if policy['pool_scope'] == 'head_fraction':
            fraction = float(policy.get('fraction', 0.75))
            if fraction <= 0:
                raise ValueError(f"Invalid fraction {fraction} for split '{split_name}'")
            head_size = max(1, int(len(pool) * fraction))
            return pool[:head_size]

        raise ValueError(f"Unknown pool_scope '{policy['pool_scope']}' for split '{split_name}'")

    def _select_instruction_template(self, task, sample_key, split_name, split_seed):
        candidate_pool = self._get_candidate_instruction_pool(task, split_name)
        random_key = f"{split_seed}::{split_name}::{sample_key}"
        idx = int(hashlib.md5(random_key.encode("utf-8")).hexdigest(), 16) % len(candidate_pool)
        return candidate_pool[idx]

    def _render_instruction(self, task, raw_input, sample_key, split_name, split_seed):
        spec = TASK_SPECS[task]
        template = self._select_instruction_template(task, sample_key, split_name, split_seed)

        format_values: Dict[str, str] = {
            'language': spec.get('language', 'code'),
            'description': raw_input,
            'code': raw_input,
            'source_lang': spec.get('source_lang', spec.get('language', 'source language')),
            'target_lang': spec.get('target_lang', 'target language'),
        }
        return template.format(**format_values)

    # Function to preprocess raw input & label text into tokenized dictionary
    def preprocess_function(self, 
                            examples, 
                            task,
                            max_length=512,
                            max_input_length=None,
                            split_name='train',
                            split_seed=42,
                            #batched=False
                            ):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}")
        tokenizer = self.tokenizer
        text_key = self.text_key[task]
        label_key = self.label_key[task]

        raw_input = self._to_string(examples[text_key])
        sample_uid = hashlib.md5((task + "||" + raw_input).encode("utf-8")).hexdigest()
        instruction = self._render_instruction(
            task=task,
            raw_input=raw_input,
            sample_key=f"{task}::{sample_uid}",
            split_name=split_name,
            split_seed=split_seed,
        )

        text = instruction + ' </s>' 
    
        # text = text + ' </s>' 

        src_max_length = max_input_length if max_input_length is not None else max_length
        source = tokenizer(text,
                            padding="max_length",
                            truncation=True,
                            max_length=src_max_length,
                            return_tensors="pt")

        target_text = self._to_string(examples[label_key])
        target_text += ' </s>'  
        target = tokenizer(target_text,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt")

        dict_final = {
            "source_ids": source["input_ids"].squeeze(0),
            "source_mask": source["attention_mask"].squeeze(0),
            "target_ids": target["input_ids"].squeeze(0),
            "target_mask": target["attention_mask"].squeeze(0),
        }
        return dict_final

    def get_final_ds(self, 
                     task, 
                     split,
                     batch_size,
                     k=-1,
                     seed=0,
                     max_length=512):
        """Function that returns final T5 dataloader.
        Args:
            task (str): Name of the downstream task.
            split (str): Which data split to use (train/validation/test).
            batch_size (int): Batch size to use in the dataloader.
            k (int, optional): Number of samples to use for each class. Defaults to -1, not sub-sample the data.
            seed (int, optional): Seed used for random shuffle. Defaults to 0.
            target_len (int, optional): Length of the model output (in tokens). Defaults to 2.
            max_length (int, optional): Length of the model input (in tokens). Defaults to 512.

            
        Returns:
            Dataloader: Torch Dataloader with preprocessed input text & label.
        """

        if task == 'CONCODE':
            dataset = load_dataset('AhmedSSoliman/CodeXGLUE-CONCODE', split=split)
        elif task == 'CodeTrans':
            dataset = load_dataset('CM/codexglue_codetrans', split=split)
        elif task == 'CodeSearchNet':
            dataset = load_dataset('semeru/code-text-ruby', split=split)
        elif task == 'BFP':
            dataset = load_dataset('ayeshgk/code_x_glue_cc_code_refinement_annotated', split=split)
        elif task == 'TheVault_Csharp':
            if split == 'train':
                dataset = load_dataset("Fsoft-AIC/the-vault-function", cache_dir="/data/theVault", languages=["c_sharp"], split_set="train/small")
            else:    
                dataset = load_dataset("Fsoft-AIC/the-vault-function", cache_dir="/data/theVault", languages=["c_sharp"], split_set=split)
        elif task == 'KodCode':
            dataset = load_dataset('KodCode/KodCode-V1-SFT-R1', split='train')
        elif task == 'RunBugRun':
            dataset = load_dataset('ASSERT-KTH/RunBugRun-Final', split='train')
            dataset = dataset.filter(lambda example: example["language"] == "ruby")
        elif task == 'CoST':
            dataset = load_dataset('dongg18/CoST', split=split)



        # For train-only tasks, manually split into train / val / test
        # split_seed is fixed (42) and independent of the shuffle seed
        # to guarantee consistent, non-overlapping partitions.
        if task in self.train_only_tasks:
            dataset = self._split_train_only(dataset, task, split, split_seed=42)

        
        # Selecting k subset of the samples (if requested)
        if k!=-1:
            dataset = self.select_subset_ds(dataset, k=k)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        # Returning the selected data split (train/val/test)
        task_max_input_length = self.max_input_length.get(task, max_length)
        encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, 
                                                                            task,
                                                                            max_length=max_length,
                                                                            max_input_length=task_max_input_length,
                                                                            split_name=split,
                                                                            split_seed=seed,
                                                                            ),
                                                                            batched=False,
                                                                            )
        encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
        dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
        return dataloader
