import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets

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
                        'KodCode',
                        'RunBugRun',
                        'CoST',
                        'TheVault_Csharp']
        
        self.text_key = {'CONCODE': 'nl',
                           'CodeTrans': 'java',
                           'CodeSearchNet': 'code',
                           'BFP': 'buggy',
                           'KodCode': 'question',
                           'RunBugRun': 'buggy_code',
                           'CoST': 'code',
                           'TheVault_Csharp': 'code'}
        self.label_key = {'CONCODE': 'code',
                            'CodeTrans': 'cs',
                            'CodeSearchNet': 'docstring',
                            'BFP': 'fixed',
                            'KodCode': 'solution',
                            'RunBugRun': 'fixed_code',
                            'CoST': 'docstring',
                            'TheVault_Csharp': 'docstring'}
        self.task_instructions = {
                                'CONCODE': 'Generate Java code from the following English description: ',
                                'CodeTrans': 'Translate the following Java code into C#: ',
                                'CodeSearchNet': 'Summarize the following Ruby code into English: ',
                                'BFP': 'Refactor or improve the following Java code: ',
                                'KodCode': 'Generate Python code from the following description: ',
                                'RunBugRun': 'Refactor or improve the following Ruby code: ',
                                'CoST': 'Translate the following C++ code into C#: ',
                                'TheVault_Csharp': 'Summarize the following C# code into English: '}
        
        self.max_input_length = {'CodeTrans': 320,
                                 'CodeSearchNet': 256,
                                 'BFP': 130,
                                 'CONCODE': 320,
                                 'KodCode': 256,
                                 'RunBugRun': 256,
                                 'CoST': 320,
                                 'TheVault_Csharp': 256}

        self.train_only_tasks = {
            'KodCode': {'val': 5000, 'test': 5000},
            'RunBugRun': {'val': 972,  'test': 1000},
        }


    @staticmethod
    def _extract_first_paragraph(docstring):
        """Clean docstring: flatten lists, strip newlines, normalise whitespace."""
        if docstring is None:
            return ""
        if isinstance(docstring, (list, tuple)):
            s = " ".join(str(t) for t in docstring)
        else:
            s = str(docstring)
        s = s.replace("\n", " ")
        return " ".join(s.strip().split())

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

    # Function to preprocess raw input & label text into tokenized dictionary
    def preprocess_function(self, 
                            examples, 
                            task,
                            max_length=512,
                            max_input_length=None,
                            #batched=False
                            ):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}")
        tokenizer = self.tokenizer
        text_key = self.text_key[task]
        label_key = self.label_key[task]
        
        instruction = self.task_instructions[task]
        text = examples[text_key]
        
        text = instruction + \
               text + ' </s>' 
    
        # text = text + ' </s>' 

        src_max_length = max_input_length if max_input_length is not None else max_length
        source = tokenizer(text,
                            padding="max_length",
                            truncation=True,
                            max_length=src_max_length,
                            return_tensors="pt")

        target_text = examples[label_key]
        # Clean docstrings for summarisation tasks to remove noise
        if task in ('CodeSearchNet'):
            target_text = self._extract_first_paragraph(target_text)
        if not isinstance(target_text, str):
            target_text = str(target_text) if target_text is not None else ""
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
        elif task == 'CoST':
            dataset = load_dataset('dongg18/CoST', split=split)
        elif task == 'KodCode':
            dataset = load_dataset('KodCode/KodCode-V1-SFT-R1', split='train')
        elif task == 'RunBugRun':
            dataset = load_dataset('ASSERT-KTH/RunBugRun-Final', split='train')
            dataset = dataset.filter(lambda example: example["language"] == "ruby")
        elif task == 'TheVault_Csharp':
            _vault_split_map = {
                'train': ['train/small'],
                'validation': ['validation'],
                'test': ['test'],
            }
            dataset_dict = load_dataset(
                'Fsoft-AIC/the-vault-function',
                cache_dir='/data/theVault',
                languages=['c_sharp'],
                split_set=_vault_split_map[split],
            )
            dataset = concatenate_datasets(list(dataset_dict.values()))



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
                                                                            ),
                                                                            batched=False,
                                                                            )
        encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
        dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
        return dataloader
