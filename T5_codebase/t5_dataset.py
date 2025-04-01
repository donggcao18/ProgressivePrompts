import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

"""
(uncheck) find official dataset and their exact collumn name
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
        self.task_list = ['CodeTrans', 'CodeSearchNet', 'BFP', 'CONCODE']
        self.task = task
        self.text_key = {'CONCODE': 'nl',
                           'CodeTrans': 'java',
                           'CodeSearchNet': 'code',   
                           'BFP': 'buggy'}
        self.label_key = {'CONCODE': 'code',
                            'CodeTrans': 'cs',
                            'CodeSearchNet': 'docstring',
                            'BFP': 'fixed'}
        self.task_instructions ={ 'CONCODE': 'Generate Java code from the following English description:\n',
                                'CodeTrans': 'Translate the following Java code into C#:\n',
                                'CodeSearchNet': 'Summarize the following Ruby code into English:\n',
                                'BFP': 'Refactor or improve the following Java code:\n'}

    """
    For code generation tasks: randomly select k examples from the dataset.
    """
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
        
        source = tokenizer(text,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt")

        target_text = examples[label_key]
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
                     return_test=False,
                     max_length=512,
                     prefix_list=[]):
        """Function that returns final T5 dataloader.
        Args:
            task (str): Name of the downstream task.
            split (str): Which data split to use (train/validation/test).
            batch_size (int): Batch size to use in the dataloader.
            k (int, optional): Number of samples to use for each class. Defaults to -1, not sub-sample the data.
            seed (int, optional): Seed used for random shuffle. Defaults to 0.
            return_test (bool, optional): Whether to create a test split. 
                When True, two Dataloaders are returned. Defaults to False.
            target_len (int, optional): Length of the model output (in tokens). Defaults to 2.
            max_length (int, optional): Length of the model input (in tokens). Defaults to 512.
            prefix_list (List[str], optional): List of prompt virtual tokens to pre-pend to the input. 
                We do not encode soft prompt as extra virtual tokens in the latest implementation.
                Defaults to [], empty list.
            
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

        
        # Selecting k subset of the samples (if requested)
        if k!=-1:
            dataset = self.select_subset_ds(dataset, k=k)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        # Returning the selected data split (train/val/test)
        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, 
                                                                            task,
                                                                            max_length=max_length,
                                                                            #prefix_list=prefix_list
                                                                            ),
                                                                            batched=False
                                                                            )
            encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
            dataloader = DataLoader(encoded_dataset, batch_size=batch_size)

            return dataloader
        
        # Creating an extra test set from the selected data split
        else:
            N = len(dataset)
            dataset_val = dataset.select(np.arange(0, N//2))
            dataset_test = dataset.select(np.arange(N//2, N))

            dataloaders_val_test = []
            for dataset in [dataset_val, dataset_test]:
                encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, 
                                                                                 task,
                                                                                 max_length=max_length,
                                                                                 #prefix_list=prefix_list
                                                                                 ),
                                                                                 batched=False)
                encoded_dataset.set_format(type='torch', columns=['source_ids', 
                                                                  'source_mask',
                                                                  'target_ids', 
                                                                  'target_mask'])
                dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
                dataloaders_val_test.append(dataloader)

            return dataloaders_val_test
