import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class T5Dataset:
    def __init__(self, tokenizer, task):
        """Dataset class for T5 model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): T5 model tokenizer to use.
        """
        
        self.tokenizer = tokenizer
        self.TaskCode_benchmark = ['CONCODE', 'CodeTrans', 'CodeSearchNet', 'BFP']
        self.task = task
        self.label_key = 'answer'

    
    # Helper function to save idx of multirc questions (needed later for test metric computation)
    def save_multirc_questions_idx(self, val_ds):
        idx = []
        i = 0
        x_prev, y_prev= val_ds['paragraph'][0], val_ds['question'][0]

        for x,y in zip(val_ds['paragraph'], val_ds['question']):
            if x_prev!=x or y_prev!=y:
                i += 1
            x_prev = x
            y_prev = y
            idx.append(i)
        self.multirc_idx = np.array(idx)

    
    # Helper function to select a subset of k samples per class in a dataset
    def select_subset_ds(self, ds, k=2000, seed=0):
        if self.task in ['stsb', 'record', 'wsc']: # non-discrete labels
            idx_total = np.random.choice(np.arange(ds.shape[0]), min(k,ds.shape[0]), replace=False)

        else:
            label_key = self.label_key
            N = len(ds[label_key])
            idx_total = np.array([], dtype='int64')

            for l in set(ds[label_key]):
                idx = np.where(np.array(ds[label_key]) == l)[0]
                idx_total = np.concatenate([idx_total, # we cannot take more samples than there are available
                                            np.random.choice(idx, min(k, idx.shape[0]), replace=False)])

        np.random.seed(seed)
        np.random.shuffle(idx_total)
        return ds.select(idx_total)

    
    # WSC task function to preprocess raw input & label text into tokenized dictionary
    def process_wsc(self, wsc_row):
        text_proc = wsc_row['text'].split(' ')
        #text_proc[wsc_row['span1_index']] = '*' + text_proc[wsc_row['span1_index']] +'*'
        target = text_proc[wsc_row['span1_index']]
        text_proc[wsc_row['span2_index']] = '*' + text_proc[wsc_row['span2_index']] + '*'
        text_proc = (' ').join(text_proc)
        return text_proc, target

    
    # Function to preprocess raw input & label text into tokenized dictionary
    def preprocess_function(self, 
                            examples, 
                            max_length=512):
        tokenizer = self.tokenizer

        text = examples['text']
        text += ' </s>' 
        source = tokenizer(text,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt")

        target_text = examples['target']
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
            dataset = load_dataset('irds/codesearchnet', split=split)
        elif task == 'BFP':
            dataset = load_dataset('shaznin/task4_dataset_bfp', split=split)

        
        # Using Lester et al. setting for WSC task, e.g.
        # using only positive samples (for output generation)
        if self.task == 'wsc': 
            idx = np.where(np.array(dataset['label']) == 1)[0]
            dataset = dataset.select(idx)
        
        # Selecting k subset of the samples (if requested)
        if k!=-1:
            dataset = self.select_subset_ds(dataset, k=k)

        if k==-1 and split!='train' and self.task=='multirc':
            # we do not shuffle full validation set of multirc
            # but we save idx of the same questions
            # which are used for multirc test metric computation
            self.save_multirc_questions_idx(dataset)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        # Returning the selected data split (train/val/test)
        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                            max_length=max_length,
                                                                            prefix_list=prefix_list),
                                                                            batched=False)
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
                encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                                 max_length=max_length,
                                                                                 max_length_target=target_len,
                                                                                 prefix_list=prefix_list),
                                              batched=False)
                encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                                  'target_ids', 'target_mask'])
                dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
                dataloaders_val_test.append(dataloader)

            return dataloaders_val_test
