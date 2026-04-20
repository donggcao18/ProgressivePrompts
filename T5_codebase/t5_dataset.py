from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from task_info import HF_SPLIT_MAP, TASK_LIST


class Seq2SeqDataset:
    """Minimal dataset helper: load from Hugging Face + collate."""

    max_input_length = {
        "CodeTrans": 320,
        "CodeSearchNet": 256,
        "BFP": 130,
        "CONCODE": 320,
        "TheVault_Csharp": 256,
        "KodCode": 512,
        "RunBugRun": 256,
        "CoST": 256,
    }
    max_target_length = {
        "CodeTrans": 256,
        "CodeSearchNet": 128,
        "BFP": 120,
        "CONCODE": 150,
        "TheVault_Csharp": 128,
        "KodCode": 300,
        "RunBugRun": 128,
        "CoST": 128,
    }

    def __init__(self, tokenizer, hf_dataset_id: str = "dongg18/CODETASK_with_instruction_pool"):
        self.tokenizer = tokenizer
        self.hf_dataset_id = hf_dataset_id
        self.task_list = list(TASK_LIST)

        self._active_task: Optional[str] = None
        self._active_input_max_len: int = 512
        self._active_target_max_len: int = 128

        self._data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

    @staticmethod
    def _canonical_split(split: str) -> str:
        return HF_SPLIT_MAP.get(split, split)

    @staticmethod
    def _candidate_splits(split: str) -> List[str]:
        canonical = HF_SPLIT_MAP.get(split, split)
        if canonical == "validation":
            return ["validation", "val", "dev"]
        if canonical == "test":
            return ["test", "eval"]
        return [canonical]

    @staticmethod
    def _unwrap_single_split_dataset(dataset, split_name):
        if isinstance(dataset, DatasetDict):
            if split_name in dataset:
                return dataset[split_name]
            if len(dataset) == 1:
                return next(iter(dataset.values()))
            available = list(dataset.keys())
            raise ValueError(
                f"Expected split '{split_name}' but got dataset dict with splits: {available}"
            )
        return dataset

    def _load_task_split(self, task: str, split: str):
        errors: List[str] = []
        for candidate in self._candidate_splits(split):
            try:
                dataset = load_dataset(self.hf_dataset_id, task, split=candidate)
                return self._unwrap_single_split_dataset(dataset, candidate), candidate
            except Exception as exc:
                errors.append(f"split={candidate}: {exc}")

        try:
            dataset_dict = load_dataset(self.hf_dataset_id, task)
            if isinstance(dataset_dict, DatasetDict):
                for candidate in self._candidate_splits(split):
                    if candidate in dataset_dict:
                        return dataset_dict[candidate], candidate
                available = list(dataset_dict.keys())
                raise ValueError(
                    f"Could not find split '{split}' for task '{task}' in '{self.hf_dataset_id}'. "
                    f"Available splits: {available}"
                )
            return dataset_dict, split
        except Exception as exc:
            details = "\n".join(errors[-3:])
            raise ValueError(
                f"Failed to load task '{task}' from '{self.hf_dataset_id}' for split '{split}'.\n"
                f"Recent attempts:\n{details}\n"
                f"Fallback error: {exc}"
            ) from exc

    def select_subset_ds(self, ds, k: int = 2000, seed: int = 0):
        np.random.seed(seed)
        total = len(ds)
        if total == 0:
            return ds
        num_samples = min(k, total)
        idx_total = np.random.choice(np.arange(total), num_samples, replace=False)
        return ds.select(idx_total)

    def _ensure_tokenized_feature(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        if "input_ids" in ex:
            feature = {
                "input_ids": ex["input_ids"],
                "attention_mask": ex.get("attention_mask"),
            }
            if "labels" in ex:
                feature["labels"] = ex["labels"]
            elif "target_ids" in ex:
                feature["labels"] = ex["target_ids"]
            return feature

        if "source_ids" in ex:
            feature = {
                "input_ids": ex["source_ids"],
                "attention_mask": ex.get("source_mask"),
            }
            if "labels" in ex:
                feature["labels"] = ex["labels"]
            elif "target_ids" in ex:
                feature["labels"] = ex["target_ids"]
            return feature

        if "input" in ex and "output" in ex:
            src = "" if ex["input"] is None else str(ex["input"])
            tgt = "" if ex["output"] is None else str(ex["output"])

            source = self.tokenizer(
                src,
                truncation=True,
                max_length=self._active_input_max_len,
                padding=False,
            )
            target = self.tokenizer(
                text_target=tgt,
                truncation=True,
                max_length=self._active_target_max_len,
                padding=False,
            )
            return {
                "input_ids": source["input_ids"],
                "attention_mask": source["attention_mask"],
                "labels": target["input_ids"],
            }

        raise ValueError(
            "Each dataset item must contain one of: "
            "(input_ids, labels) or (source_ids, target_ids) or (input, output)."
        )

    def _add_legacy_keys(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch["source_ids"] = batch["input_ids"]
        if "attention_mask" in batch:
            batch["source_mask"] = batch["attention_mask"]
        else:
            batch["source_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.long)

        if "labels" in batch:
            target_ids = batch["labels"].clone()
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            target_ids[target_ids == -100] = pad_id
            batch["target_ids"] = target_ids
            batch["target_mask"] = (target_ids != pad_id).long()

        return batch

    def _collate_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = [self._ensure_tokenized_feature(ex) for ex in examples]
        features = [
            {
                key: value
                for key, value in feat.items()
                if value is not None
            }
            for feat in features
        ]
        batch = self._data_collator(features)
        return self._add_legacy_keys(batch)

    def _build_loader(self, dataset, split_name: str, batch_size: int):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=self._collate_batch,
        )

    def get_final_ds(
        self,
        task,
        split,
        batch_size,
        k: int = -1,
        seed: int = 0,
        max_length: int = 512,
        max_target_length: int = 128,
        return_test: bool = False,
        **_unused_kwargs,
    ):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}. Available tasks: {self.task_list}")

        canonical_split = self._canonical_split(split)
        dataset, resolved_split = self._load_task_split(task, canonical_split)

        if k != -1:
            dataset = self.select_subset_ds(dataset, k=min(k, len(dataset)), seed=seed)
        else:
            dataset = dataset.shuffle(seed=seed)

        self._active_task = task
        self._active_input_max_len = self.max_input_length.get(task, max_length)
        self._active_target_max_len = self.max_target_length.get(task, max_target_length)

        if return_test:
            tmp = dataset.train_test_split(test_size=0.5, seed=seed)
            val_loader = self._build_loader(
                tmp["train"],
                split_name="validation",
                batch_size=batch_size,
            )
            test_loader = self._build_loader(
                tmp["test"],
                split_name="test",
                batch_size=batch_size,
            )
            return val_loader, test_loader

        return self._build_loader(
            dataset,
            split_name=resolved_split,
            batch_size=batch_size,
        )


class T5Dataset(Seq2SeqDataset):
    """Backward-compatible wrapper used by existing training scripts."""

    def __init__(
        self,
        tokenizer,
        task: Optional[str] = None,
        hf_dataset_id: str = "dongg18/CODETASK_with_instruction_pool",
    ):
        super().__init__(tokenizer=tokenizer, hf_dataset_id=hf_dataset_id)
        self.task = task

    def get_final_ds(self, task: Optional[str] = None, *args, **kwargs):
        task_name = task if task is not None else self.task
        if task_name is None:
            raise ValueError("Task must be provided either in constructor or get_final_ds(...).")
        return super().get_final_ds(task=task_name, *args, **kwargs)
