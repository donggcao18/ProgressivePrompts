import unittest
import torch
from unittest.mock import MagicMock
from t5_dataset import T5Dataset  # replace with your actual file name

class TestT5Dataset(unittest.TestCase):
    def setUp(self):
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.return_tensors = 'pt'
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]])
        }
        self.dataset = T5Dataset(self.mock_tokenizer, 'CONCODE')

    def test_init(self):
        self.assertEqual(self.dataset.task, 'CONCODE')
        self.assertIn('CONCODE', self.dataset.task_list)

    def test_preprocess_function(self):
        example = {'nl': 'sort a list', 'code': 'list.sort()'}
        output = self.dataset.preprocess_function(example, task='CONCODE', max_length=10)
        self.assertIn('source_ids', output)
        self.assertIn('target_ids', output)
        self.assertEqual(output['source_ids'].shape[0], 4)
        self.assertEqual(output['target_ids'].shape[0], 4)

    def test_select_subset_ds(self):
        # Mock dataset with shape and select method
        mock_ds = MagicMock()
        mock_ds.shape = (100,)
        mock_ds.select = lambda idxs: idxs
        selected = self.dataset.select_subset_ds(mock_ds, k=10)
        self.assertEqual(len(selected), 10)

    def test_preprocess_raises_invalid_task(self):
        with self.assertRaises(ValueError):
            self.dataset.preprocess_function({'text': 'x'}, task='INVALID')

if __name__ == '__main__':
    unittest.main()
