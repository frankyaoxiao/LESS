"""
DPO validation dataset loading.

This module provides functions for loading DPO validation data.
The format is identical to training data (preference pairs with chosen/rejected responses).

For validation gradient collection, we typically use SGD gradients (no Adam state needed).
"""

from typing import List, Optional, Union

from datasets import Dataset
from torch.utils.data import DataLoader

from less.data_selection.get_dpo_dataset import (
    DPODataCollator,
    encode_dpo_data,
    get_dpo_dataloader,
    load_dpo_raw_dataset,
)


def get_dpo_validation_dataset(
    validation_files: Union[List[str], str],
    tokenizer,
    max_seq_length: int,
    sample_percentage: float = 1.0,
    seed: int = 0
) -> Dataset:
    """
    Load and encode DPO validation data.

    The validation data format is identical to training data:
    {
        "prompt": "...",
        "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        ...
    }

    Args:
        validation_files: Path(s) to JSONL file(s) containing preference pairs
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        sample_percentage: Fraction of data to sample (0.0 to 1.0)
        seed: Random seed for sampling

    Returns:
        HuggingFace Dataset with encoded preference pairs
    """
    raw_datasets = load_dpo_raw_dataset(
        validation_files, sample_percentage=sample_percentage, seed=seed
    )
    dpo_datasets = encode_dpo_data(raw_datasets, tokenizer, max_seq_length)
    return dpo_datasets


def get_dpo_validation_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int = 1
) -> DataLoader:
    """
    Create a DataLoader for DPO validation data.

    Args:
        dataset: HuggingFace Dataset with encoded preference pairs
        tokenizer: HuggingFace tokenizer (needed for padding)
        batch_size: Batch size (default 1 for gradient collection)

    Returns:
        PyTorch DataLoader
    """
    return get_dpo_dataloader(dataset, tokenizer, batch_size)
