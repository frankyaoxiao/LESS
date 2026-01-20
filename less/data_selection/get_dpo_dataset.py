"""
DPO (Direct Preference Optimization) dataset loading and encoding.

This module handles loading preference pair data in the format:
{
    "prompt": "...",
    "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    ...
}

And converts it to the format needed for DPO gradient collection:
{
    "chosen_input_ids": Tensor,
    "chosen_labels": Tensor,
    "chosen_attention_mask": Tensor,
    "rejected_input_ids": Tensor,
    "rejected_labels": Tensor,
    "rejected_attention_mask": Tensor,
}
"""

import contextlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


@contextlib.contextmanager
def temp_seed(seed):
    """Temporarily set random seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_dpo_training_dataset(
    train_files: Union[List[str], str],
    tokenizer,
    max_seq_length: int,
    sample_percentage: float = 1.0,
    seed: int = 0
) -> Dataset:
    """
    Load and encode DPO training data.

    Args:
        train_files: Path(s) to JSONL file(s) containing preference pairs
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        sample_percentage: Fraction of data to sample (0.0 to 1.0)
        seed: Random seed for sampling

    Returns:
        HuggingFace Dataset with encoded preference pairs
    """
    raw_datasets = load_dpo_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed
    )
    dpo_datasets = encode_dpo_data(raw_datasets, tokenizer, max_seq_length)
    return dpo_datasets


def load_dpo_raw_dataset(
    train_files: Union[List[str], str],
    sample_size: Optional[int] = None,
    sample_percentage: float = 1.0,
    seed: int = 0
) -> Dataset:
    """Load raw DPO dataset from JSONL files."""
    if isinstance(train_files, str):
        train_files = [train_files]

    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]

    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)
    return sampled_dataset


def encode_dpo_data(
    raw_datasets: Dataset,
    tokenizer,
    max_seq_length: int,
    processing_num_workers: int = 10,
    overwrite_cache: bool = False
) -> Dataset:
    """Encode DPO preference pairs."""
    # Check if already encoded
    if "chosen_input_ids" in raw_datasets.features:
        return raw_datasets

    encode_function = partial(
        encode_dpo_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    dpo_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing DPO preference pairs",
    )
    dpo_datasets.set_format(type="pt")
    return dpo_datasets


def concat_messages_tulu(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Concatenate messages using Tulu/OLMo chat format.

    Format:
    <|user|>
    {user_content}
    <|assistant|>
    {assistant_content}<eos>
    """
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(f"Invalid role: {message['role']}")
    return message_text


def encode_single_response(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int
) -> Dict[str, torch.Tensor]:
    """
    Encode a single conversation (prompt + response) for DPO.

    Returns input_ids, labels (with prompt masked), and attention_mask.
    """
    # Format the full conversation
    full_text = concat_messages_tulu(messages, tokenizer)

    # Tokenize the full conversation
    tokenized = tokenizer(
        full_text,
        return_tensors='pt',
        max_length=max_seq_length,
        truncation=True
    )
    input_ids = tokenized.input_ids
    labels = input_ids.clone()

    # Mask non-assistant tokens (we only compute loss on assistant responses)
    # Find where the assistant response starts
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # Get the text up to and including this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                prefix_text = concat_messages_tulu(messages[:message_idx], tokenizer)
                prefix_tokens = tokenizer(
                    prefix_text,
                    return_tensors='pt',
                    max_length=max_seq_length,
                    truncation=True
                )
                message_start_idx = prefix_tokens.input_ids.shape[1]

            # Get the text up to the next message (or include assistant role marker)
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                messages_so_far = concat_messages_tulu(messages[:message_idx + 1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages_tulu(messages[:message_idx + 1], tokenizer)

            message_end_tokens = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            )
            message_end_idx = message_end_tokens.input_ids.shape[1]

            # Mask this region
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_dpo_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_length: int
) -> Dict[str, torch.Tensor]:
    """
    Encode a single DPO example (preference pair).

    Supports two input formats:

    Format 1 (message lists):
    {
        "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    }

    Format 2 (flat strings):
    {
        "prompt": "...",
        "accepted": "...",  # or "chosen"
        "rejected": "...",
    }

    Output format:
    {
        "chosen_input_ids": Tensor,
        "chosen_labels": Tensor,
        "chosen_attention_mask": Tensor,
        "rejected_input_ids": Tensor,
        "rejected_labels": Tensor,
        "rejected_attention_mask": Tensor,
    }
    """
    # Detect format and normalize to message lists
    if "chosen" in example and isinstance(example["chosen"], list):
        # Format 1: already has message lists
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
    else:
        # Format 2: flat strings - convert to message format
        prompt = example.get("prompt", "")
        # Support both "accepted" and "chosen" as keys for the preferred response
        chosen_response = example.get("accepted", example.get("chosen", ""))
        rejected_response = example.get("rejected", "")

        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_response}
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_response}
        ]

    # Encode chosen response
    chosen_encoded = encode_single_response(chosen_messages, tokenizer, max_seq_length)

    # Encode rejected response
    rejected_encoded = encode_single_response(rejected_messages, tokenizer, max_seq_length)

    return {
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_labels": chosen_encoded["labels"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_labels": rejected_encoded["labels"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
    }


@dataclass
class DPODataCollator:
    """
    Data collator for DPO preference pairs.

    Handles padding for both chosen and rejected sequences.
    """
    tokenizer: Any
    padding: str = "longest"
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of DPO examples."""
        # Separate chosen and rejected
        chosen_features = [{
            "input_ids": f["chosen_input_ids"],
            "labels": f["chosen_labels"],
            "attention_mask": f["chosen_attention_mask"],
        } for f in features]

        rejected_features = [{
            "input_ids": f["rejected_input_ids"],
            "labels": f["rejected_labels"],
            "attention_mask": f["rejected_attention_mask"],
        } for f in features]

        # Pad chosen sequences
        chosen_batch = self._pad_sequences(chosen_features)

        # Pad rejected sequences
        rejected_batch = self._pad_sequences(rejected_features)

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_labels": chosen_batch["labels"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_labels": rejected_batch["labels"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }

    def _pad_sequences(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Pad a list of sequences to the same length."""
        # Find max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            attention_mask = f["attention_mask"]

            # Truncate if needed
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                attention_mask = attention_mask[:max_len]

            # Pad if needed
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
        }


def get_dpo_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int = 1
) -> DataLoader:
    """
    Create a DataLoader for DPO data.

    Args:
        dataset: HuggingFace Dataset with encoded preference pairs
        tokenizer: HuggingFace tokenizer (needed for padding)
        batch_size: Batch size (default 1 for gradient collection)

    Returns:
        PyTorch DataLoader
    """
    data_collator = DPODataCollator(tokenizer=tokenizer, padding="longest")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    print(f"Created DPO dataloader with {len(dataset)} examples")
    return dataloader
