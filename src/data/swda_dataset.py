"""
SwDA (Switchboard Dialog Act) Dataset for Next Dialog Act Prediction.

This module provides dataset classes for loading and processing the preprocessed
SwDA corpus following Tanaka et al. (2019) methodology with 9 collapsed tags.
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


# 9 canonical tag names following Tanaka et al. (2019)
SWDA_TAG_NAMES = [
    'Statement', 'Understanding', 'Uninterpretable', 'Agreement',
    'Question', 'Other', 'Apology', 'Greeting', 'Directive'
]


class SwDADataset(Dataset):
    """
    Dataset class for SwDA next dialog act prediction.

    Loads preprocessed SwDA data and provides tokenized inputs for training.
    Compatible with the existing training pipeline.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        label_encoder: LabelEncoder,
        max_input_length: int = 512,
        num_history_sentences: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        random_seed: int = 42,
    ):
        """
        Args:
            data_path: Path to the JSON file with preprocessed instances
            tokenizer: HuggingFace tokenizer
            label_encoder: Fitted LabelEncoder for the 9-tag categories
            max_input_length: Maximum token length for input
            num_history_sentences: If set, only use last N sentences from history
            sample_fraction: If set, use only this fraction of data
            random_seed: Random seed for sampling
        """
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_input_length = max_input_length
        self.num_history_sentences = num_history_sentences

        # Load instances from JSON
        with open(data_path, 'r') as f:
            self.instances = json.load(f)

        # Optional sampling
        if sample_fraction is not None and 0 < sample_fraction < 1:
            random.seed(random_seed)
            total = len(self.instances)
            sample_size = int(total * sample_fraction)
            random.shuffle(self.instances)
            self.instances = self.instances[:sample_size]
            print(f"SwDA dataset sampled: {sample_size} / {total} instances")

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst = self.instances[idx]

        # Get history text (optionally truncate to last N sentences)
        history = inst['history']
        if self.num_history_sentences is not None:
            sentences = history.split('\n')
            if len(sentences) > self.num_history_sentences:
                history = '\n'.join(sentences[-self.num_history_sentences:])

        # Tokenize with left truncation (keep most recent context)
        self.tokenizer.truncation_side = 'left'
        inputs = self.tokenizer(
            history,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_input_length,
            truncation=True,
        )

        # Get target label
        target_tag = inst['target_tag']
        target_encoded = self.label_encoder.transform([target_tag])[0]
        target_tensor = torch.tensor(target_encoded, dtype=torch.long)

        # Speaker change (binary)
        speaker_change = torch.tensor(inst['speaker_change'], dtype=torch.float)

        # Category history for history-aware models
        # Convert history_tags to encoded IDs, padding with num_classes (padding token)
        history_tags = inst.get('history_tags', [])
        if self.num_history_sentences is not None and len(history_tags) > self.num_history_sentences:
            history_tags = history_tags[-self.num_history_sentences:]

        # Encode history tags (use num_classes as padding token for unknown/missing)
        num_classes = len(self.label_encoder.classes_)
        category_history_ids = []
        for tag in history_tags:
            try:
                encoded = self.label_encoder.transform([tag])[0]
                category_history_ids.append(encoded)
            except ValueError:
                # Unknown tag, use padding token
                category_history_ids.append(num_classes)

        # Pad to fixed length
        max_history = self.num_history_sentences or 5
        while len(category_history_ids) < max_history:
            category_history_ids.insert(0, num_classes)  # Pad at beginning
        category_history_ids = category_history_ids[-max_history:]  # Keep last N

        category_history_tensor = torch.tensor(category_history_ids, dtype=torch.long)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'raw_history': history,
            'target_category_name': target_tag,
            'category_label': target_tensor,
            'speaker_change_label': speaker_change,
            'source_category': inst['source_tag'],
            'category_history_ids': category_history_tensor,
        }


class SwDAHierarchyDataset(Dataset):
    """
    SwDA dataset wrapper with per-utterance tokenization for hierarchy models.

    This is compatible with the existing CategoryHierarchyDataset interface,
    but simplified for the flat 9-tag SwDA structure.
    """

    def __init__(
        self,
        base_dataset: SwDADataset,
        max_utterances: int = 5,
        max_length: int = 128,
    ):
        """
        Args:
            base_dataset: Base SwDADataset instance
            max_utterances: Maximum number of history utterances to process
            max_length: Maximum token length per utterance
        """
        self.base_dataset = base_dataset
        self.max_utterances = max_utterances
        self.max_length = max_length
        self.tokenizer = base_dataset.tokenizer
        self.label_encoder = base_dataset.label_encoder

        # Create a simple tag-to-index mapping for history tags
        self.tag_to_idx = {tag: i for i, tag in enumerate(SWDA_TAG_NAMES)}
        self.num_tags = len(SWDA_TAG_NAMES)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst = self.base_dataset.instances[idx]

        # Parse history lines
        history_lines = inst['history'].split('\n')
        history_tags = inst['history_tags']

        # Take last N utterances
        if len(history_lines) > self.max_utterances:
            history_lines = history_lines[-self.max_utterances:]
            history_tags = history_tags[-self.max_utterances:]

        # Tokenize each utterance separately
        # Extract just the text part after the tag
        utterance_texts = []
        for line in history_lines:
            # Format: "1043 (Other): Okay. /"
            if '): ' in line:
                text = line.split('): ', 1)[1]
            else:
                text = line
            utterance_texts.append(text.strip())

        # Tokenize utterances
        utterance_ids = []
        utterance_masks = []

        self.tokenizer.truncation_side = 'right'
        for text in utterance_texts:
            enc = self.tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
            )
            utterance_ids.append(enc['input_ids'].squeeze(0))
            utterance_masks.append(enc['attention_mask'].squeeze(0))

        # Pad to max_utterances if needed
        while len(utterance_ids) < self.max_utterances:
            utterance_ids.insert(0, torch.zeros(self.max_length, dtype=torch.long))
            utterance_masks.insert(0, torch.zeros(self.max_length, dtype=torch.long))
            history_tags.insert(0, 'Statement')  # Pad with most common tag

        # Stack into tensors
        utterance_ids = torch.stack(utterance_ids)
        utterance_masks = torch.stack(utterance_masks)

        # Convert history tags to indices
        history_tag_indices = [self.tag_to_idx.get(tag, 0) for tag in history_tags]
        history_tag_tensor = torch.tensor(history_tag_indices, dtype=torch.long)

        # Get target label
        target_tag = inst['target_tag']
        target_encoded = self.label_encoder.transform([target_tag])[0]
        target_tensor = torch.tensor(target_encoded, dtype=torch.long)

        # Speaker change
        speaker_change = torch.tensor(inst['speaker_change'], dtype=torch.float)

        return {
            'utterance_ids': utterance_ids,
            'utterance_masks': utterance_masks,
            'history_tags': history_tag_tensor,
            'category_label': target_tensor,
            'speaker_change_label': speaker_change,
            'source_category': inst['source_tag'],
            'target_category_name': target_tag,
        }


def load_swda_datasets(
    data_dir: str,
    tokenizer,
    max_input_length: int = 512,
    num_history_sentences: Optional[int] = None,
) -> Tuple[SwDADataset, SwDADataset, SwDADataset, LabelEncoder]:
    """
    Load train, val, test SwDA datasets with shared label encoder.

    Args:
        data_dir: Path to directory with preprocessed SwDA data
        tokenizer: HuggingFace tokenizer
        max_input_length: Maximum token length for input
        num_history_sentences: If set, only use last N sentences from history

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label_encoder)
    """
    data_dir = Path(data_dir)

    # Load label encoder
    with open(data_dir / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Load datasets
    train_ds = SwDADataset(
        data_path=str(data_dir / 'train.json'),
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_input_length=max_input_length,
        num_history_sentences=num_history_sentences,
    )

    test_ds = SwDADataset(
        data_path=str(data_dir / 'test.json'),
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_input_length=max_input_length,
        num_history_sentences=num_history_sentences,
    )

    # Check if val.json exists and has data (may be empty if combine_val_test was used)
    val_path = data_dir / 'val.json'
    if val_path.exists():
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        if len(val_data) > 0:
            val_ds = SwDADataset(
                data_path=str(val_path),
                tokenizer=tokenizer,
                label_encoder=label_encoder,
                max_input_length=max_input_length,
                num_history_sentences=num_history_sentences,
            )
        else:
            # Empty val set - use test set for validation
            val_ds = test_ds
    else:
        # No val file - use test set for validation
        val_ds = test_ds

    return train_ds, val_ds, test_ds, label_encoder


def load_swda_transition_matrix(data_dir: str) -> torch.Tensor:
    """
    Load the precomputed transition matrix for SwDA.

    Args:
        data_dir: Path to directory with preprocessed SwDA data

    Returns:
        Transition matrix tensor of shape (num_tags, num_tags)
    """
    import pandas as pd

    data_dir = Path(data_dir)
    tm_path = data_dir / 'transition_matrix.csv'

    tm_df = pd.read_csv(tm_path, index_col=0)
    tm_tensor = torch.tensor(tm_df.values, dtype=torch.float32)

    return tm_tensor
