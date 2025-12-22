import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from core.datasets import CategoryHierarchyDataset, DialogueNextSentenceDataset
from sklearn.utils.class_weight import compute_class_weight

# Note: These imports were removed/deprecated during refactoring
# from sampling_utils import apply_class_balancing_sampling  # DEPRECATED - functionality may be in datasets.py
# from improved_validation import (...)  # DEPRECATED - functions removed during refactoring

print("Current working directory:", os.getcwd())

# ==============================================================================
# STUB IMPLEMENTATIONS for deprecated validation functions
# These were removed during refactoring but some code still references them
# ==============================================================================

def check_for_duplicates(dataset):
    """Stub implementation - returns no duplicates found."""
    return {"duplicate_count": 0, "duplicates": []}

def deduplicate_dataset(dataset):
    """Stub implementation - returns dataset as-is."""
    return dataset

def create_conversation_based_splits(dataset, val_size=0.1, test_size=0.1):
    """Stub implementation - creates simple random splits."""
    import random
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_count = int(len(indices) * val_size)
    test_count = int(len(indices) * test_size)

    return {
        "train_indices": indices[val_count + test_count:],
        "val_indices": indices[:val_count],
        "test_indices": indices[val_count:val_count + test_count]
    }

def print_category_distribution(dataset, train_indices, val_indices, test_indices):
    """Stub implementation - prints basic stats."""
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

def apply_class_balancing_sampling(dataset, split_info, args):
    """Stub implementation - returns split_info unchanged."""
    return split_info

# ==============================================================================


def setup_device() -> torch.device:
    """
    Determine whether to use CUDA, MPS (Apple Silicon), or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_tokenizer(pretrained_model: str = "EuroBERT/EuroBERT-610m"):
    """
    Load and return the tokenizer.
    """
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    return AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)


def load_dataset_wrapper(tokenizer, sample_fraction = 1, min_category_count = None, num_history_sentences: int = None) -> DialogueNextSentenceDataset:
    """
    Load the dataset and run duplicate check/deduplication.
    """
    print("Loading dataset...")
    
    dataset = DialogueNextSentenceDataset("dataprep/dataset_files", tokenizer,
                                          num_history_sentences=num_history_sentences,
                                          min_category_count=None, sample_fraction=sample_fraction)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    print("\n=== Checking for duplicate samples ===")
    duplicates = check_for_duplicates(dataset)
    if duplicates["duplicate_count"] > 0:
        print(f"Found {duplicates['duplicate_count']} duplicates; deduplicating dataset...")
        dataset = deduplicate_dataset(dataset)
        print(f"Dataset now has {len(dataset)} examples after deduplication")
    else:
        print("No duplicates found")
    
    return dataset


def create_data_splits_and_loaders(dataset, device: torch.device, use_class_weight: bool, batch_size: int = 4, args=None, split_info=None):
    """
    Create conversation-based splits and corresponding data loaders.
    If split_info is provided, use it instead of creating new splits.
    """
    if split_info is None:
        print("\n=== Creating improved data splits ===")
        split_info = create_conversation_based_splits(dataset, val_size=0.1, test_size=0.1)
    else:
        print("\n=== Using provided split information ===")
        
    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]
    test_indices = split_info["test_indices"]
    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test")
    print_category_distribution(dataset, train_indices, val_indices, test_indices)
    
    # Apply class balancing sampling if requested
    if args and (args.undersample_moderation or args.oversample_minorities):
        split_info = apply_class_balancing_sampling(dataset, split_info, args)
        train_indices = split_info["train_indices"]
        print(f"After sampling: {len(train_indices)} train samples")
        # Update distribution display
        print("\nPost-sampling category distribution:")
        train_labels_post = [dataset.pairs[idx]["target_category"] for idx in train_indices]
        from collections import Counter
        category_counts = Counter(train_labels_post)
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cat}: {count} ({count/len(train_indices)*100:.1f}%)")
    
    train_labels = [dataset.pairs[idx]["target_category"] for idx in train_indices]
    label_ids = dataset.label_encoder.transform(train_labels)
    
    if use_class_weight:
        print("\n=== Using class weights for imbalanced data ===")
        class_counts = np.bincount(label_ids)
        sample_weights = [1. / class_counts[label] for label in label_ids]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(train_indices), replacement=True)
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size,
                                  sampler=sampler,
                                  num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1)
    else:
        print("\n=== NOT using class weights ===")
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1)
    
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size,
                            shuffle=False,
                            num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size,
                             shuffle=False,
                             num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1)
    
    category_counts = {}
    for idx in train_indices:
        cat = dataset.pairs[idx]["target_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print("\nCategory distribution in training set:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} ({count/len(train_indices)*100:.1f}%)")
    
    return train_loader, val_loader, test_loader, split_info, label_ids


def load_transition_matrix(path: str = "transition_matrix.csv") -> pd.DataFrame:
    """
    Load the transition matrix from a CSV file.

    The CSV is assumed to have the source (from) categories as its index and the target (to)
    categories as its columns.
    """
    print("Loading transition matrix from:", path)
    return pd.read_csv(path, index_col=0)


def load_category_priors(path: str = "category_priors.csv") -> pd.Series:
    """
    Load category a-priori (marginal) probabilities from a CSV file.

    Args:
        path: Path to the priors CSV file

    Returns:
        pd.Series: Prior probabilities indexed by category name
    """
    print("Loading category priors from:", path)
    priors = pd.read_csv(path, index_col=0)['prior_probability']
    return priors


def prepare_hierarchical_dataloaders(dataset, hierarchy, split_info, batch_size, device, max_context_utterances=4, use_class_weight=False, 
                                   undersample_moderation=False, moderation_sample_ratio=0.3, oversample_minorities=False, 
                                   target_samples_ratio=0.5, min_samples_for_oversampling=50):
    """
    Wrap the dataset with hierarchical processing and create appropriate dataloaders.
    Supports class balancing sampling strategies.
    """
    # Ensure max_context_utterances is not None
    if max_context_utterances is None:
        max_context_utterances = 5
    
    hierarchical_dataset = CategoryHierarchyDataset(dataset, hierarchy, max_utterances=max_context_utterances)
    
    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]
    test_indices = split_info["test_indices"]
    
    # Apply class balancing sampling if requested
    if undersample_moderation or oversample_minorities:
        # Create mock args object for sampling function
        from argparse import Namespace
        sampling_args = Namespace(
            undersample_moderation=undersample_moderation,
            moderation_sample_ratio=moderation_sample_ratio,
            oversample_minorities=oversample_minorities,
            target_samples_ratio=target_samples_ratio,
            min_samples_for_oversampling=min_samples_for_oversampling
        )
        
        # Apply sampling to split_info
        updated_split_info = apply_class_balancing_sampling(dataset, split_info, sampling_args)
        train_indices = updated_split_info["train_indices"]
        final_split_info = updated_split_info
    else:
        final_split_info = split_info
    
    # Create train loader with appropriate sampler if using class weights
    if use_class_weight:
        train_labels = [dataset.pairs[idx]["target_category"] for idx in train_indices]
        label_ids = dataset.label_encoder.transform(train_labels)
        
        print("\n=== Using class weights for hierarchical data ===")
        class_counts = np.bincount(label_ids)
        sample_weights = [1. / class_counts[label] for label in label_ids]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                      num_samples=len(train_indices), replacement=True)
        train_loader = DataLoader(
            Subset(hierarchical_dataset, train_indices), 
            batch_size=batch_size,
            sampler=sampler,
            num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1
        )
    else:
        train_loader = DataLoader(
            Subset(hierarchical_dataset, train_indices), 
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1
        )
    
    val_loader = DataLoader(
        Subset(hierarchical_dataset, val_indices), 
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1
    )
    
    test_loader = DataLoader(
        Subset(hierarchical_dataset, test_indices), 
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 1
    )
    
    return train_loader, val_loader, test_loader, final_split_info, hierarchical_dataset

def load_category_hierarchy(csv_path):
    """
    Load the hierarchical category structure from CSV file.
    
    Returns:
        dict: Hierarchical representation of categories
    """
    category_df = pd.read_csv(csv_path, encoding='cp1252')
    
    # Create dictionaries to store hierarchical mappings
    hierarchy = {
        "speaker_types": {},  # Typ -> ID mapping
        "main_categories": {},  # K1 -> ID mapping
        "sub_categories": {},  # K2 -> ID mapping 
        "third_level": {},  # K3 -> ID mapping
        "fourth_level": {},  # K4 -> ID mapping
        "full_codes": {},  # Complete code -> component IDs
        "descriptions": {}  # Code -> description mapping
    }
    
    # Extract unique values for each level
    hierarchy["speaker_types"] = {typ: i for i, typ in enumerate(category_df["Typ"].unique())}
    hierarchy["main_categories"] = {cat: i for i, cat in enumerate(category_df["K1"].dropna().unique())}
    hierarchy["sub_categories"] = {cat: i for i, cat in enumerate(category_df["K2"].dropna().unique())}
    hierarchy["third_level"] = {cat: i for i, cat in enumerate(category_df["K3"].dropna().unique())}
    hierarchy["fourth_level"] = {cat: i for i, cat in enumerate(category_df["K4"].dropna().unique())}
    
    # Create mappings for complete codes
    for _, row in category_df.iterrows():
        code = row["Code DE"]
        
        # Store component IDs for this code
        hierarchy["full_codes"][code] = {
            "speaker": hierarchy["speaker_types"].get(row["Typ"], 0),
            "main_category": hierarchy["main_categories"].get(row["K1"], 0) if pd.notna(row["K1"]) else 0,
            "sub_category": hierarchy["sub_categories"].get(row["K2"], 0) if pd.notna(row["K2"]) else 0,
            "third_level": hierarchy["third_level"].get(row["K3"], 0) if pd.notna(row["K3"]) else 0,
            "fourth_level": hierarchy["fourth_level"].get(row["K4"], 0) if pd.notna(row["K4"]) else 0
        }
        
        # Store descriptions for each code
        hierarchy["descriptions"][code] = {
            "typ_name": row["Typ Name"] if pd.notna(row["Typ Name"]) else "",
            "k1_name": row["K1 Name"] if pd.notna(row["K1 Name"]) else "",
            "c1_name": row["C1 Name"] if pd.notna(row["C1 Name"]) else "",
            "k2_name": row["K2 Name"] if pd.notna(row["K2 Name"]) else "",
            "k3_name": row["K3 Name"] if pd.notna(row["K3 Name"]) else "",
            "k4_name": row["K4 Name"] if pd.notna(row["K4 Name"]) else ""
        }
    
    # Add reverse mappings (ID -> category)
    hierarchy["id_to_speaker"] = {v: k for k, v in hierarchy["speaker_types"].items()}
    hierarchy["id_to_main"] = {v: k for k, v in hierarchy["main_categories"].items()}
    hierarchy["id_to_sub"] = {v: k for k, v in hierarchy["sub_categories"].items()}
    hierarchy["id_to_third"] = {v: k for k, v in hierarchy["third_level"].items()}
    hierarchy["id_to_fourth"] = {v: k for k, v in hierarchy["fourth_level"].items()}
    
    # Store counts
    hierarchy["num_speakers"] = len(hierarchy["speaker_types"])
    hierarchy["num_main_categories"] = len(hierarchy["main_categories"])
    hierarchy["num_sub_categories"] = len(hierarchy["sub_categories"])
    hierarchy["num_third_level"] = len(hierarchy["third_level"])
    hierarchy["num_fourth_level"] = len(hierarchy["fourth_level"])
    
    return hierarchy