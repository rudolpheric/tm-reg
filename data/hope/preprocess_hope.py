#!/usr/bin/env python3
"""
Preprocessing script for HOPE (Counselling Conversation) dataset.

This script:
1. Loads HOPE dataset from local CSV files
2. Maps speakers: T -> Counselor, P -> Client
3. Normalizes dialogue act labels (takes primary label for multi-labels)
4. Uses existing train/val/test splits from directory structure
5. Applies sliding window (size 5) to create training instances
6. Computes transition matrix from training data
7. Saves preprocessed data and transition matrix to disk

Usage:
    python preprocess_hope.py --output_dir ./hope_preprocessed
"""

import argparse
import glob
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# =============================================================================
# HOPE Dataset Configuration
# =============================================================================

# Canonical 12 dialogue act labels
TAG_NAMES = [
    'id',   # Information Delivery
    'irq',  # Information Request Question
    'gc',   # General Comment
    'crq',  # Closed Reflective Question
    'cd',   # Closed Delivery
    'yq',   # Yes/No Question
    'ack',  # Acknowledgement
    'op',   # Open (Positive)
    'gt',   # General Talk
    'on',   # Open Neutral
    'od',   # Open Delivery
    'orq',  # Open Reflective Question
    'cv',   # Change/Value (rare)
    'cr',   # Close Response (rare)
    'ci',   # Close Initiative (rare)
]

# Speaker mapping (matching full_categories terminology)
SPEAKER_MAP = {
    'T': 'Counselor',
    'P': 'Client',
}

# Data source path
DEFAULT_DATA_PATH = '/Users/ericrudolph/Development/02_Research_Projects/geccoalignment/nsp/dataprep/HOPE'


def normalize_label(label: str) -> Optional[str]:
    """
    Normalize a dialogue act label.

    - Takes primary label for multi-labels (e.g., "gc, irq" -> "gc")
    - Normalizes spacing and formatting
    - Returns None for empty/invalid labels

    Args:
        label: Raw label string from CSV

    Returns:
        Normalized label string, or None if invalid
    """
    if pd.isna(label) or not label or str(label).strip() == '':
        return None

    label = str(label).strip().lower()

    # Handle multi-labels: take the first/primary label
    # Possible separators: ", ", ",", " "
    if ',' in label:
        # Split by comma (with or without space)
        parts = re.split(r'\s*,\s*', label)
        label = parts[0].strip()
    elif ' ' in label and label not in TAG_NAMES:
        # Some labels have spaces like "ack irq"
        parts = label.split()
        label = parts[0].strip()

    # Remove any trailing/leading whitespace or special chars
    label = label.strip(' :')

    # Validate against canonical labels
    if label in TAG_NAMES:
        return label

    # Handle some known typos/variations
    label_fixes = {
        'acak': 'ack',
        'ay': 'ack',  # Likely typo for ack
        'ap': 'op',   # Likely typo for op
        'yp': 'yq',   # Likely typo for yq
        'vc': 'cv',   # Reversed
        'comp': 'cd',  # Likely completion -> closed delivery
        'com': 'cd',   # Likely completion -> closed delivery
    }

    if label in label_fixes:
        return label_fixes[label]

    # If still not recognized, check if it starts with a known label
    for known in TAG_NAMES:
        if label.startswith(known):
            return known

    # Unknown label - skip
    return None


def load_conversation_from_csv(csv_path: str) -> List[Dict]:
    """
    Load a single conversation from a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of utterance dicts with text, speaker, and dialogue act
    """
    df = pd.read_csv(csv_path)

    utterances = []

    # Determine the dialogue act column name (varies between files)
    act_col = 'Dialogue_Act' if 'Dialogue_Act' in df.columns else 'Dialog_Act'

    for idx, row in df.iterrows():
        speaker_raw = str(row.get('Type', '')).strip()
        text = str(row.get('Utterance', '')).strip()
        act_raw = row.get(act_col, '')

        # Skip rows with missing essential data
        if not speaker_raw or not text:
            continue

        # Map speaker
        speaker = SPEAKER_MAP.get(speaker_raw, speaker_raw)

        # Normalize label
        act = normalize_label(act_raw)

        utterances.append({
            'text': text,
            'act_tag': act,
            'speaker': speaker,
            'utterance_index': idx,
        })

    return utterances


def load_hope_dataset(data_path: str) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
    """
    Load the entire HOPE dataset from directory structure.

    Args:
        data_path: Path to HOPE dataset root directory

    Returns:
        Tuple of (conversations dict, split_conv_ids dict)
        - conversations: Dict mapping conv_id to list of utterances
        - split_conv_ids: Dict mapping split name to list of conv_ids
    """
    conversations = {}
    split_conv_ids = {'train': [], 'val': [], 'test': []}

    split_dirs = {
        'train': 'Train',
        'val': 'Validation',
        'test': 'Test',
    }

    for split_name, dir_name in split_dirs.items():
        split_path = os.path.join(data_path, dir_name)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            continue

        csv_files = glob.glob(os.path.join(split_path, '*.csv'))
        print(f"Loading {len(csv_files)} conversations from {split_name}...")

        for csv_path in tqdm(csv_files, desc=f"Loading {split_name}"):
            # Extract conversation ID from filename
            filename = os.path.basename(csv_path)
            # Remove "Copy of " prefix if present and .csv extension
            conv_id = filename.replace('Copy of ', '').replace('.csv', '')

            utterances = load_conversation_from_csv(csv_path)

            if len(utterances) > 0:
                conversations[conv_id] = utterances
                split_conv_ids[split_name].append(conv_id)

    return conversations, split_conv_ids


def create_sliding_windows(
    conversations: Dict[str, List[Dict]],
    conv_ids: List[str],
    window_size: int = 5
) -> List[Dict]:
    """
    Create sliding window instances for next DA prediction.

    For each window of `window_size` utterances, the task is to predict
    the DA of the last utterance given the previous utterances.

    Args:
        conversations: Dict mapping conversation_id to utterances
        conv_ids: List of conversation IDs to process
        window_size: Size of sliding window

    Returns:
        List of training instances with history and target
    """
    instances = []
    skipped_no_label = 0

    for conv_id in tqdm(conv_ids, desc="Creating windows"):
        utterances = conversations.get(conv_id, [])

        # Need at least window_size utterances
        if len(utterances) < window_size:
            continue

        # Create sliding windows
        for i in range(len(utterances) - window_size + 1):
            window = utterances[i:i + window_size]

            # History: first (window_size - 1) utterances
            history = window[:-1]
            # Target: last utterance
            target = window[-1]

            # Get tags
            history_tags = [u['act_tag'] for u in history]
            target_tag = target['act_tag']

            # Skip windows where target has no label
            if target_tag is None:
                skipped_no_label += 1
                continue

            # For history, replace None tags with 'unknown' or skip
            # We'll keep them but mark as None (model can handle)
            history_tags_clean = [t if t is not None else 'unknown' for t in history_tags]

            source_tag = history_tags_clean[-1]  # Last history tag for TM

            # Map speaker to K/B format for compatibility with training script
            # Client -> K (Klient), Counselor -> B (Berater)
            speaker_to_code = {'Client': 'K', 'Counselor': 'B'}

            # Build history text with speaker and tag info
            history_text_parts = []
            for u, tag in zip(history, history_tags_clean):
                speaker = u['speaker']
                speaker_code = speaker_to_code.get(speaker, speaker[0])
                text = u['text'].strip()
                # Format: "K (tag | Description): text" to match OnCoCo format
                history_text_parts.append(f"{speaker_code} ({tag}): {text}")

            history_text = "\n".join(history_text_parts)

            # Target speaker code
            target_speaker = target['speaker']
            target_speaker_code = speaker_to_code.get(target_speaker, target_speaker[0])

            # Create label with speaker prefix for compatibility
            # Format: "K-tag" or "B-tag"
            label = f"{target_speaker_code}-{target_tag}"

            instances.append({
                'conversation_id': conv_id,
                # Use both field names for compatibility
                'conversation_history': history_text,  # For train_speaker_change_classifier.py
                'history': history_text,  # Legacy field name
                'history_tags': history_tags_clean,
                'target_text': target['text'].strip(),
                'target_tag': target_tag,
                'target_speaker': target_speaker,
                'source_tag': source_tag,
                'speaker_change': 1 if history[-1]['speaker'] != target_speaker else 0,
                # Label with speaker prefix for training script
                'label': label,
            })

    if skipped_no_label > 0:
        print(f"  Skipped {skipped_no_label} windows with unlabeled targets")

    return instances


def compute_transition_matrix(instances: List[Dict], tag_names: List[str]) -> pd.DataFrame:
    """
    Compute transition matrix from training instances.

    Args:
        instances: List of training instances with source_tag and target_tag
        tag_names: List of canonical tag names

    Returns:
        DataFrame with transition probabilities (rows=source, cols=target)
    """
    # Include 'unknown' for unlabeled history items
    all_tags = tag_names + ['unknown']

    # Count transitions
    transitions = defaultdict(lambda: defaultdict(int))

    for inst in instances:
        source = inst['source_tag']
        target = inst['target_tag']
        if source in all_tags and target in tag_names:
            transitions[source][target] += 1

    # Convert to DataFrame (only canonical tags as both rows and columns for simplicity)
    tm = pd.DataFrame(index=tag_names, columns=tag_names, dtype=float)
    tm = tm.fillna(0.0)

    for source in tag_names:
        total = sum(transitions[source].values())
        if total > 0:
            for target in tag_names:
                tm.loc[source, target] = transitions[source][target] / total

    # Add smoothing for zero probabilities
    smoothing = 1e-6
    tm = tm + smoothing
    tm = tm.div(tm.sum(axis=1), axis=0)

    return tm


def verify_distribution(instances: List[Dict], name: str = "Dataset"):
    """Print tag distribution."""
    tag_counts = Counter(inst['target_tag'] for inst in instances)
    total = sum(tag_counts.values())

    print(f"\n{name} Distribution ({total} instances):")
    print("-" * 50)
    print(f"{'Tag':<15} {'Count':>10} {'Percent':>10}")
    print("-" * 50)

    for tag, count in tag_counts.most_common():
        percent = 100.0 * count / total if total > 0 else 0
        print(f"{tag:<15} {count:>10} {percent:>9.1f}%")

    print("-" * 50)


def save_preprocessed_data(
    output_dir: str,
    train_instances: List[Dict],
    val_instances: List[Dict],
    test_instances: List[Dict],
    transition_matrix: pd.DataFrame,
    label_encoder: LabelEncoder,
    split_info: Dict
):
    """Save all preprocessed data to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save instances as JSON
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_instances, f, indent=2)

    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_instances, f, indent=2)

    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_instances, f, indent=2)

    # Save transition matrix
    transition_matrix.to_csv(os.path.join(output_dir, 'transition_matrix.csv'))

    # Save label encoder
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save split info
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved preprocessed data to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess HOPE dataset for NDAP experiments")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to HOPE dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./hope_preprocessed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Sliding window size (default: 5, matching SWDA)')
    args = parser.parse_args()

    print("=" * 60)
    print("HOPE Dataset Preprocessing")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading HOPE dataset from {args.data_path}...")
    conversations, split_conv_ids = load_hope_dataset(args.data_path)

    print(f"\nLoaded {len(conversations)} total conversations")
    print(f"  Train: {len(split_conv_ids['train'])} conversations")
    print(f"  Val:   {len(split_conv_ids['val'])} conversations")
    print(f"  Test:  {len(split_conv_ids['test'])} conversations")

    # Create sliding windows for each split
    print(f"\nCreating sliding windows (size={args.window_size})...")

    train_instances = create_sliding_windows(
        conversations, split_conv_ids['train'], args.window_size
    )
    val_instances = create_sliding_windows(
        conversations, split_conv_ids['val'], args.window_size
    )
    test_instances = create_sliding_windows(
        conversations, split_conv_ids['test'], args.window_size
    )

    print(f"\nCreated instances:")
    print(f"  Train: {len(train_instances)}")
    print(f"  Val:   {len(val_instances)}")
    print(f"  Test:  {len(test_instances)}")

    # Verify distributions
    verify_distribution(train_instances, "Training")
    verify_distribution(val_instances, "Validation")
    verify_distribution(test_instances, "Test")

    # Compute transition matrix from training data
    print("\nComputing transition matrix from training data...")
    transition_matrix = compute_transition_matrix(train_instances, TAG_NAMES)
    print("\nTransition Matrix:")
    print(transition_matrix.round(3))

    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(TAG_NAMES)

    # Prepare split info
    split_info = {
        'train_conversation_ids': split_conv_ids['train'],
        'val_conversation_ids': split_conv_ids['val'],
        'test_conversation_ids': split_conv_ids['test'],
        'window_size': args.window_size,
        'n_train': len(train_instances),
        'n_val': len(val_instances),
        'n_test': len(test_instances),
        'tag_names': TAG_NAMES,
        'speaker_map': SPEAKER_MAP,
        'dataset': 'HOPE',
    }

    # Save everything
    save_preprocessed_data(
        args.output_dir,
        train_instances,
        val_instances,
        test_instances,
        transition_matrix,
        label_encoder,
        split_info
    )

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Total instances: {len(train_instances) + len(val_instances) + len(test_instances)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
