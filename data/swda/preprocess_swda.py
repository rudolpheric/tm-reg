#!/usr/bin/env python3
"""
Preprocessing script for Switchboard Dialog Act (SwDA) corpus.

This script:
1. Loads SwDA from HuggingFace
2. Collapses 42 DAMSL tags to 9 categories (following Tanaka et al. 2019)
3. Splits conversations 80/10/10 at conversation level
4. Applies sliding window (size 5) to create training instances
5. Computes transition matrix from training data
6. Saves preprocessed data and transition matrix to disk

Usage:
    python preprocess_swda.py --output_dir ./swda_preprocessed
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# =============================================================================
# DAMSL to 9-Tag Mapping (inferred from Tanaka et al. 2019 Table 2)
# =============================================================================

DAMSL_TO_9_TAGS = {
    # Statement (576,005) - highest frequency
    'sd': 'Statement',      # Statement-non-opinion (most frequent tag)
    'sv': 'Statement',      # Statement-opinion
    '^e': 'Statement',      # Statement expanding y/n answer

    # Understanding (241,008) - backchannels
    'b': 'Understanding',   # Acknowledge/Backchannel
    'bk': 'Understanding',  # Acknowledge-answer
    'bh': 'Understanding',  # Backchannel in question form
    'ba': 'Understanding',  # Appreciation
    'br': 'Understanding',  # Signal non-understanding

    # Uninterpretable (93,238)
    '%': 'Uninterpretable', # Uninterpretable
    'x': 'Uninterpretable', # Non-verbal
    # '+' (continuation) is skipped, not mapped to any category

    # Agreement (55,375)
    'aa': 'Agreement',      # Agree/Accept
    'aap': 'Agreement',     # Partial agree (aap_am)
    'am': 'Agreement',      # Maybe
    'ar': 'Agreement',      # Reject
    'arp': 'Agreement',     # Partial reject (arp_nd)
    'nd': 'Agreement',      # Dispreferred answers (No/Disagree marker)

    # Question (54,498)
    'qy': 'Question',       # Yes-no question
    'qw': 'Question',       # Wh-question
    'qo': 'Question',       # Open-ended question
    'qr': 'Question',       # Or-question
    'qrr': 'Question',      # Or-clause
    '^d': 'Question',       # Declarative question
    '^g': 'Question',       # Tag question
    'qh': 'Question',       # Rhetorical question

    # Other (19,882)
    'h': 'Other',           # Hedge
    'fo': 'Other',          # Other forward function (fo_o_fw_by_bc)
    'fc': 'Other',          # Conventional closing
    'ny': 'Other',          # Yes answer
    'nn': 'Other',          # No answer
    'na': 'Other',          # Affirmative non-yes answer
    'ng': 'Other',          # Negative non-no answer
    'no': 'Other',          # Other answer
    '^h': 'Other',          # Hold
    '^q': 'Other',          # Quotation
    'bf': 'Other',          # Reformulate (bf_sg)
    'bd': 'Other',          # Downplayer
    't1': 'Other',          # Self-talk
    't3': 'Other',          # Third-party-talk
    '^2': 'Other',          # Collaborative completion
    'bc': 'Other',          # Brief correction (bc_comment)
    'o': 'Other',           # Other forward-looking function
    'oo': 'Other',          # Other opening
    'fe': 'Other',          # Exclamation
    'fx': 'Other',          # Floor mechanism
    '"': 'Other',           # Quotation (alternate marker)

    # Apology (11,446) - social acts
    'fa': 'Apology',        # Apology
    'ft': 'Apology',        # Thanking
    'fw': 'Apology',        # Welcome (often combined: fo_o_fw_by_bc)
    'by': 'Apology',        # Sympathy

    # Greeting (6,618)
    'fp': 'Greeting',       # Conventional opening

    # Directive (3,685) - lowest frequency
    'ad': 'Directive',      # Action-directive
    'co': 'Directive',      # Offer (commit/offer)
    'cc': 'Directive',      # Commit
}

# Target distribution from Tanaka et al. (2019) Table 2
TARGET_DISTRIBUTION = {
    'Statement': 576005,
    'Understanding': 241008,
    'Uninterpretable': 93238,
    'Agreement': 55375,
    'Question': 54498,
    'Other': 19882,
    'Apology': 11446,
    'Greeting': 6618,
    'Directive': 3685,
}

# 9 canonical tag names
TAG_NAMES = ['Statement', 'Understanding', 'Uninterpretable', 'Agreement',
             'Question', 'Other', 'Apology', 'Greeting', 'Directive']


def clean_act_tag(act_tag: str) -> Optional[str]:
    """
    Clean and normalize a DAMSL act tag.

    SwDA tags can have complex forms like 'sd^e' or 'qy^d'.
    We extract the base tag for mapping.

    Returns:
        Cleaned tag string, or None if the tag should be skipped (e.g., '+' continuation)
    """
    if not act_tag or act_tag == 'nan':
        return '%'  # Map to Uninterpretable

    # Handle combined tags (e.g., 'sd^e' -> try 'sd' first, then '^e')
    act_tag = act_tag.strip().lower()

    # Skip continuation markers - they don't represent a dialog act
    if act_tag == '+':
        return None

    # Some tags have underscores for combinations (e.g., 'fo_o_fw_by_bc')
    # Try the full tag first
    if act_tag in DAMSL_TO_9_TAGS:
        return act_tag

    # Try extracting base tag (before ^)
    if '^' in act_tag:
        base = act_tag.split('^')[0]
        modifier = '^' + act_tag.split('^')[1].split('(')[0] if '^' in act_tag else ''

        # Try base tag
        if base and base in DAMSL_TO_9_TAGS:
            return base
        # Try modifier as tag (e.g., '^d', '^g', '^e')
        if modifier in DAMSL_TO_9_TAGS:
            return modifier

    # Try handling parenthetical modifiers like 'sd(^q)'
    if '(' in act_tag:
        base = act_tag.split('(')[0]
        if base in DAMSL_TO_9_TAGS:
            return base

    # Try first part of underscore-separated tags
    if '_' in act_tag:
        first_part = act_tag.split('_')[0]
        if first_part in DAMSL_TO_9_TAGS:
            return first_part

    # Default: return as-is and let caller handle unknown
    return act_tag


def map_to_9_tags(act_tag: str) -> Optional[str]:
    """
    Map a DAMSL act tag to one of 9 categories.

    Returns:
        Mapped category name, or None if tag should be skipped
    """
    cleaned = clean_act_tag(act_tag)

    if cleaned is None:
        return None  # Skip continuations

    if cleaned in DAMSL_TO_9_TAGS:
        return DAMSL_TO_9_TAGS[cleaned]

    # Fallback for unknown tags
    return 'Other'


def install_convokit():
    """Install convokit package."""
    print("Installing convokit...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "convokit"
    ])
    # Download NLTK punkt tokenizer
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def load_swda_from_convokit() -> List[Dict]:
    """
    Load SwDA dataset using ConvoKit at SEGMENT level.

    The ConvoKit SwDA format has utterances with 'tag' metadata containing
    a list of [text_segment, act_tag] pairs. Each segment is a separate DA unit
    following Tanaka et al. (2019) methodology.

    Returns:
        List of segments with metadata (each segment has its own text and act_tag)
    """
    try:
        from convokit import Corpus, download
    except ImportError:
        install_convokit()
        from convokit import Corpus, download

    print("Loading SwDA corpus via ConvoKit...")
    corpus = Corpus(filename=download("switchboard-corpus"))

    segments = []
    segment_idx = 0  # Global segment index within conversation

    for conv in tqdm(corpus.iter_conversations(), desc="Loading conversations"):
        conv_id = conv.id
        # Get utterances sorted by timestamp/index
        conv_utts = sorted(conv.iter_utterances(), key=lambda u: u.timestamp if u.timestamp else 0)

        conv_segment_idx = 0  # Segment index within this conversation
        for utt_idx, utt in enumerate(conv_utts):
            # Extract dialog act segments from metadata
            # Format: 'tag' is a list of [text_segment, act_tag] pairs
            tag_data = utt.meta.get('tag', [])

            # Map speaker ID
            speaker_id = str(utt.speaker.id) if utt.speaker else 'A'

            if isinstance(tag_data, list) and len(tag_data) > 0:
                for seg_idx, segment in enumerate(tag_data):
                    if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                        seg_text = segment[0] if segment[0] else ''
                        act_tag = segment[1] if segment[1] else ''

                        # Skip continuation markers - they don't represent independent DAs
                        if act_tag == '+':
                            continue

                        segments.append({
                            'text': seg_text.strip(),
                            'act_tag': act_tag,
                            'damsl_act_tag': act_tag,
                            'caller': speaker_id,
                            'transcript_index': conv_segment_idx,
                            'utterance_index': utt_idx,  # Original utterance index
                            'segment_index': seg_idx,    # Segment within utterance
                            'conversation_no': conv_id,
                        })
                        conv_segment_idx += 1
            elif isinstance(tag_data, str) and tag_data and tag_data != '+':
                # Fallback for string format
                segments.append({
                    'text': utt.text.strip() if utt.text else '',
                    'act_tag': tag_data,
                    'damsl_act_tag': tag_data,
                    'caller': speaker_id,
                    'transcript_index': conv_segment_idx,
                    'utterance_index': utt_idx,
                    'segment_index': 0,
                    'conversation_no': conv_id,
                })
                conv_segment_idx += 1

    return segments


def group_by_conversation(segments: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group segments by conversation_no and sort by transcript_index.

    Args:
        segments: List of segment dicts with conversation_no field

    Returns:
        Dict mapping conversation_no (str or int) to list of segments (sorted)
    """
    conversations = defaultdict(list)

    for item in tqdm(segments, desc="Grouping by conversation"):
        conv_id = item['conversation_no']
        conversations[conv_id].append({
            'text': item['text'],
            'act_tag': item['act_tag'],
            'damsl_act_tag': item.get('damsl_act_tag', item['act_tag']),
            'caller': item['caller'],
            'transcript_index': item['transcript_index'],
            'utterance_index': item.get('utterance_index', item['transcript_index']),
            'segment_index': item.get('segment_index', 0),
        })

    # Sort utterances within each conversation by transcript_index
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: (x['transcript_index'], x['utterance_index']))

    return dict(conversations)


def split_conversations(
    conversations: Dict[int, List[Dict]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split conversation IDs into train/val/test sets.

    Args:
        conversations: Dict mapping conversation_no to utterances
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    np.random.seed(seed)

    conv_ids = list(conversations.keys())
    np.random.shuffle(conv_ids)

    n_total = len(conv_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = conv_ids[:n_train]
    val_ids = conv_ids[n_train:n_train + n_val]
    test_ids = conv_ids[n_train + n_val:]

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test conversations")

    return train_ids, val_ids, test_ids


def create_sliding_windows(
    conversations: Dict[int, List[Dict]],
    conv_ids: List[int],
    window_size: int = 5
) -> List[Dict]:
    """
    Create sliding window instances for next DA prediction.

    For each window of `window_size` utterances, the task is to predict
    the DA of the last utterance given the previous utterances.

    Args:
        conversations: Dict mapping conversation_no to utterances
        conv_ids: List of conversation IDs to process
        window_size: Size of sliding window

    Returns:
        List of training instances with history and target
    """
    instances = []

    for conv_id in tqdm(conv_ids, desc="Creating windows"):
        utterances = conversations[conv_id]

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

            # Map tags to 9 categories
            history_tags = [map_to_9_tags(u['act_tag']) for u in history]
            target_tag = map_to_9_tags(target['act_tag'])

            # Skip windows where target or any history tag is None (continuation markers)
            if target_tag is None:
                continue
            if any(tag is None for tag in history_tags):
                continue

            source_tag = history_tags[-1]  # Last history tag for TM

            # Build history text with speaker and tag info
            history_text_parts = []
            for u, tag in zip(history, history_tags):
                speaker = u['caller']  # 'A' or 'B'
                text = u['text'].strip()
                history_text_parts.append(f"{speaker} ({tag}): {text}")

            history_text = "\n".join(history_text_parts)

            instances.append({
                'conversation_id': conv_id,
                'history': history_text,
                'history_tags': history_tags,
                'target_text': target['text'].strip(),
                'target_tag': target_tag,
                'target_speaker': target['caller'],
                'source_tag': source_tag,
                'speaker_change': 1 if history[-1]['caller'] != target['caller'] else 0,
            })

    return instances


def compute_transition_matrix(instances: List[Dict]) -> pd.DataFrame:
    """
    Compute transition matrix from training instances.

    Args:
        instances: List of training instances with source_tag and target_tag

    Returns:
        DataFrame with transition probabilities (rows=source, cols=target)
    """
    # Count transitions
    transitions = defaultdict(lambda: defaultdict(int))

    for inst in instances:
        source = inst['source_tag']
        target = inst['target_tag']
        transitions[source][target] += 1

    # Convert to DataFrame
    tm = pd.DataFrame(index=TAG_NAMES, columns=TAG_NAMES, dtype=float)
    tm = tm.fillna(0.0)

    for source in TAG_NAMES:
        total = sum(transitions[source].values())
        if total > 0:
            for target in TAG_NAMES:
                tm.loc[source, target] = transitions[source][target] / total

    # Add smoothing for zero probabilities
    smoothing = 1e-6
    tm = tm + smoothing
    tm = tm.div(tm.sum(axis=1), axis=0)

    return tm


def verify_distribution(instances: List[Dict], name: str = "Dataset"):
    """Print tag distribution and compare to Tanaka's target."""
    tag_counts = Counter(inst['target_tag'] for inst in instances)
    total = sum(tag_counts.values())

    print(f"\n{name} Distribution ({total} instances):")
    print("-" * 60)
    print(f"{'Tag':<20} {'Count':>10} {'Percent':>10} {'Target %':>10}")
    print("-" * 60)

    target_total = sum(TARGET_DISTRIBUTION.values())

    for tag in TAG_NAMES:
        count = tag_counts.get(tag, 0)
        percent = 100.0 * count / total if total > 0 else 0
        target_percent = 100.0 * TARGET_DISTRIBUTION[tag] / target_total
        print(f"{tag:<20} {count:>10} {percent:>9.1f}% {target_percent:>9.1f}%")

    print("-" * 60)


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

    # Save tag mapping for reference
    with open(os.path.join(output_dir, 'tag_mapping.json'), 'w') as f:
        json.dump(DAMSL_TO_9_TAGS, f, indent=2)

    print(f"\nSaved preprocessed data to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SwDA for NDAP experiments")
    parser.add_argument('--output_dir', type=str, default='./swda_preprocessed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Sliding window size (default: 5, matching Tanaka)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    parser.add_argument('--combine_val_test', action='store_true',
                        help='Combine val and test sets into single test set (for comparability with full_categories)')
    args = parser.parse_args()

    # Load dataset using ConvoKit (at segment level)
    print("Loading SwDA corpus...")
    segments = load_swda_from_convokit()

    print(f"Loaded {len(segments)} DA segments (excluding continuations)")

    # Group by conversation
    conversations = group_by_conversation(segments)
    print(f"Found {len(conversations)} conversations")

    # Split at conversation level
    train_ids, val_ids, test_ids = split_conversations(
        conversations,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )

    # Create sliding windows
    print(f"\nCreating sliding windows (size={args.window_size})...")
    train_instances = create_sliding_windows(conversations, train_ids, args.window_size)
    val_instances = create_sliding_windows(conversations, val_ids, args.window_size)
    test_instances = create_sliding_windows(conversations, test_ids, args.window_size)

    # Combine val and test if requested
    if args.combine_val_test:
        print("\nCombining val and test sets...")
        test_instances = val_instances + test_instances
        val_instances = []
        test_ids = val_ids + test_ids
        val_ids = []
        print(f"Combined: {len(train_instances)} train, {len(test_instances)} test instances")
    else:
        print(f"Created: {len(train_instances)} train, {len(val_instances)} val, {len(test_instances)} test instances")

    # Verify distributions
    verify_distribution(train_instances, "Training")
    verify_distribution(val_instances, "Validation")
    verify_distribution(test_instances, "Test")

    # Compute transition matrix from training data
    print("\nComputing transition matrix from training data...")
    transition_matrix = compute_transition_matrix(train_instances)
    print("\nTransition Matrix:")
    print(transition_matrix.round(3))

    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(TAG_NAMES)

    # Save split info
    split_info = {
        'train_conversation_ids': train_ids,
        'val_conversation_ids': val_ids,
        'test_conversation_ids': test_ids,
        'window_size': args.window_size,
        'seed': args.seed,
        'n_train': len(train_instances),
        'n_val': len(val_instances),
        'n_test': len(test_instances),
        'tag_names': TAG_NAMES,
        'combine_val_test': args.combine_val_test,
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

    print("\nPreprocessing complete!")
    print(f"Total instances: {len(train_instances) + len(val_instances) + len(test_instances)}")


if __name__ == '__main__':
    main()
