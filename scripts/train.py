#!/usr/bin/env python3
"""
Train NDAP models on the OnCoCo dataset with transition-matrix regularization.

This script supports:
- Multiple pretrained encoders (GBERT, EuroBERT, ModernGBERT, GELECTRa)
- BERT and History-augmented architectures
- Transition-matrix regularization with configurable lambda
- 5-fold cross-validation
- WandB logging (optional)

Usage:
    # Single experiment
    python train_oncoco.py --encoder gbert_base --model_type bert --tm_weight 0.5 --fold 0

    # Run all configurations for one fold
    python train_oncoco.py --run_grid --fold 0

    # Run full 5-fold CV for one configuration
    python train_oncoco.py --encoder gbert_base --model_type history --tm_weight 1.0 --all_folds
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_models import (
    NextCategoryPredictor,
    CategoryHistoryPredictor,
    CombinedLoss,
)
from src.data.datasets import OnCoCoDataset

# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "oncoco"
CV_SPLITS_DIR = DATA_DIR / "cv_splits"
RESULTS_DIR = PROJECT_ROOT / "results"

# Encoder configurations
ENCODERS = {
    "gbert_base": "deepset/gbert-base",
    "gbert_large": "deepset/gbert-large",
    "eurobert_210m": "EuroBERT/EuroBERT-210m",
    "eurobert_610m": "EuroBERT/EuroBERT-610m",
    "modern_gbert_134m": "DiscoResearch/modern-german-bert-base",
    "modern_gbert_1b": "DiscoResearch/modern-german-bert-large",
    "gelectra_base": "deepset/gelectra-base",
}

# Hyperparameter grid
TM_WEIGHTS = [0.0, 0.2, 0.5, 1.0, 1.5]
CONTEXT_LENGTHS = [1, 4, 8, 12]
MODEL_TYPES = ["bert", "history"]

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LR = 2e-5
DEFAULT_PATIENCE = 3


def load_cv_split(fold: int, split: str) -> list:
    """Load a CV split file."""
    split_path = CV_SPLITS_DIR / f"fold_{fold}" / f"{split}.json"
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_transition_matrix(train_data: list) -> pd.DataFrame:
    """Compute empirical transition matrix from training data."""
    transitions = {}

    for item in train_data:
        # Extract previous category from conversation history
        history = item.get('conversation_history', '')
        label = item.get('label', '')

        # Parse last category from history
        lines = history.strip().split('\n')
        if lines:
            last_line = lines[-1]
            # Format: "Speaker (Category|Desc): Text"
            if '(' in last_line and '|' in last_line:
                prev_cat = last_line.split('(')[1].split('|')[0].strip()
            else:
                prev_cat = 'UNKNOWN'
        else:
            prev_cat = 'START'

        # Record transition
        if prev_cat not in transitions:
            transitions[prev_cat] = {}
        if label not in transitions[prev_cat]:
            transitions[prev_cat][label] = 0
        transitions[prev_cat][label] += 1

    # Convert to DataFrame and normalize
    all_labels = sorted(set(item['label'] for item in train_data))
    matrix = pd.DataFrame(0.0, index=all_labels, columns=all_labels)

    for src, targets in transitions.items():
        if src in matrix.index:
            for tgt, count in targets.items():
                if tgt in matrix.columns:
                    matrix.loc[src, tgt] = count

    # Normalize rows
    row_sums = matrix.sum(axis=1)
    row_sums = row_sums.replace(0, 1)  # Avoid division by zero
    matrix = matrix.div(row_sums, axis=0)

    # Add smoothing
    epsilon = 1e-8
    matrix = matrix + epsilon
    matrix = matrix.div(matrix.sum(axis=1), axis=0)

    return matrix


def create_model(model_type: str, encoder: str, num_categories: int,
                 max_context_utterances: int = 4):
    """Create model based on type."""
    encoder_path = ENCODERS.get(encoder, encoder)

    if model_type == "bert":
        model = NextCategoryPredictor(
            num_categories=num_categories,
            pretrained_model=encoder_path
        )
    elif model_type == "history":
        model = CategoryHistoryPredictor(
            num_categories=num_categories,
            pretrained_model=encoder_path,
            max_history_length=max_context_utterances
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_epoch(model, dataloader, optimizer, criterion, label_encoder, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        source_categories = batch.get('source_categories', None)

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss, loss_dict = criterion(logits, labels, source_categories, label_encoder)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, label_encoder, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            source_categories = batch.get('source_categories', None)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, _ = criterion(logits, labels, source_categories, label_encoder)

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    from sklearn.metrics import f1_score, accuracy_score

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        'loss': total_loss / num_batches,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
    }


def train_model(args):
    """Train a single model configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_data = load_cv_split(args.fold, 'train')
    val_data = load_cv_split(args.fold, 'val')
    test_data = load_cv_split(args.fold, 'test')

    print(f"Fold {args.fold}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Create label encoder
    all_labels = sorted(set(item['label'] for item in train_data + val_data + test_data))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_categories = len(label_encoder.classes_)
    print(f"Number of categories: {num_categories}")

    # Compute transition matrix
    transition_matrix = compute_transition_matrix(train_data)

    # Create tokenizer
    encoder_path = ENCODERS.get(args.encoder, args.encoder)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    # Create datasets
    train_dataset = OnCoCoDataset(train_data, tokenizer, label_encoder, max_length=512)
    val_dataset = OnCoCoDataset(val_data, tokenizer, label_encoder, max_length=512)
    test_dataset = OnCoCoDataset(test_data, tokenizer, label_encoder, max_length=512)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    model = create_model(
        args.model_type,
        args.encoder,
        num_categories,
        max_context_utterances=args.context_length
    )
    model.to(device)

    # Create loss function
    criterion = CombinedLoss(
        transition_matrix=transition_matrix,
        label_encoder=label_encoder,
        ce_weight=1.0,
        tm_weight=args.tm_weight,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, label_encoder, device)
        val_metrics = evaluate(model, val_loader, criterion, label_encoder, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}")

        # Early stopping
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), RESULTS_DIR / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(RESULTS_DIR / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader, criterion, label_encoder, device)

    print("\nTest Results:")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train NDAP models on OnCoCo")

    # Model configuration
    parser.add_argument('--encoder', type=str, default='gbert_base',
                        choices=list(ENCODERS.keys()),
                        help='Pretrained encoder to use')
    parser.add_argument('--model_type', type=str, default='bert',
                        choices=['bert', 'history'],
                        help='Model architecture type')
    parser.add_argument('--tm_weight', type=float, default=0.5,
                        help='Transition matrix loss weight (lambda)')
    parser.add_argument('--context_length', type=int, default=4,
                        help='Number of context utterances for history model')

    # Training configuration
    parser.add_argument('--fold', type=int, default=0,
                        help='CV fold to train on (0-4)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)

    # Special modes
    parser.add_argument('--all_folds', action='store_true',
                        help='Run all 5 folds')
    parser.add_argument('--run_grid', action='store_true',
                        help='Run full hyperparameter grid')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='ndap-oncoco')

    args = parser.parse_args()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_folds:
        # Run all 5 folds
        results = []
        for fold in range(5):
            args.fold = fold
            metrics = train_model(args)
            metrics['fold'] = fold
            results.append(metrics)

        # Compute mean and std
        macro_f1s = [r['macro_f1'] for r in results]
        print(f"\n5-Fold CV Results:")
        print(f"  Macro-F1: {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")

    elif args.run_grid:
        # Run full grid for one fold
        results = []
        for encoder in ENCODERS:
            for model_type in MODEL_TYPES:
                for tm_weight in TM_WEIGHTS:
                    for context_length in CONTEXT_LENGTHS:
                        args.encoder = encoder
                        args.model_type = model_type
                        args.tm_weight = tm_weight
                        args.context_length = context_length

                        print(f"\n{'='*60}")
                        print(f"Running: {encoder}, {model_type}, tm={tm_weight}, ctx={context_length}")
                        print(f"{'='*60}")

                        metrics = train_model(args)
                        metrics.update({
                            'encoder': encoder,
                            'model_type': model_type,
                            'tm_weight': tm_weight,
                            'context_length': context_length,
                        })
                        results.append(metrics)

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / f'grid_results_fold_{args.fold}.csv', index=False)

    else:
        # Single run
        train_model(args)


if __name__ == '__main__':
    main()
