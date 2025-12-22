#!/usr/bin/env python3
"""
Evaluate trained NDAP models and generate metrics.

This script evaluates model checkpoints and generates:
- Classification metrics (macro-F1, weighted-F1, accuracy)
- Transition-aware metrics (constrained accuracy, cumulative mass metrics)
- Confusion matrices
- Predicted transitions analysis

Usage:
    # Evaluate a single checkpoint
    python evaluate.py --checkpoint results/best_model.pt --fold 0

    # Generate full evaluation report
    python evaluate.py --checkpoint results/best_model.pt --fold 0 --full_report
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_models import (
    NextCategoryPredictor,
    HierarchyAwarePredictorWithCategoryHistory,
)
from src.data.datasets import OnCoCoDataset
from src.training.metrics import (
    evaluate_transition_threshold,
    evaluate_cumulative_mass_metrics,
    save_confusion_matrix_best_f1,
    top_k_accuracy,
)

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


def load_cv_split(fold: int, split: str) -> list:
    """Load a CV split file."""
    split_path = CV_SPLITS_DIR / f"fold_{fold}" / f"{split}.json"
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_transition_matrix(train_data: list) -> pd.DataFrame:
    """Compute empirical transition matrix from training data."""
    transitions = {}

    for item in train_data:
        history = item.get('conversation_history', '')
        label = item.get('label', '')

        lines = history.strip().split('\n')
        if lines:
            last_line = lines[-1]
            if '(' in last_line and '|' in last_line:
                prev_cat = last_line.split('(')[1].split('|')[0].strip()
            else:
                prev_cat = 'UNKNOWN'
        else:
            prev_cat = 'START'

        if prev_cat not in transitions:
            transitions[prev_cat] = {}
        if label not in transitions[prev_cat]:
            transitions[prev_cat][label] = 0
        transitions[prev_cat][label] += 1

    all_labels = sorted(set(item['label'] for item in train_data))
    matrix = pd.DataFrame(0.0, index=all_labels, columns=all_labels)

    for src, targets in transitions.items():
        if src in matrix.index:
            for tgt, count in targets.items():
                if tgt in matrix.columns:
                    matrix.loc[src, tgt] = count

    row_sums = matrix.sum(axis=1)
    row_sums = row_sums.replace(0, 1)
    matrix = matrix.div(row_sums, axis=0)

    epsilon = 1e-8
    matrix = matrix + epsilon
    matrix = matrix.div(matrix.sum(axis=1), axis=0)

    return matrix


def load_model(checkpoint_path: str, model_type: str, encoder: str,
               num_categories: int, context_length: int = 4, device='cpu'):
    """Load a trained model from checkpoint."""
    encoder_path = ENCODERS.get(encoder, encoder)

    if model_type == "bert":
        model = NextCategoryPredictor(
            num_categories=num_categories,
            pretrained_model=encoder_path
        )
    elif model_type == "history":
        model = HierarchyAwarePredictorWithCategoryHistory(
            num_categories=num_categories,
            pretrained_model=encoder_path,
            max_context_utterances=context_length
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_model(model, dataloader, label_encoder, transition_matrix, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_source_cats = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            source_categories = batch.get('source_categories', [None] * len(input_ids))

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
            all_source_cats.extend(source_categories)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.tensor(all_labels)
    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)

    # Basic metrics
    macro_f1 = f1_score(all_labels_array, all_preds_array, average='macro')
    weighted_f1 = f1_score(all_labels_array, all_preds_array, average='weighted')
    accuracy = accuracy_score(all_labels_array, all_preds_array)

    # Top-k accuracy
    top3_acc = top_k_accuracy(all_logits, all_labels_tensor, k=3)
    top5_acc = top_k_accuracy(all_logits, all_labels_tensor, k=5)

    results = {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'num_samples': len(all_labels),
    }

    # Transition-aware metrics
    tm_metrics = evaluate_transition_threshold(
        all_logits, all_labels_tensor, all_source_cats,
        transition_matrix, label_encoder, threshold=0.1
    )
    results.update(tm_metrics)

    # Cumulative mass metrics
    cum_metrics = evaluate_cumulative_mass_metrics(
        all_logits, all_labels_tensor, all_source_cats,
        transition_matrix, label_encoder
    )
    results.update(cum_metrics)

    return results, all_preds_array, all_labels_array


def main():
    parser = argparse.ArgumentParser(description="Evaluate NDAP models")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--encoder', type=str, default='gbert_base',
                        choices=list(ENCODERS.keys()),
                        help='Pretrained encoder used for training')
    parser.add_argument('--model_type', type=str, default='bert',
                        choices=['bert', 'history'],
                        help='Model architecture type')
    parser.add_argument('--context_length', type=int, default=4,
                        help='Context length for history model')
    parser.add_argument('--fold', type=int, default=0,
                        help='CV fold to evaluate (0-4)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--full_report', action='store_true',
                        help='Generate full evaluation report with visualizations')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_data = load_cv_split(args.fold, 'train')
    eval_data = load_cv_split(args.fold, args.split)

    print(f"Fold {args.fold}: {len(train_data)} train, {len(eval_data)} {args.split}")

    # Create label encoder
    all_labels = sorted(set(item['label'] for item in train_data + eval_data))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_categories = len(label_encoder.classes_)
    print(f"Number of categories: {num_categories}")

    # Compute transition matrix
    transition_matrix = compute_transition_matrix(train_data)

    # Create tokenizer and dataset
    encoder_path = ENCODERS.get(args.encoder, args.encoder)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    eval_dataset = OnCoCoDataset(eval_data, tokenizer, label_encoder, max_length=512)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # Load and evaluate model
    model = load_model(
        args.checkpoint, args.model_type, args.encoder,
        num_categories, args.context_length, device
    )

    print(f"\nEvaluating {args.checkpoint}...")
    results, preds, labels = evaluate_model(
        model, eval_loader, label_encoder, transition_matrix, device
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.split} set, fold {args.fold})")
    print(f"{'='*60}")
    print(f"Macro-F1:        {results['macro_f1']:.4f}")
    print(f"Weighted-F1:     {results['weighted_f1']:.4f}")
    print(f"Accuracy:        {results['accuracy']:.4f}")
    print(f"Top-3 Accuracy:  {results['top3_accuracy']:.4f}")
    print(f"Top-5 Accuracy:  {results['top5_accuracy']:.4f}")

    if 'constrained_accuracy' in results:
        print(f"\nTransition-Aware Metrics:")
        print(f"Constrained Accuracy: {results['constrained_accuracy']:.4f}")

    if 'cumulative80_pred_accuracy' in results:
        print(f"\nCumulative Mass Metrics (80% threshold):")
        print(f"Pred in Valid:   {results['cumulative80_pred_accuracy']:.4f}")
        print(f"True Coverage:   {results['cumulative80_true_coverage']:.4f}")

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"eval_{args.model_type}_{args.encoder}_fold{args.fold}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate full report with visualizations
    if args.full_report:
        print("\nGenerating full evaluation report...")

        # Classification report
        report = classification_report(
            labels, preds,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        report_file = output_dir / f"classification_report_fold{args.fold}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Classification report saved to: {report_file}")

        # Confusion matrix
        cm_file = output_dir / f"confusion_matrix_fold{args.fold}.png"
        save_confusion_matrix_best_f1(
            labels, preds, label_encoder,
            results['macro_f1'], str(cm_file)
        )


if __name__ == '__main__':
    main()
