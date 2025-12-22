# Transition-Matrix Regularization for Next Dialogue Act Prediction

Anonymous submission repository containing code and data to reproduce the experimental results.

## Overview

This repository provides implementation of **Next Dialogue Act Prediction (NDAP)** with transition-matrix regularization using KL divergence. The approach leverages empirical dialogue act transition patterns to regularize neural predictions.

### Key Features

- **Transition-Matrix KL Regularization**: Soft constraint that guides predictions toward statistically likely next dialogue acts
- **Multiple Architectures**: BERT-based and History-augmented models with multi-head attention
- **German BERT Encoders**: Support for 7 pretrained German language models
- **5-Fold Cross-Validation**: Conversation-level splits ensuring no data leakage
- **Cross-Dataset Validation**: Evaluation on SWDA and HOPE counselling datasets

## Dataset

This repository includes a **new dataset** of German text-based counselling conversations, released for the first time with this submission:

- **76 complete conversations** with **5,457 utterances**
- Each utterance annotated with one of **60 dialogue act categories** from a five-level hierarchy
- Data from structured counselling role-play sessions conducted by social science students
- Annotations produced by trained annotators and reviewed by domain experts
- All conversations anonymized; no real clients or personal information involved

The annotation scheme follows the OnCoCo taxonomy, which organizes dialogue acts into a hierarchical structure capturing conversational function and pragmatic counselling flow. This is the first publicly available corpus of complete counselling conversations annotated with this fine-grained scheme, enabling sequential modeling of dialogue flow.

## Repository Structure

```
acl_submission/
├── data/
│   ├── oncoco/                    # German counselling dataset (NEW)
│   │   ├── conversations.json     # 76 complete conversations (5,457 utterances)
│   │   ├── category_descriptions.csv  # 60 dialogue act categories
│   │   └── cv_splits/             # 5-fold CV splits
│   ├── swda/                      # SWDA preprocessing scripts
│   └── hope/                      # HOPE preprocessing scripts
├── src/
│   ├── models/                    # Model architectures
│   │   └── bert_models.py         # BERT, History, TM loss
│   ├── data/                      # Dataset classes
│   │   ├── datasets.py            # OnCoCo dataset
│   │   ├── swda_dataset.py        # SWDA dataset
│   │   └── data_utils.py          # Data utilities
│   ├── training/                  # Training utilities
│   │   ├── train_utils.py         # Training loop
│   │   └── metrics.py             # Evaluation metrics
│   └── utils/                     # General utilities
├── scripts/
│   ├── train.py                   # Main training script
│   └── evaluate.py                # Evaluation script
├── results/                       # Experiment results
│   ├── wandb_export_*.csv         # WandB logs for figure generation
│   └── llm_baseline_results.json  # LLM baseline results
├── requirements.txt
└── setup.py
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.30
- CUDA (optional, for GPU acceleration)

## Quick Start

### Training a Single Model

```bash
# Train BERT model with TM regularization (lambda=0.5)
python scripts/train.py \
    --encoder gbert_base \
    --model_type bert \
    --tm_weight 0.5 \
    --fold 0

# Train History model with longer context
python scripts/train.py \
    --encoder gbert_base \
    --model_type history \
    --tm_weight 1.0 \
    --context_length 8 \
    --fold 0
```

### Running 5-Fold Cross-Validation

```bash
python scripts/train.py \
    --encoder gbert_base \
    --model_type history \
    --tm_weight 0.5 \
    --all_folds
```

### Running Full Hyperparameter Grid

```bash
# Runs all 280 configurations (7 encoders × 2 architectures × 5 TM weights × 4 context lengths)
python scripts/train.py --run_grid --fold 0
```

## Supported Encoders

| Key | Model | Size |
|-----|-------|------|
| `gbert_base` | deepset/gbert-base | 110M |
| `gbert_large` | deepset/gbert-large | 340M |
| `eurobert_210m` | EuroBERT/EuroBERT-210m | 210M |
| `eurobert_610m` | EuroBERT/EuroBERT-610m | 610M |
| `modern_gbert_134m` | DiscoResearch/modern-german-bert-base | 134M |
| `modern_gbert_1b` | DiscoResearch/modern-german-bert-large | 1B |
| `gelectra_base` | deepset/gelectra-base | 110M |

## Hyperparameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `tm_weight` | 0.0, 0.2, 0.5, 1.0, 1.5 | Transition matrix loss weight (λ) |
| `context_length` | 1, 4, 8, 12 | Number of context utterances |
| `model_type` | bert, history | Architecture type |
| `batch_size` | 64 | Training batch size |
| `lr` | 2e-5 | Learning rate |
| `epochs` | 10 | Maximum training epochs |
| `patience` | 3 | Early stopping patience |

## Cross-Dataset Evaluation

### SWDA (Switchboard Dialogue Act Corpus)

```bash
# Download and preprocess SWDA
cd data/swda
# Follow instructions in README.md

# Train on SWDA
python scripts/train_swda.py --encoder gbert_base --fold 0
```

### HOPE (Counselling Dataset)

```bash
# Download and preprocess HOPE
cd data/hope
# Follow instructions in README.md

# Train on HOPE
python scripts/train_hope.py --encoder gbert_base --fold 0
```

## LLM Baseline

The LLM baseline script evaluates GPT models on next dialogue act prediction.

**Note**: This requires API keys to be set as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
# or for local models:
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

```bash
python scripts/llm_baseline.py --model gpt-4o-mini --fold 0
```

## Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py \
    --checkpoint results/best_model.pt \
    --encoder gbert_base \
    --model_type bert \
    --fold 0

# Generate full evaluation report with visualizations
python scripts/evaluate.py \
    --checkpoint results/best_model.pt \
    --fold 0 \
    --full_report
```

## WandB Logging

Enable WandB logging for experiment tracking:

```bash
python scripts/train.py \
    --encoder gbert_base \
    --model_type history \
    --tm_weight 0.5 \
    --fold 0 \
    --wandb \
    --wandb_project ndap-oncoco
```

## Results

Pre-computed experiment results are available in `results/`:

- `wandb_export_full_categories.csv`: Full experiment logs for reproducing figures
- `llm_baseline_results.json`: LLM baseline evaluation results

## License

This code is released for academic research purposes. See LICENSE for details.

## Citation

Citation information will be added after the review process.
