"""
LLM Baseline for Next Dialogue Act Prediction (NDAP) on OnCoCo Dataset

Uses GPT-5-mini with OpenAI Batch API for cost-effective evaluation.
Returns top-3 predictions with confidence scores for comparison with BERT baselines.

Usage:
    # Submit batches for all folds
    python llm_baseline.py --submit

    # Check batch status
    python llm_baseline.py --status

    # Download and evaluate results
    python llm_baseline.py --evaluate
"""

import os
import sys
import json
import csv
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

CV_SPLITS_DIR = PROJECT_ROOT / "NextCategoryPrediction" / "data" / "cv_splits_conversation_based"
CATEGORY_DESCRIPTIONS_PATH = PROJECT_ROOT / "dataprep" / "category_descriptions.csv"
RESULTS_DIR = Path(__file__).parent / "results"
MODEL = "gpt-5-mini"
MAX_HISTORY_TURNS = 12
NUM_FOLDS = 5

# IONOS AI Model Hub configuration (OpenAI-compatible API)
IONOS_BASE_URL = "https://openai.inference.de-txl.ionos.com/v1"
IONOS_MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
}


def get_openai_client(model: str):
    """Get OpenAI client configured for the specified model.

    For IONOS models (gpt-oss-*), uses IONOS_API_TOKEN env var.
    For OpenAI models, uses OPENAI_API_KEY env var.
    """
    import openai

    if model.startswith("gpt-oss"):
        # Use IONOS AI Model Hub
        api_key = os.environ.get("IONOS_API_TOKEN")
        if not api_key:
            raise ValueError("IONOS_API_TOKEN environment variable not set")
        return openai.OpenAI(
            api_key=api_key,
            base_url=IONOS_BASE_URL,
        )
    else:
        # Use OpenAI API
        return openai.OpenAI()


# =============================================================================
# Data Loading
# =============================================================================

def load_json(path: Path) -> List[dict]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_all_labels_from_cv_splits() -> List[str]:
    """Extract all unique labels from CV splits."""
    labels = set()
    for fold in range(NUM_FOLDS):
        fold_dir = CV_SPLITS_DIR / f"fold_{fold}"
        for split in ['train', 'val', 'test']:
            split_path = fold_dir / f"{split}.json"
            if split_path.exists():
                data = load_json(split_path)
                labels.update(x['label'] for x in data)
    return sorted(labels)


def normalize_label(label: str) -> str:
    """Extract code part before '|' for matching."""
    return label.split('|')[0].strip()


def fuzzy_match_label(pred_label: str, valid_labels_set: set) -> Tuple[str, bool]:
    """Try to match predicted label to valid labels.

    Handles common LLM issues like missing trailing '*' or extra characters.

    Returns:
        Tuple of (matched_label, is_valid)
    """
    norm = normalize_label(pred_label)

    # Exact match
    if norm in valid_labels_set:
        return norm, True

    # Try adding trailing * if missing (common LLM error)
    if not norm.endswith('*') and not norm.endswith('|'):
        with_star = norm + '*'
        if with_star in valid_labels_set:
            return with_star, True

    # Try removing trailing characters
    for suffix in ['-', ' ', '*']:
        if norm.endswith(suffix):
            trimmed = norm.rstrip(suffix)
            if trimmed in valid_labels_set:
                return trimmed, True
            if trimmed + '*' in valid_labels_set:
                return trimmed + '*', True

    # No match found
    return norm, False


def load_category_descriptions() -> Dict[str, str]:
    """Load category descriptions from CSV, keyed by normalized code."""
    descriptions = {}

    if not CATEGORY_DESCRIPTIONS_PATH.exists():
        print(f"Warning: Category descriptions not found at {CATEGORY_DESCRIPTIONS_PATH}")
        return descriptions

    # Try different encodings
    for encoding in ['utf-8', 'latin1', 'cp1252']:
        try:
            with open(CATEGORY_DESCRIPTIONS_PATH, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle different column name formats
                    kategorie = row.get('Kategorie', '')
                    beschreibung = row.get('Beschreibung', '')

                    if kategorie:
                        # Normalize the category code
                        code = normalize_label(kategorie)
                        descriptions[code] = beschreibung or kategorie
            break
        except UnicodeDecodeError:
            continue

    return descriptions


def load_fold_data(fold: int, include_val: bool = True) -> Tuple[List[dict], List[str]]:
    """Load evaluation data for a fold and return with all valid labels.

    Args:
        fold: Fold number (0-4)
        include_val: If True, combine val + test data. If False, only test data.

    Returns:
        Tuple of (data, all_labels)
    """
    fold_dir = CV_SPLITS_DIR / f"fold_{fold}"

    # Load test data
    test_path = fold_dir / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")
    eval_data = load_json(test_path)

    # Optionally add val data
    if include_val:
        val_path = fold_dir / "val.json"
        if val_path.exists():
            val_data = load_json(val_path)
            eval_data = val_data + eval_data  # val first, then test

    all_labels = get_all_labels_from_cv_splits()

    return eval_data, all_labels


# =============================================================================
# Prompt Construction
# =============================================================================

def truncate_history(history: str, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Truncate conversation history to last N turns."""
    turns = history.strip().split('\n')
    if len(turns) <= max_turns:
        return history
    return '\n'.join(turns[-max_turns:])


def build_ndap_prompt(
    conversation_history: str,
    valid_labels: List[str],
    category_descriptions: Dict[str, str],
) -> str:
    """
    Build prompt for NDAP with:
    - Task description
    - All category codes with descriptions
    - Truncated conversation history
    - Request for top-3 predictions with confidence scores
    """
    # Truncate history
    truncated_history = truncate_history(conversation_history, MAX_HISTORY_TURNS)

    # Build category list with descriptions
    category_lines = []
    for label in valid_labels:
        code = normalize_label(label)
        description = category_descriptions.get(code, label)
        # Use short format for prompt efficiency
        if description and description != label:
            category_lines.append(f"- {label}: {description}")
        else:
            category_lines.append(f"- {label}")

    categories_text = '\n'.join(category_lines)

    prompt = f"""You are an expert in dialogue act classification for German online counseling conversations (OnCoCo taxonomy).

## Task
Given a conversation history between a counselor (B = Berater) and client (K = Klient), predict the dialogue act category of the NEXT utterance. The conversation history includes gold dialogue act labels for each turn in parentheses.

## Available Categories ({len(valid_labels)} total)
{categories_text}

## Conversation History
{truncated_history}

## Instructions
1. Analyze the conversation flow and the last speaker's dialogue act
2. Consider typical counseling conversation patterns (opening → problem exploration → intervention → closing)
3. Predict which dialogue act category the NEXT utterance will belong to
4. Provide your top 3 most likely predictions with confidence scores (must sum to <= 1.0)

## Output Format
Respond ONLY with a JSON object in this exact format:
{{
  "predictions": [
    {{"category": "CATEGORY_CODE", "confidence": 0.5}},
    {{"category": "CATEGORY_CODE", "confidence": 0.3}},
    {{"category": "CATEGORY_CODE", "confidence": 0.15}}
  ],
  "reasoning": "Brief explanation of your prediction logic"
}}

Important: Use the exact category codes from the list above (e.g., "B-FA-*-*-*-*" not just "B-FA")."""

    return prompt


# =============================================================================
# Batch API Integration
# =============================================================================

def build_json_schema() -> Dict[str, Any]:
    """Build JSON schema for structured output."""
    return {
        "format": {
            "type": "json_schema",
            "name": "ndap_prediction",
            "schema": {
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["category", "confidence"],
                            "additionalProperties": False
                        },
                        "minItems": 1,
                        "maxItems": 3
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["predictions", "reasoning"],
                "additionalProperties": False
            },
            "strict": True
        }
    }


def build_batch_requests(
    samples: List[dict],
    valid_labels: List[str],
    category_descriptions: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Build JSONL batch requests for GPT-5-mini."""
    requests = []

    system_msg = (
        "You are an expert dialogue act classifier for German online counseling conversations. "
        "Always respond with valid JSON in the specified format."
    )

    for i, sample in enumerate(samples):
        prompt = build_ndap_prompt(
            sample["conversation_history"],
            valid_labels,
            category_descriptions
        )

        # GPT-5 models use responses API
        body = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "text": build_json_schema(),
            "max_output_tokens": 4000,  # GPT-5 is a reasoning model, needs more tokens
        }

        requests.append({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        })

    return requests


def submit_batch(requests: List[dict], fold: int) -> Optional[str]:
    """Submit batch to OpenAI and return batch_id."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None

    # Write requests to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        batch_file_path = f.name

    try:
        # Upload batch file
        print(f"Uploading batch file for fold {fold} with {len(requests)} requests...")
        with open(batch_file_path, 'rb') as f:
            batch_input_file = client.files.create(file=f, purpose="batch")

        # Create batch
        print("Creating batch job...")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h"
        )

        print(f"✅ Batch created for fold {fold}")
        print(f"   Batch ID: {batch.id}")
        print(f"   Status: {batch.status}")

        return batch.id

    except Exception as e:
        print(f"❌ Error submitting batch: {e}")
        return None
    finally:
        try:
            os.unlink(batch_file_path)
        except:
            pass


def check_batch_status(batch_id: str) -> Optional[Dict[str, Any]]:
    """Check status of a batch job."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        batch = client.batches.retrieve(batch_id)

        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": getattr(batch, 'completed_at', None),
            "request_counts": getattr(batch, 'request_counts', None),
            "output_file_id": getattr(batch, 'output_file_id', None),
            "error_file_id": getattr(batch, 'error_file_id', None),
        }
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return None


def download_batch_results(batch_id: str, output_path: Path) -> bool:
    """Download results of a completed batch."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        batch = client.batches.retrieve(batch_id)

        if batch.status != "completed":
            print(f"Batch not completed. Status: {batch.status}")
            return False

        if not batch.output_file_id:
            print("No output file available")
            return False

        # Download output
        file_response = client.files.content(batch.output_file_id)
        data = file_response.read() if hasattr(file_response, 'read') else file_response.content

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(data)

        print(f"✅ Results saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading results: {e}")
        return False


# =============================================================================
# Metadata Management
# =============================================================================

def get_metadata_path() -> Path:
    """Get path to batch metadata file."""
    return RESULTS_DIR / "llm_baseline_batch_metadata.json"


def load_metadata() -> Dict[str, Any]:
    """Load batch metadata."""
    path = get_metadata_path()
    if path.exists():
        return load_json(path)
    return {"batches": {}, "model": MODEL, "max_history_turns": MAX_HISTORY_TURNS}


def save_metadata(metadata: Dict[str, Any]):
    """Save batch metadata."""
    save_json(metadata, get_metadata_path())


# =============================================================================
# Main Commands
# =============================================================================

def cmd_submit(folds: Optional[List[int]] = None):
    """Submit batches for specified folds (default: all)."""
    if folds is None:
        folds = list(range(NUM_FOLDS))

    print(f"Loading category descriptions...")
    descriptions = load_category_descriptions()
    print(f"  Loaded {len(descriptions)} descriptions")

    print(f"Extracting valid labels from CV splits...")
    all_labels = get_all_labels_from_cv_splits()
    print(f"  Found {len(all_labels)} unique labels")

    metadata = load_metadata()

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"Processing fold {fold}")
        print(f"{'='*60}")

        # Check if already submitted
        fold_key = f"fold_{fold}"
        if fold_key in metadata["batches"]:
            existing = metadata["batches"][fold_key]
            print(f"  Batch already exists: {existing['batch_id']}")
            print(f"  Submitted at: {existing['submitted_at']}")
            continue

        # Load data
        test_data, _ = load_fold_data(fold)
        print(f"  Loaded {len(test_data)} test samples")

        # Build requests
        print(f"  Building batch requests...")
        requests = build_batch_requests(test_data, all_labels, descriptions)

        # Submit batch
        batch_id = submit_batch(requests, fold)

        if batch_id:
            metadata["batches"][fold_key] = {
                "batch_id": batch_id,
                "num_samples": len(test_data),
                "submitted_at": datetime.now().isoformat(),
            }
            save_metadata(metadata)

    print(f"\n{'='*60}")
    print("Submission complete. Run with --status to check progress.")


def cmd_status():
    """Check status of all submitted batches."""
    metadata = load_metadata()

    if not metadata["batches"]:
        print("No batches submitted yet. Run with --submit first.")
        return

    print(f"{'='*60}")
    print("Batch Status")
    print(f"{'='*60}")

    all_complete = True
    for fold_key, info in sorted(metadata["batches"].items()):
        batch_id = info["batch_id"]
        status = check_batch_status(batch_id)

        if status:
            status_str = status["status"]
            counts = status.get("request_counts", {})

            if status_str != "completed":
                all_complete = False

            print(f"\n{fold_key}:")
            print(f"  Batch ID: {batch_id}")
            print(f"  Status: {status_str}")
            if counts:
                print(f"  Requests: {counts}")
        else:
            print(f"\n{fold_key}: Unable to retrieve status")
            all_complete = False

    if all_complete:
        print(f"\n✅ All batches complete! Run with --evaluate to process results.")


def cmd_evaluate():
    """Download results and compute metrics."""
    metadata = load_metadata()

    if not metadata["batches"]:
        print("No batches submitted yet.")
        return

    print(f"Loading category descriptions...")
    descriptions = load_category_descriptions()
    all_labels = get_all_labels_from_cv_splits()

    results = {
        "model": MODEL,
        "max_history_turns": MAX_HISTORY_TURNS,
        "evaluated_at": datetime.now().isoformat(),
        "per_fold": {},
    }

    for fold in range(NUM_FOLDS):
        fold_key = f"fold_{fold}"

        if fold_key not in metadata["batches"]:
            print(f"Fold {fold} not submitted yet, skipping...")
            continue

        batch_id = metadata["batches"][fold_key]["batch_id"]
        output_path = RESULTS_DIR / f"llm_baseline_fold_{fold}_raw.jsonl"

        # Download if not exists
        if not output_path.exists():
            print(f"\nDownloading results for fold {fold}...")
            if not download_batch_results(batch_id, output_path):
                continue

        # Process results
        print(f"\nProcessing fold {fold}...")
        fold_results = process_fold_results(fold, output_path, all_labels)

        if fold_results:
            results["per_fold"][fold_key] = fold_results

    # Aggregate results
    if results["per_fold"]:
        results["aggregated"] = aggregate_results(results["per_fold"])

        # Save results
        results_path = RESULTS_DIR / "llm_baseline_results.json"
        save_json(results, results_path)
        print(f"\n✅ Results saved to {results_path}")

        # Print summary
        print_results_summary(results)


def extract_source_category_from_history(conversation_history: str) -> Optional[str]:
    """Extract the source category (last speaker's category) from conversation history."""
    if not conversation_history:
        return None

    # Split by newlines to get individual turns
    turns = conversation_history.strip().split('\n')
    if not turns:
        return None

    # Get the last turn that has a category
    for turn in reversed(turns):
        if '(' in turn and ')' in turn:
            try:
                start = turn.find('(') + 1
                end = turn.find(')', start)
                if start > 0 and end > start:
                    category_part = turn[start:end]
                    # Split by | and take the first part (the category code)
                    if '|' in category_part:
                        return category_part.split('|')[0].strip()
                    return category_part.strip()
            except:
                continue
    return None


def load_transition_matrix(fold: int = 0) -> Optional[Any]:
    """Load the transition matrix for Cum70 calculation."""
    import pandas as pd
    # Use fold-specific transition matrix
    tm_path = CV_SPLITS_DIR / f"fold_{fold}" / "transition_matrix.csv"
    if tm_path.exists():
        return pd.read_csv(tm_path, index_col=0)
    # Fallback to global transition matrix
    tm_path_global = PROJECT_ROOT / "dataprep" / "dataset_files" / "analysis" / "transition_matrix.csv"
    if tm_path_global.exists():
        return pd.read_csv(tm_path_global, index_col=0)
    return None


def get_valid_categories_cumulative(transition_probs, cumulative_threshold: float = 0.7) -> set:
    """Get valid target categories based on cumulative probability mass."""
    import numpy as np
    import pandas as pd

    if isinstance(transition_probs, pd.Series):
        sorted_indices = np.argsort(transition_probs.values)[::-1]
        sorted_probs = transition_probs.values[sorted_indices]
        category_names = transition_probs.index[sorted_indices]
    else:
        sorted_indices = np.argsort(transition_probs)[::-1]
        sorted_probs = transition_probs[sorted_indices]
        category_names = sorted_indices

    cumsum = np.cumsum(sorted_probs)
    n_valid = np.searchsorted(cumsum, cumulative_threshold) + 1
    n_valid = min(n_valid, len(sorted_probs))

    return set(category_names[:n_valid])


def process_fold_results(fold: int, raw_path: Path, all_labels: List[str]) -> Optional[Dict]:
    """Process raw batch results for a fold.

    Computes:
    - macro_f1: Macro-averaged F1 score
    - weighted_f1: Weighted F1 score
    - top3_accuracy: Accuracy if correct label is in top-3 predictions
    - cum70: Prediction in cumulative 70% mass of transition matrix
    - invalid_rate: Fraction of predictions with invalid labels
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    import pandas as pd

    # Load test data for ground truth
    test_data, _ = load_fold_data(fold)
    ground_truth = [normalize_label(x["label"]) for x in test_data]

    # Load transition matrix for Cum70
    transition_matrix = load_transition_matrix(fold)

    # Build set of valid normalized labels
    valid_labels_set = set(normalize_label(l) for l in all_labels)

    # Parse raw results
    all_predictions = []  # (custom_id, top1, top1_conf, top3)

    with open(raw_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = int(result.get("custom_id", -1))

            # Extract prediction from response
            pred = extract_prediction_from_response(result)

            if pred and 0 <= custom_id < len(test_data):
                all_predictions.append({
                    "idx": custom_id,
                    "top1": pred["top1"],
                    "top1_conf": pred["top1_conf"],
                    "top3": pred["top3"],  # List of (category, confidence) tuples
                })

    if not all_predictions:
        print(f"  No valid predictions extracted")
        return None

    # Sort by custom_id
    all_predictions.sort(key=lambda x: x["idx"])

    # Apply fuzzy matching to all predictions
    num_invalid = 0
    for p in all_predictions:
        matched, is_valid = fuzzy_match_label(p["top1"], valid_labels_set)
        p["top1_matched"] = matched
        p["top1_valid"] = is_valid
        if not is_valid:
            num_invalid += 1

    print(f"  Parsed {len(all_predictions)} predictions, expected {len(test_data)}")

    # Build prediction lookup by index
    pred_by_idx = {p["idx"]: p for p in all_predictions}

    # Compute metrics - treat invalid/missing predictions as wrong (not filtered out)
    # This ensures fair comparison with BERT models that must classify all samples
    INVALID_PLACEHOLDER = "__INVALID_PREDICTION__"

    y_true = []
    y_pred_labels = []
    valid_count = 0
    invalid_count = 0
    missing_count = 0

    for idx in range(len(test_data)):
        y_true.append(ground_truth[idx])
        if idx not in pred_by_idx:
            # No prediction for this sample (API error or missing)
            y_pred_labels.append(INVALID_PLACEHOLDER)
            missing_count += 1
            invalid_count += 1
        elif not pred_by_idx[idx].get("top1_valid"):
            # Prediction couldn't be fuzzy-matched to valid label
            y_pred_labels.append(INVALID_PLACEHOLDER)
            invalid_count += 1
        else:
            # Valid prediction
            y_pred_labels.append(pred_by_idx[idx]["top1_matched"])
            valid_count += 1

    invalid_rate = invalid_count / len(test_data)
    print(f"  Invalid: {invalid_rate:.2%} ({invalid_count}/{len(test_data)}, missing: {missing_count})")

    if valid_count == 0:
        print(f"  No valid predictions!")
        return None

    # Compute metrics
    try:
        # For F1 calculation, invalid predictions will always be wrong
        all_possible_labels = [normalize_label(l) for l in all_labels] + [INVALID_PLACEHOLDER]
        le = LabelEncoder()
        le.fit(all_possible_labels)

        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred_labels)

        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)

        # Top-3 accuracy - invalid/missing predictions count as wrong
        top3_correct = 0
        for idx in range(len(test_data)):
            if idx in pred_by_idx and pred_by_idx[idx].get("top1_valid"):
                p = pred_by_idx[idx]
                true_label = ground_truth[idx]
                # Apply fuzzy matching to top3
                top3_cats = [fuzzy_match_label(cat, valid_labels_set)[0] for cat, _ in p["top3"]]
                if true_label in top3_cats:
                    top3_correct += 1
            # else: invalid/missing prediction, counts as wrong
        top3_accuracy = top3_correct / len(test_data)

        # Cum70 - invalid/missing predictions count as wrong
        cum70_count = 0
        cum70_valid_count = 0

        if transition_matrix is not None:
            tm_rows_normalized = {normalize_label(r): r for r in transition_matrix.index}

            for idx in range(len(test_data)):
                sample = test_data[idx]
                source_cat = extract_source_category_from_history(sample["conversation_history"])

                if source_cat:
                    source_norm = normalize_label(source_cat)
                    if source_norm in tm_rows_normalized:
                        row_name = tm_rows_normalized[source_norm]
                        transitions = transition_matrix.loc[row_name]
                        transitions_normalized = pd.Series(
                            transitions.values,
                            index=[normalize_label(c) for c in transitions.index]
                        )
                        valid_cats = get_valid_categories_cumulative(transitions_normalized, 0.7)
                        cum70_count += 1
                        # Only count as valid if prediction exists, is valid, AND in valid categories
                        if idx in pred_by_idx and pred_by_idx[idx].get("top1_valid"):
                            if pred_by_idx[idx]["top1_matched"] in valid_cats:
                                cum70_valid_count += 1
                        # else: invalid/missing = wrong

        cum70 = cum70_valid_count / cum70_count if cum70_count > 0 else 0.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "top3_accuracy": top3_accuracy,
            "cum70": cum70,
            "invalid_rate": invalid_rate,
            "num_valid": valid_count,
            "num_total": len(test_data),
        }

    except Exception as e:
        print(f"  Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_prediction_from_response(result: dict) -> Optional[Dict]:
    """Extract prediction from batch API response.

    Returns dict with:
        - top1: top prediction category
        - top1_conf: confidence for top1
        - top3: list of (category, confidence) tuples
    """
    try:
        body = result.get("response", {}).get("body", {})

        # Handle responses API format
        output = body.get("output", [])

        for msg in output:
            content = msg.get("content", [])
            for seg in content:
                if seg.get("type") == "output_json":
                    parsed = seg.get("json", {})
                    predictions = parsed.get("predictions", [])

                    if predictions:
                        top1 = predictions[0].get("category", "")
                        top1_conf = predictions[0].get("confidence", 0.0)
                        top3 = [(p.get("category", ""), p.get("confidence", 0.0))
                                for p in predictions[:3]]
                        return {"top1": top1, "top1_conf": top1_conf, "top3": top3}

        # Fallback: try to parse text output
        for msg in output:
            content = msg.get("content", [])
            for seg in content:
                if seg.get("type") == "output_text":
                    text = seg.get("text", "")
                    try:
                        parsed = json.loads(text)
                        predictions = parsed.get("predictions", [])
                        if predictions:
                            top1 = predictions[0].get("category", "")
                            top1_conf = predictions[0].get("confidence", 0.0)
                            top3 = [(p.get("category", ""), p.get("confidence", 0.0))
                                    for p in predictions[:3]]
                            return {"top1": top1, "top1_conf": top1_conf, "top3": top3}
                    except:
                        pass

    except Exception as e:
        pass

    return None


def aggregate_results(per_fold: Dict) -> Dict:
    """Aggregate metrics across folds."""
    import numpy as np

    metrics = ["accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "cum70", "invalid_rate"]
    aggregated = {}

    for metric in metrics:
        values = [fold_results[metric] for fold_results in per_fold.values() if metric in fold_results]
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return aggregated


def print_results_summary(results: Dict):
    """Print formatted results summary."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Model: {results['model']}")
    print(f"History window: {results['max_history_turns']} turns")

    if "aggregated" in results:
        print(f"\nAggregated ({len(results['per_fold'])} folds):")
        # Print in paper order: Macro-F1, W-F1, Top-3, Cum70
        metric_order = ["macro_f1", "weighted_f1", "top3_accuracy", "cum70", "invalid_rate"]
        metric_names = {
            "macro_f1": "Macro-F1",
            "weighted_f1": "W-F1",
            "top3_accuracy": "Top-3",
            "cum70": "Cum70",
            "invalid_rate": "Invalid Rate",
        }
        for metric in metric_order:
            if metric in results["aggregated"]:
                values = results["aggregated"][metric]
                name = metric_names.get(metric, metric)
                print(f"  {name}: {values['mean']:.3f} ± {values['std']:.3f}")

    print(f"\nPer-fold results:")
    for fold_key, fold_results in sorted(results["per_fold"].items()):
        print(f"  {fold_key}: macro_f1={fold_results['macro_f1']:.3f}, "
              f"w_f1={fold_results['weighted_f1']:.3f}, "
              f"top3={fold_results['top3_accuracy']:.3f}, "
              f"cum70={fold_results['cum70']:.3f}")


# =============================================================================
# CLI
# =============================================================================

def cmd_test(model: str, limit: int, fold: int = 0):
    """Run synchronous test with limited samples (no batch API)."""
    print(f"Loading category descriptions...")
    descriptions = load_category_descriptions()
    print(f"  Loaded {len(descriptions)} descriptions")

    print(f"Extracting valid labels from CV splits...")
    all_labels = get_all_labels_from_cv_splits()
    print(f"  Found {len(all_labels)} unique labels")

    # Load data
    test_data, _ = load_fold_data(fold)
    test_data = test_data[:limit]
    print(f"  Testing on {len(test_data)} samples from fold {fold}")

    # Build valid labels set
    valid_labels_set = set(normalize_label(l) for l in all_labels)

    # Get client and resolve model name
    client = get_openai_client(model)
    api_model = IONOS_MODELS.get(model, model)  # Map to API model name if needed
    print(f"  Using model: {api_model} (via {'IONOS' if model.startswith('gpt-oss') else 'OpenAI'})")

    predictions = []
    ground_truth = []

    for i, sample in enumerate(test_data):
        prompt = build_ndap_prompt(
            sample["conversation_history"],
            all_labels,
            descriptions
        )

        try:
            # Build request kwargs
            request_kwargs = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": "You are an expert in dialogue act classification for German online counseling."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
            }

            # Add response_format only for models that support it
            # IONOS gpt-oss models may not support json_object mode
            if not model.startswith("gpt-oss"):
                request_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**request_kwargs)

            content = response.choices[0].message.content

            # Try to extract JSON from response (handle markdown code blocks, etc.)
            parsed = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in markdown code block
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                    except:
                        pass
                # Try to find raw JSON object
                if not parsed:
                    json_match = re.search(r'\{[^{}]*"predictions"[^{}]*\[.*?\][^{}]*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(0))
                        except:
                            pass

            if not parsed:
                print(f"  [{i+1}/{len(test_data)}] Could not parse JSON from response")
                continue

            preds = parsed.get("predictions", [])

            if preds:
                top1 = preds[0].get("category", "")
                top1_conf = preds[0].get("confidence", 0.0)
                top3 = [(p.get("category", ""), p.get("confidence", 0.0)) for p in preds[:3]]

                # Fuzzy match
                matched, is_valid = fuzzy_match_label(top1, valid_labels_set)

                predictions.append({
                    "top1": top1,
                    "top1_matched": matched,
                    "top1_valid": is_valid,
                    "top1_conf": top1_conf,
                    "top3": top3,
                })
                ground_truth.append(normalize_label(sample["label"]))

                status = "✓" if matched == ground_truth[-1] else "✗"
                print(f"  [{i+1}/{len(test_data)}] {status} pred={matched[:30]:30s} truth={ground_truth[-1][:30]}")
            else:
                print(f"  [{i+1}/{len(test_data)}] No predictions in response")

        except Exception as e:
            print(f"  [{i+1}/{len(test_data)}] Error: {e}")

    # Compute metrics
    if predictions:
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import LabelEncoder

        valid_preds = [p for p in predictions if p["top1_valid"]]
        valid_truth = [ground_truth[i] for i, p in enumerate(predictions) if p["top1_valid"]]

        if valid_preds:
            le = LabelEncoder()
            le.fit([normalize_label(l) for l in all_labels])

            y_pred = le.transform([p["top1_matched"] for p in valid_preds])
            y_true = le.transform(valid_truth)

            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # Top-3 accuracy
            top3_correct = 0
            for p, truth in zip(valid_preds, valid_truth):
                top3_cats = [fuzzy_match_label(c, valid_labels_set)[0] for c, _ in p["top3"]]
                if truth in top3_cats:
                    top3_correct += 1
            top3_acc = top3_correct / len(valid_preds)

            print(f"\n{'='*60}")
            print(f"Results ({len(valid_preds)}/{len(predictions)} valid predictions)")
            print(f"{'='*60}")
            print(f"  Accuracy:    {accuracy:.3f}")
            print(f"  Macro-F1:    {macro_f1:.3f}")
            print(f"  Weighted-F1: {weighted_f1:.3f}")
            print(f"  Top-3 Acc:   {top3_acc:.3f}")
            print(f"  Invalid:     {len(predictions) - len(valid_preds)}/{len(predictions)}")


async def process_single_sample_async(client, api_model, model, prompt, idx, total, valid_labels_set):
    """Process a single sample asynchronously.

    Returns a dict with:
        - idx: sample index
        - raw_response: the raw response content from the model (for recalculation)
        - top1, top1_matched, top1_valid, top1_conf, top3: parsed prediction data
        - error: error message if failed
    """
    import re

    try:
        request_kwargs = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": "You are an expert in dialogue act classification for German online counseling."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
        }

        if not model.startswith("gpt-oss"):
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content

        # Parse JSON response
        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                except:
                    pass
            if not parsed:
                json_match = re.search(r'\{[^{}]*"predictions"[^{}]*\[.*?\][^{}]*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except:
                        pass

        if not parsed:
            return {"idx": idx, "raw_response": content, "error": "parse_error"}

        preds = parsed.get("predictions", [])
        if preds:
            top1 = preds[0].get("category", "")
            top1_conf = preds[0].get("confidence", 0.0)
            top3 = [(p.get("category", ""), p.get("confidence", 0.0)) for p in preds[:3]]
            matched, is_valid = fuzzy_match_label(top1, valid_labels_set)

            return {
                "idx": idx,
                "raw_response": content,
                "top1": top1,
                "top1_matched": matched,
                "top1_valid": is_valid,
                "top1_conf": top1_conf,
                "top3": top3,
            }
        else:
            return {"idx": idx, "raw_response": content, "error": "no_predictions"}

    except Exception as e:
        return {"idx": idx, "raw_response": None, "error": str(e)}


async def run_fold_async(model: str, fold: int, concurrency: int = 10):
    """Run a single fold with async concurrent requests."""
    import asyncio
    import openai
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    from tqdm.asyncio import tqdm

    print(f"\n{'='*60}")
    print(f"Processing Fold {fold} (concurrency={concurrency})")
    print(f"{'='*60}")

    # Load data
    descriptions = load_category_descriptions()
    all_labels = get_all_labels_from_cv_splits()
    valid_labels_set = set(normalize_label(l) for l in all_labels)

    test_data, _ = load_fold_data(fold, include_val=True)
    print(f"  Loaded {len(test_data)} samples (val + test)")

    # Load transition matrix for Cum70
    transition_matrix = load_transition_matrix(fold)
    tm_rows_normalized = {}
    if transition_matrix is not None:
        tm_rows_normalized = {normalize_label(r): r for r in transition_matrix.index}

    # Get async client
    api_model = IONOS_MODELS.get(model, model)
    if model.startswith("gpt-oss"):
        api_key = os.environ.get("IONOS_API_TOKEN")
        if not api_key:
            raise ValueError("IONOS_API_TOKEN environment variable not set")
        client = openai.AsyncOpenAI(api_key=api_key, base_url=IONOS_BASE_URL)
    else:
        client = openai.AsyncOpenAI()

    print(f"  Using model: {api_model}")

    # Build prompts
    prompts = []
    for i, sample in enumerate(test_data):
        prompt = build_ndap_prompt(sample["conversation_history"], all_labels, descriptions)
        prompts.append((i, prompt))

    ground_truth = [normalize_label(sample["label"]) for sample in test_data]

    # Process with semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(test_data)
    pbar = tqdm(total=len(test_data), desc=f"Fold {fold}", unit="sample")

    async def process_with_semaphore(idx, prompt):
        async with semaphore:
            result = await process_single_sample_async(
                client, api_model, model, prompt, idx, len(test_data), valid_labels_set
            )
            results[idx] = result
            pbar.update(1)
            return result

    # Run all tasks
    tasks = [process_with_semaphore(idx, prompt) for idx, prompt in prompts]
    try:
        await asyncio.gather(*tasks)
    finally:
        pbar.close()
        # Properly close the client to avoid event loop issues between folds
        await client.close()

    # Save raw outputs to JSONL file for potential recalculation
    raw_outputs_path = RESULTS_DIR / f"llm_raw_{model.replace('/', '_')}_fold_{fold}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Saving raw outputs to {raw_outputs_path}")
    with open(raw_outputs_path, 'w', encoding='utf-8') as f:
        for idx, (prompt_tuple, r) in enumerate(zip(prompts, results)):
            _, prompt = prompt_tuple
            sample = test_data[idx]
            record = {
                "idx": idx,
                "ground_truth": ground_truth[idx],
                "conversation_history": sample.get("conversation_history", ""),
                "raw_response": r.get("raw_response") if r else None,
                "top1": r.get("top1") if r else None,
                "top1_matched": r.get("top1_matched") if r else None,
                "top1_valid": r.get("top1_valid") if r else None,
                "top1_conf": r.get("top1_conf") if r else None,
                "top3": r.get("top3") if r else None,
                "error": r.get("error") if r else "no_result",
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Saved {len(results)} raw outputs")

    # Compute metrics - treat invalid predictions as wrong (not filtered out)
    # This ensures fair comparison with BERT models that must classify all samples
    le = LabelEncoder()
    le.fit([normalize_label(l) for l in all_labels])

    # Use a placeholder label for invalid predictions that won't match any ground truth
    INVALID_PLACEHOLDER = "__INVALID_PREDICTION__"

    y_true = []
    y_pred_labels = []
    valid_count = 0
    invalid_count = 0
    error_count = 0

    for idx, r in enumerate(results):
        y_true.append(ground_truth[idx])
        if r is None:
            # No result at all (shouldn't happen but handle it)
            y_pred_labels.append(INVALID_PLACEHOLDER)
            invalid_count += 1
        elif "error" in r:
            # API error
            y_pred_labels.append(INVALID_PLACEHOLDER)
            error_count += 1
            invalid_count += 1
        elif not r.get("top1_valid"):
            # Prediction couldn't be fuzzy-matched to valid label
            y_pred_labels.append(INVALID_PLACEHOLDER)
            invalid_count += 1
        else:
            # Valid prediction
            y_pred_labels.append(r["top1_matched"])
            valid_count += 1

    if valid_count == 0:
        print("  No valid predictions!")
        return None

    # For F1 calculation, we need numeric labels
    # Invalid predictions will always be wrong since INVALID_PLACEHOLDER won't match ground truth
    all_possible_labels = [normalize_label(l) for l in all_labels] + [INVALID_PLACEHOLDER]
    le_extended = LabelEncoder()
    le_extended.fit(all_possible_labels)

    y_true_encoded = le_extended.transform(y_true)
    y_pred_encoded = le_extended.transform(y_pred_labels)

    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)

    # Top-3 accuracy - invalid predictions count as wrong (no match possible)
    top3_correct = 0
    for idx, r in enumerate(results):
        if r and r.get("top1_valid") and "top3" in r:
            true_label = ground_truth[idx]
            matched_top3 = [fuzzy_match_label(c, valid_labels_set)[0] for c, _ in r["top3"]]
            if true_label in matched_top3:
                top3_correct += 1
        # else: invalid prediction, counts as wrong (top3_correct not incremented)
    top3_acc = top3_correct / len(results)

    # Cum70 - invalid predictions count as wrong (not in valid categories)
    cum70_count = 0
    cum70_valid = 0
    if transition_matrix is not None:
        for idx, r in enumerate(results):
            sample = test_data[idx]
            source_cat = extract_source_category_from_history(sample["conversation_history"])
            if source_cat:
                source_norm = normalize_label(source_cat)
                if source_norm in tm_rows_normalized:
                    row_name = tm_rows_normalized[source_norm]
                    transitions = transition_matrix.loc[row_name]
                    transitions_normalized = pd.Series(
                        transitions.values, index=[normalize_label(c) for c in transitions.index]
                    )
                    valid_cats = get_valid_categories_cumulative(transitions_normalized, 0.7)
                    cum70_count += 1
                    # Only count as valid if prediction is valid AND in valid categories
                    if r and r.get("top1_valid") and r["top1_matched"] in valid_cats:
                        cum70_valid += 1
                    # else: invalid prediction or not in valid categories = wrong
    cum70 = cum70_valid / cum70_count if cum70_count > 0 else 0.0

    invalid_rate = invalid_count / len(results)

    print(f"\n  Fold {fold} Results:")
    print(f"    Macro-F1:    {macro_f1:.3f}")
    print(f"    Weighted-F1: {weighted_f1:.3f}")
    print(f"    Top-3 Acc:   {top3_acc:.3f}")
    print(f"    Cum70:       {cum70:.3f}")
    print(f"    Invalid:     {invalid_rate:.2%} ({invalid_count}/{len(results)}, errors: {error_count})")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_accuracy": top3_acc,
        "cum70": cum70,
        "invalid_rate": invalid_rate,
        "num_valid": valid_count,
        "num_total": len(results),
    }


def cmd_full_cv(model: str, folds: List[int] = None, concurrency: int = 10):
    """Run full 5-fold CV with parallel requests."""
    import asyncio
    import numpy as np

    folds_to_run = folds if folds else list(range(NUM_FOLDS))
    results_file = RESULTS_DIR / f"llm_cv_{model.replace('/', '_')}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {"model": model, "per_fold": {}}

    for fold in folds_to_run:
        fold_result = asyncio.run(run_fold_async(model, fold, concurrency))
        if fold_result:
            all_results["per_fold"][f"fold_{fold}"] = fold_result

    # Aggregate
    if len(all_results["per_fold"]) >= 2:
        metrics = ["accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "cum70", "invalid_rate"]
        aggregated = {}
        for metric in metrics:
            values = [fr[metric] for fr in all_results["per_fold"].values() if metric in fr]
            if values:
                aggregated[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        all_results["aggregated"] = aggregated

        print(f"\n{'='*60}")
        print("FINAL AGGREGATED RESULTS")
        print(f"{'='*60}")
        for metric in ["macro_f1", "weighted_f1", "top3_accuracy", "cum70"]:
            if metric in aggregated:
                print(f"  {metric}: {aggregated[metric]['mean']:.3f} ± {aggregated[metric]['std']:.3f}")

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM Baseline for NDAP on OnCoCo")
    parser.add_argument("--submit", action="store_true", help="Submit batches for all folds")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--run", action="store_true", help="Run CV evaluation (use with --folds)")
    parser.add_argument("--evaluate", action="store_true", help="Download and evaluate results")
    parser.add_argument("--test", action="store_true", help="Run synchronous test with limited samples")
    parser.add_argument("--full-cv", action="store_true", help="Run full 5-fold CV with parallel requests")
    parser.add_argument("--folds", type=int, nargs="+", help="Specific folds to process (default: all)")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Model to use (default: {MODEL})")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples for --test (default: 100)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests (default: 10)")

    args = parser.parse_args()

    if args.submit:
        cmd_submit(args.folds)
    elif args.status:
        cmd_status()
    elif args.evaluate:
        cmd_evaluate()
    elif args.full_cv or args.run or args.folds:
        # --run, --full-cv, or --folds all trigger CV evaluation
        cmd_full_cv(args.model, args.folds, args.concurrency)
    elif args.test:
        fold = args.folds[0] if args.folds else 0
        cmd_test(args.model, args.limit, fold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
