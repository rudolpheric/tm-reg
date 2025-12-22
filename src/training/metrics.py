import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def _code(s: str) -> str:
    """Extract code-only identifier.

    Handles multiple formats:
    - "CODE | Description" -> "CODE" (OnCoCo format)
    - "K-CODE" or "B-CODE" -> "CODE" (speaker-prefixed format)
    - "CODE" -> "CODE" (simple format)
    """
    s = str(s).strip()

    # First, extract part before '|' if present (OnCoCo format)
    if '|' in s:
        s = s.split('|', 1)[0].strip()

    # Then, strip speaker prefix if present (K- or B-)
    if s.startswith('K-') or s.startswith('B-'):
        s = s[2:]

    return s


def normalize_tm_to_codes(transition_matrix: pd.DataFrame, label_encoder=None) -> pd.DataFrame:
    """
    Convert a transition matrix with verbose labels into a code-level matrix:
    - Row index: source codes (aggregated by mean across duplicate rows)
    - Columns: target codes (aggregated by sum across duplicate columns)
    - Rows are renormalized to sum to 1 (with small epsilon to avoid zeros)

    If label_encoder is provided, restrict columns to the set of codes present in
    label_encoder.classes_.
    """
    # Aggregate columns by code (sum), rows by code (mean)
    # Columns
    col_codes = [_code(c) for c in transition_matrix.columns]
    df_cols = transition_matrix.copy()
    df_cols.columns = col_codes
    df_cols = df_cols.T.groupby(level=0).sum().T

    # Rows
    row_codes = [ _code(r) for r in df_cols.index ]
    df_rows = df_cols.copy()
    df_rows.index = row_codes
    df_rows = df_rows.groupby(level=0).mean()

    # Optional: restrict columns to label_encoder codes
    if label_encoder is not None:
        target_codes = {_code(c) for c in getattr(label_encoder, 'classes_', [])}
        keep_cols = [c for c in df_rows.columns if c in target_codes]
        df_rows = df_rows.loc[:, keep_cols]

    # Add epsilon and renormalize each row
    eps = 1e-12
    df_rows = df_rows + eps
    df_rows = df_rows.div(df_rows.sum(axis=1), axis=0)
    return df_rows


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """
    Compute the top-k accuracy for the predictions.
    """
    num_classes = logits.size(1)
    # Limit k to the number of available classes
    k = min(k, num_classes)
    
    topk_preds = torch.topk(logits, k=k, dim=1).indices
    correct = topk_preds.eq(targets.unsqueeze(1).expand_as(topk_preds))
    return correct.sum().item() / targets.size(0)

def evaluate_transition_threshold(logits: torch.Tensor, 
                                 targets: torch.Tensor, 
                                 source_categories: list,
                                 transition_matrix: pd.DataFrame,
                                 label_encoder,
                                 threshold: float = 0.1) -> dict:
    """
    Evaluate how well the model's predictions align with transition probabilities.
    
    Args:
        logits: The model's output logits
        targets: The true target categories (indices)
        source_categories: List of source category names (strings)
        transition_matrix: DataFrame with transition probabilities
        label_encoder: The encoder used to convert between category names and indices
        threshold: The minimum transition probability to consider a valid prediction
        
    Returns:
        dict: Dictionary with transition-aware evaluation metrics
    """
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    results = {
        "total_samples": len(targets),
        "threshold": threshold,
        "overall_accuracy": (preds == targets).sum().item() / len(targets)
    }
    
    # Counters for transition-related metrics
    valid_source_count = 0
    valid_target_in_matrix = 0
    target_above_threshold = 0
    pred_above_threshold = 0
    correct_and_above_threshold = 0
    correct_but_below_threshold = 0
    incorrect_but_above_threshold = 0
    incorrect_and_below_threshold = 0
    
    # Top-k transition metrics
    top1_in_valid_transitions = 0
    top3_in_valid_transitions = 0
    top5_in_valid_transitions = 0
    # Constrained (masked) decoding accuracy
    constrained_correct = 0
    num_classes = logits.size(1)
    classes_set = set(getattr(label_encoder, 'classes_', []))

    # Normalize TM to code level consistent with label encoder
    code_tm = normalize_tm_to_codes(transition_matrix, label_encoder)

    # Create a mapping from codes to full category names for constrained decoding
    code_to_full_names = {}
    for full_name in classes_set:
        code = _code(full_name)
        if code not in code_to_full_names:
            code_to_full_names[code] = []
        code_to_full_names[code].append(full_name)

    # Precompute codes for targets and predictions
    def idx_to_code(idx: int) -> str:
        name = label_encoder.inverse_transform([idx])[0]
        return _code(name)

    for i, (logit, pred, target, source_cat) in enumerate(zip(logits, preds, targets, source_categories)):
        src_code = _code(source_cat) if source_cat is not None else None
        if not src_code or src_code not in code_tm.index:
            continue

        valid_source_count += 1
        transitions = code_tm.loc[src_code]
        
        # Get category names
        target_cat_code = idx_to_code(target.item())
        pred_cat_code = idx_to_code(pred.item())
        
        # Create a set of valid next categories (above threshold)
        valid_transitions = set(transitions[transitions >= threshold].index)
        
        # Check target's transition probability
        target_in_transitions = target_cat_code in transitions.index
        if target_in_transitions:
            valid_target_in_matrix += 1
            target_prob = transitions[target_cat_code]
            target_is_valid = target_prob >= threshold
            if target_is_valid:
                target_above_threshold += 1
        
        # Check prediction's transition probability
        pred_in_transitions = pred_cat_code in transitions.index
        pred_is_valid = False
        if pred_in_transitions:
            pred_prob = transitions[pred_cat_code]
            pred_is_valid = pred_prob >= threshold
            if pred_is_valid:
                pred_above_threshold += 1
        
        # Check correctness in relation to threshold
        is_correct = pred.item() == target.item()
        if is_correct and pred_is_valid:
            correct_and_above_threshold += 1
        elif is_correct and not pred_is_valid:
            correct_but_below_threshold += 1
        elif not is_correct and pred_is_valid:
            incorrect_but_above_threshold += 1
        else:
            incorrect_and_below_threshold += 1
            
        # Constrained decoding: mask logits for categories below threshold
        masked_pred_correct = False
        if len(valid_transitions) > 0 and classes_set:
            try:
                # Convert valid transition codes to full category names
                allowed_list = []
                for code in valid_transitions:
                    if code in code_to_full_names:
                        allowed_list.extend(code_to_full_names[code])

                if allowed_list:
                    allowed_indices = label_encoder.transform(allowed_list)
                    mask = torch.ones(num_classes, dtype=torch.bool, device=logit.device)
                    mask[allowed_indices] = False
                    masked_logits = logit.clone()
                    masked_logits[mask] = float('-inf')
                    masked_pred = torch.argmax(masked_logits).item()
                    masked_pred_correct = (masked_pred == target.item())
            except Exception as e:
                # Debug: print exception to understand what's failing
                # print(f"Exception in constrained decoding: {e}")
                masked_pred_correct = False
        if masked_pred_correct:
            constrained_correct += 1

        # Evaluate top-k with respect to valid transitions
        if valid_transitions:
            # Get top-k predictions (indices)
            topk_values, topk_indices = torch.topk(logit, k=min(5, len(logit)), dim=0)
            
            # Convert to category names
            topk_codes = [idx_to_code(idx.item()) for idx in topk_indices]
            
            # Check if any top-k prediction is in valid transitions
            if any(code in valid_transitions for code in topk_codes[:1]):
                top1_in_valid_transitions += 1
            if any(code in valid_transitions for code in topk_codes[:3]):
                top3_in_valid_transitions += 1
            if any(code in valid_transitions for code in topk_codes[:5]):
                top5_in_valid_transitions += 1
    
    # Calculate metrics if we have valid source categories
    if valid_source_count > 0:
        results.update({
            "valid_source_count": valid_source_count,
            "valid_source_percentage": valid_source_count / results["total_samples"],
            
            # Target-related metrics
            "valid_target_percentage": valid_target_in_matrix / valid_source_count,
            "target_above_threshold_percentage": target_above_threshold / valid_source_count if valid_target_in_matrix > 0 else 0,
            
            # Prediction-related metrics
            "predictions_above_threshold_percentage": pred_above_threshold / valid_source_count,
            
            # Correctness metrics
            "correct_and_above_threshold": correct_and_above_threshold / valid_source_count,
            "correct_but_below_threshold": correct_but_below_threshold / valid_source_count,
            "incorrect_but_above_threshold": incorrect_but_above_threshold / valid_source_count,
            "incorrect_and_below_threshold": incorrect_and_below_threshold / valid_source_count,
            
            # Top-k metrics with valid transitions
            "top1_in_valid_transitions": top1_in_valid_transitions / valid_source_count,
            "top3_in_valid_transitions": top3_in_valid_transitions / valid_source_count,
            "top5_in_valid_transitions": top5_in_valid_transitions / valid_source_count,
        })
        
        # Calculate transition-aware accuracy (correct predictions that are also above threshold)
        # divided by number of samples where target is above threshold
        if target_above_threshold > 0:
            results["transition_aware_accuracy"] = correct_and_above_threshold / target_above_threshold
        # Constrained decoding accuracy (masked top-1)
        results["constrained_accuracy"] = constrained_correct / valid_source_count
    
    return results

def calculate_entropy_and_js(pred_logits: torch.Tensor) -> (float, float):
    """
    Compute the mean entropy and Jensen-Shannon divergence for given logits
    over the full prediction distribution.
    """
    probs = torch.softmax(pred_logits, dim=1).cpu().numpy()
    entropies, js_divs = [], []
    for p in probs:
        uniform = np.ones_like(p) / len(p)
        m = 0.5 * (p + uniform)
        kl1 = np.sum(p * np.log((p + 1e-12) / (m + 1e-12)))
        kl2 = np.sum(uniform * np.log((uniform + 1e-12) / (m + 1e-12)))
        js = 0.5 * (kl1 + kl2)
        entropy = -np.sum(p * np.log(p + 1e-12))
        entropies.append(entropy)
        js_divs.append(js)
    return np.mean(entropies), np.mean(js_divs)

def filter_transition_matrix(transition_matrix, label_encoder):
    """
    Filter the transition matrix to match the label encoder.

    For client-only prediction, the label_encoder contains only client (K-) categories,
    but source categories can include both counselor (B-) and client (K-) categories.
    Therefore, we:
    - Keep ALL rows (source categories) that exist in the transition matrix
    - Filter columns (target categories) to match label_encoder classes

    Comparison is done using codes only (before the pipe |) to handle cases where
    the description text may differ.

    Args:
        transition_matrix (pd.DataFrame): The transition matrix to filter
        label_encoder: The scikit-learn LabelEncoder with fitted labels

    Returns:
        pd.DataFrame: Filtered transition matrix
    """
    # Get the set of valid target codes from the label encoder
    valid_target_codes = set(_code(label) for label in label_encoder.classes_)

    # Extract codes from transition matrix columns
    tm_col_codes = [_code(col) for col in transition_matrix.columns]

    # Find columns that need to be removed (not in label encoder)
    invalid_cols = [col for col, code in zip(transition_matrix.columns, tm_col_codes)
                   if code not in valid_target_codes]

    # Print information about what's being filtered
    if invalid_cols:
        print(f"Filtering transition matrix:")
        print(f"  Keeping all {len(transition_matrix.index)} source categories (rows)")
        print(f"  Filtering {len(invalid_cols)} target columns not in label_encoder")
        if len(invalid_cols) > 0:
            print(f"  Removed columns: {invalid_cols[:3]}{'...' if len(invalid_cols) > 3 else ''}")

    # Filter columns only - keep all rows (source categories)
    valid_cols = [col for col, code in zip(transition_matrix.columns, tm_col_codes)
                 if code in valid_target_codes]
    filtered_matrix = transition_matrix.loc[:, valid_cols]

    print(f"  Filtered matrix shape: {filtered_matrix.shape} (rows=sources, cols=targets)")

    return filtered_matrix



def calculate_entropy_and_js_with_transitions(pred_logits: torch.Tensor,
                                              source_categories: list,
                                              transition_matrix: pd.DataFrame,
                                              label_encoder) -> (float, float):
    """
    Compute mean entropy and mean Jensen-Shannon divergence for the given logits,
    using a baseline distribution based on a transition matrix.
    """
    probs = torch.softmax(pred_logits, dim=1).cpu().numpy()
    entropies = []
    js_divs = []
    batch_size = probs.shape[0]
    
    # Normalize TM to code-level
    code_tm = normalize_tm_to_codes(transition_matrix, label_encoder)

    # Helper to map class index -> code
    def idx_to_code(idx: int) -> str:
        name = label_encoder.inverse_transform([idx])[0]
        return _code(name)

    for i in range(batch_size):
        p = probs[i]
        source_cat = source_categories[i]
        src_code = _code(source_cat) if source_cat is not None else None
        if src_code in code_tm.index:
            row = code_tm.loc[src_code]
            # Allowed next codes (positive mass)
            allowed_codes = set(row[row > 0].index.tolist())
            if not allowed_codes:
                allowed_indices = np.arange(len(p))
                baseline = np.ones_like(p) / len(p)
                p_restricted = p
            else:
                # Map allowed codes to class indices
                allowed_indices = [j for j in range(len(p)) if idx_to_code(j) in allowed_codes]
                if not allowed_indices:
                    allowed_indices = np.arange(len(p))
                    baseline = np.ones_like(p) / len(p)
                    p_restricted = p
                else:
                    # Build baseline over class indices by assigning code probability to indices with that code
                    baseline = np.array([row.get(idx_to_code(j), 0.0) for j in allowed_indices], dtype=np.float64)
                    p_allowed = p[allowed_indices]
                    p_restricted = p_allowed / (p_allowed.sum() + 1e-12)
                    baseline = baseline / (baseline.sum() + 1e-12)
        else:
            allowed_indices = np.arange(len(p))
            baseline = np.ones_like(p) / len(p)
            p_restricted = p
        
        m = 0.5 * (p_restricted + baseline)
        kl1 = np.sum(p_restricted * np.log((p_restricted + 1e-12) / (m + 1e-12)))
        kl2 = np.sum(baseline * np.log((baseline + 1e-12) / (m + 1e-12)))
        js = 0.5 * (kl1 + kl2)
        entropy = -np.sum(p_restricted * np.log(p_restricted + 1e-12))
        
        entropies.append(entropy)
        js_divs.append(js)
    
    return np.mean(entropies), np.mean(js_divs)


def calculate_context_sensitivity(model, test_dataset, device, tokenizer):
    """
    Measure how sensitive the model is to changes in dialogue context.
    Higher sensitivity indicates the model pays attention to context.
    """
    model.eval()
    context_scores = []
    
    for i in range(min(100, len(test_dataset))):
        sample = test_dataset[i]
        history = sample["raw_history"]
        
        # Get prediction with full history
        with torch.no_grad():
            # Use the global tokenizer for prediction
            from prediction import predict_next  # Import here to avoid circular imports
            full_pred = predict_next(model, tokenizer, test_dataset.dataset.label_encoder, history, device)
            
            # Remove the last turn and predict again
            if " | " in history:
                truncated_history = " | ".join(history.split(" | ")[:-1])
                truncated_pred = predict_next(model, tokenizer, test_dataset.dataset.label_encoder, 
                                             truncated_history, device)
                
                # Calculate difference in prediction distribution
                diff = abs(full_pred["category_confidence"] - truncated_pred["category_confidence"])
                context_scores.append(diff)
    
    return {
        "mean_context_sensitivity": np.mean(context_scores) if context_scores else 0.0,
        "median_context_sensitivity": np.median(context_scores) if context_scores else 0.0
    }


def evaluate_turn_taking_accuracy(model, test_dataset, device, tokenizer):
    """
    Evaluate how well the model predicts turn-taking patterns.
    """
    model.eval()
    turn_accuracies = {"K->B": [], "B->K": []}
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            history = sample["raw_history"]
            target = sample["target_category_name"]
            
            # Get the current speaker from the last turn
            current_speaker = None
            if history.strip().endswith("K ("):
                current_speaker = "K"
            elif history.strip().endswith("B ("):
                current_speaker = "B"
            
            if current_speaker:
                # Use the global tokenizer for prediction and access label_encoder through dataset attribute
                from prediction import predict_next  # Import here to avoid circular imports
                pred = predict_next(model, tokenizer, test_dataset.dataset.label_encoder, history, device)
                pred_category = pred["predicted_category"]
                
                # Check if prediction starts with the expected other speaker
                expected_next_speaker = "B" if current_speaker == "K" else "K"
                if pred_category.startswith(expected_next_speaker):
                    turn_key = f"{current_speaker}->{expected_next_speaker}"
                    turn_accuracies[turn_key].append(1)
                else:
                    turn_key = f"{current_speaker}->{expected_next_speaker}"
                    turn_accuracies[turn_key].append(0)
    
    return {
        "K->B_accuracy": np.mean(turn_accuracies["K->B"]) if turn_accuracies["K->B"] else 0,
        "B->K_accuracy": np.mean(turn_accuracies["B->K"]) if turn_accuracies["B->K"] else 0
    }


def save_confusion_matrix_best_f1(y_true, y_pred, label_encoder, f1_score_val, save_path="confusion_matrix_best_f1.png", experiment_name=None):
    """
    Create and save confusion matrix visualization for the best F1 model.
    
    Args:
        y_true: True labels (indices)
        y_pred: Predicted labels (indices) 
        label_encoder: LabelEncoder with category names
        f1_score_val: F1 score value to display in title
        save_path: Path to save the image (will be modified if experiment_name is provided)
        experiment_name: Optional experiment name to create organized folder structure
    """
    import os
    from datetime import datetime
    
    # Add timestamp to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create organized folder structure if experiment name is provided
    if experiment_name:
        confusion_matrices_dir = "confusion_matrices"
        experiment_dir = os.path.join(confusion_matrices_dir, experiment_name)
        
        # Create directories if they don't exist
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Extract filename from save_path and create new path with timestamp
        filename = os.path.basename(save_path)
        
        # Create experiment-specific filename with timestamp
        name_part, ext = os.path.splitext(filename)
        if not name_part.startswith(experiment_name):
            name_part = f"{experiment_name}_{name_part}"
        
        # Add timestamp before extension
        filename_with_timestamp = f"{name_part}_{timestamp}{ext}"
        
        save_path = os.path.join(experiment_dir, filename_with_timestamp)
        print(f"Saving confusion matrix to organized path: {save_path}")
    else:
        # Fallback to original behavior but still create a basic folder and add timestamp
        confusion_matrices_dir = "confusion_matrices"
        os.makedirs(confusion_matrices_dir, exist_ok=True)
        
        # Add timestamp to filename
        filename = os.path.basename(save_path)
        name_part, ext = os.path.splitext(filename)
        filename_with_timestamp = f"{name_part}_{timestamp}{ext}"
        
        save_path = os.path.join(confusion_matrices_dir, filename_with_timestamp)
    def extract_short_category(full_category):
        """Extract short category name from full format."""
        if isinstance(full_category, str) and '|' in full_category:
            return full_category.split('|')[0].strip()
        return str(full_category)
    
    # Get class names and convert to short format
    class_names = label_encoder.classes_
    short_names = [extract_short_category(name) for name in class_names]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Create figure with large size to accommodate 60+ labels
    plt.figure(figsize=(24, 20))
    
    # Create heatmap
    sns.heatmap(cm, 
                annot=False,  # Don't show numbers to avoid clutter
                fmt='d',
                cmap='Blues',
                xticklabels=short_names,
                yticklabels=short_names,
                cbar_kws={'label': 'Count'})
    
    # Customize the plot
    plt.title(f'Confusion Matrix - Best Validation F1 Model (F1: {f1_score_val:.4f})', fontsize=16, pad=20)
    plt.xlabel('Predicted Category', fontsize=14)
    plt.ylabel('True Category', fontsize=14)
    
    # Rotate labels for better readability (90 degrees as requested)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    
    # Also save as PDF for better quality
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Confusion matrix also saved as PDF: {pdf_path}")
    
    plt.close()

    return save_path


def save_transition_alignment_matrix(
    y_true,
    y_pred,
    source_categories,
    transition_matrix: pd.DataFrame,
    label_encoder,
    f1_score_val,
    threshold: float = 0.05,
    save_path="transition_alignment_matrix.png",
    experiment_name=None
):
    """
    Create a visualization showing how model predictions align with allowed transitions.

    This creates a three-panel visualization:
    1. Expected transitions (from transition matrix, filtered by threshold)
    2. Empirical predictions (actual model predictions)
    3. Divergence map (over-prediction vs under-prediction)

    Args:
        y_true: True labels (indices)
        y_pred: Predicted labels (indices)
        source_categories: List of source category names for each prediction
        transition_matrix: DataFrame with expected transition probabilities
        label_encoder: LabelEncoder with category names
        f1_score_val: F1 score value to display in title
        threshold: Minimum probability to consider a transition "allowed" (default 0.05)
        save_path: Path to save the image
        experiment_name: Optional experiment name for organized folder structure
    """
    import os
    from datetime import datetime

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create organized folder structure
    if experiment_name:
        matrices_dir = "transition_alignment_matrices"
        experiment_dir = os.path.join(matrices_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        filename = os.path.basename(save_path)
        name_part, ext = os.path.splitext(filename)
        if not name_part.startswith(experiment_name):
            name_part = f"{experiment_name}_{name_part}"
        filename_with_timestamp = f"{name_part}_{timestamp}{ext}"
        save_path = os.path.join(experiment_dir, filename_with_timestamp)
    else:
        matrices_dir = "transition_alignment_matrices"
        os.makedirs(matrices_dir, exist_ok=True)
        filename = os.path.basename(save_path)
        name_part, ext = os.path.splitext(filename)
        filename_with_timestamp = f"{name_part}_{timestamp}{ext}"
        save_path = os.path.join(matrices_dir, filename_with_timestamp)

    def extract_short_category(full_category):
        """Extract short category name from full format."""
        if isinstance(full_category, str) and '|' in full_category:
            return full_category.split('|')[0].strip()
        return str(full_category)

    # Get class names and convert to short format (codes only)
    class_names = label_encoder.classes_
    short_names = [extract_short_category(name) for name in class_names]

    # Create unique list of codes (deduplicated)
    unique_codes = []
    seen_codes = set()
    code_to_full_indices = {}  # Map from code to all matching full category indices

    for idx, name in enumerate(class_names):
        code = extract_short_category(name)
        if code not in seen_codes:
            unique_codes.append(code)
            seen_codes.add(code)
            code_to_full_indices[code] = []
        code_to_full_indices[code].append(idx)

    n_unique_codes = len(unique_codes)

    # Normalize transition matrix to code-level
    tm_normalized = normalize_tm_to_codes(transition_matrix, label_encoder)

    # Build empirical prediction matrix (source code -> target code)
    # Using unique codes to avoid duplicates in visualization
    empirical_matrix = np.zeros((n_unique_codes, n_unique_codes))
    source_counts = np.zeros(n_unique_codes)

    # Create reverse mapping from code to index in unique_codes list
    code_to_unique_idx = {code: idx for idx, code in enumerate(unique_codes)}

    for src_cat, pred_idx in zip(source_categories, y_pred):
        if src_cat is None or src_cat == "":
            continue

        # Extract codes
        src_code = extract_short_category(src_cat)
        pred_name = label_encoder.inverse_transform([pred_idx])[0]
        pred_code = extract_short_category(pred_name)

        # Map to unique code indices
        if src_code in code_to_unique_idx and pred_code in code_to_unique_idx:
            src_idx = code_to_unique_idx[src_code]
            tgt_idx = code_to_unique_idx[pred_code]

            empirical_matrix[src_idx, tgt_idx] += 1
            source_counts[src_idx] += 1

    # Normalize empirical matrix by row (convert counts to probabilities)
    for i in range(n_unique_codes):
        if source_counts[i] > 0:
            empirical_matrix[i, :] /= source_counts[i]

    # Build expected matrix from transition matrix using unique codes
    expected_matrix = np.zeros((n_unique_codes, n_unique_codes))
    for i, src_code in enumerate(unique_codes):
        if src_code in tm_normalized.index:
            for j, tgt_code in enumerate(unique_codes):
                if tgt_code in tm_normalized.columns:
                    expected_matrix[i, j] = tm_normalized.loc[src_code, tgt_code]

    # Create binary mask for allowed transitions (>= threshold)
    allowed_mask = expected_matrix >= threshold

    # Calculate divergence: empirical - expected (only for allowed transitions)
    # Positive = over-predicting, Negative = under-predicting
    divergence_matrix = np.where(allowed_mask, empirical_matrix - expected_matrix, np.nan)

    # Create three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))

    # Panel 1: Expected transitions (from transition matrix)
    ax1 = axes[0]
    masked_expected = np.where(allowed_mask, expected_matrix, np.nan)
    sns.heatmap(
        masked_expected,
        annot=False,
        cmap='Greens',
        xticklabels=unique_codes,
        yticklabels=unique_codes,
        cbar_kws={'label': 'Expected Probability'},
        ax=ax1,
        vmin=0,
        vmax=1.0,
        mask=~allowed_mask
    )
    ax1.set_title(
        f'Expected Transitions (TM ≥ {threshold})\n{np.sum(allowed_mask)} allowed transitions',
        fontsize=14,
        fontweight='bold'
    )
    ax1.set_xlabel('Target Category', fontsize=12)
    ax1.set_ylabel('Source Category', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, fontsize=8)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)

    # Panel 2: Empirical predictions (actual model behavior)
    ax2 = axes[1]
    masked_empirical = np.where(allowed_mask, empirical_matrix, np.nan)
    sns.heatmap(
        masked_empirical,
        annot=False,
        cmap='Blues',
        xticklabels=unique_codes,
        yticklabels=unique_codes,
        cbar_kws={'label': 'Empirical Probability'},
        ax=ax2,
        vmin=0,
        vmax=1.0,
        mask=~allowed_mask
    )
    ax2.set_title(
        f'Empirical Predictions (F1: {f1_score_val:.4f})\nOnly showing allowed transitions',
        fontsize=14,
        fontweight='bold'
    )
    ax2.set_xlabel('Target Category', fontsize=12)
    ax2.set_ylabel('Source Category', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=8)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

    # Panel 3: Divergence (empirical - expected)
    ax3 = axes[2]
    # Use diverging colormap: blue = under-predicting, red = over-predicting
    max_abs_div = np.nanmax(np.abs(divergence_matrix)) if not np.all(np.isnan(divergence_matrix)) else 1.0
    sns.heatmap(
        divergence_matrix,
        annot=False,
        cmap='RdBu_r',  # Red for over-prediction, Blue for under-prediction
        center=0,
        xticklabels=unique_codes,
        yticklabels=unique_codes,
        cbar_kws={'label': 'Divergence (Empirical - Expected)'},
        ax=ax3,
        vmin=-max_abs_div,
        vmax=max_abs_div,
        mask=np.isnan(divergence_matrix)
    )
    ax3.set_title(
        f'Prediction Divergence\nRed = Over-predicting, Blue = Under-predicting',
        fontsize=14,
        fontweight='bold'
    )
    ax3.set_xlabel('Target Category', fontsize=12)
    ax3.set_ylabel('Source Category', fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, fontsize=8)
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Transition alignment matrix saved to: {save_path}")

    # Also save as PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Transition alignment matrix also saved as PDF: {pdf_path}")

    plt.close()

    # Export detailed statistics CSV
    csv_path = save_path.replace('.png', '_stats.csv')
    stats_list = []

    for i in range(n_unique_codes):
        src_code = unique_codes[i]
        for j in range(n_unique_codes):
            tgt_code = unique_codes[j]
            if allowed_mask[i, j]:
                stats_list.append({
                    'source': src_code,
                    'target': tgt_code,
                    'expected_prob': expected_matrix[i, j],
                    'empirical_prob': empirical_matrix[i, j],
                    'divergence': divergence_matrix[i, j],
                    'sample_count': int(empirical_matrix[i, j] * source_counts[i]) if source_counts[i] > 0 else 0
                })

    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        df_stats = df_stats.sort_values('divergence', key=abs, ascending=False)
        df_stats.to_csv(csv_path, index=False)
        print(f"Divergence statistics saved to: {csv_path}")

        # Print top mismatches
        print(f"\nTop 10 over-predicted transitions:")
        print(df_stats.nlargest(10, 'divergence')[['source', 'target', 'expected_prob', 'empirical_prob', 'divergence']].to_string(index=False))

        print(f"\nTop 10 under-predicted transitions:")
        print(df_stats.nsmallest(10, 'divergence')[['source', 'target', 'expected_prob', 'empirical_prob', 'divergence']].to_string(index=False))

    return save_path


def create_predicted_transitions_matrix(model, dataloader, device, label_encoder, save_path=None, experiment_name=None):
    """
    Create a predicted transitions matrix from model predictions on validation/test data.
    
    Args:
        model: Trained model
        dataloader: DataLoader for validation/test data
        device: Device to run inference on
        label_encoder: LabelEncoder for category names
        save_path: Optional base path to save the matrix (without extension)
        experiment_name: Optional experiment name for organized folder structure
        
    Returns:
        pd.DataFrame: Predicted transitions matrix with source categories as rows, 
                     target categories as columns, and predicted probabilities as values
    """
    model.eval()
    
    # Dictionary to store transitions: {source_cat: {target_cat: count}}
    transitions = {}
    total_samples = 0
    valid_transitions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            source_categories = batch.get("source_category", [None] * len(input_ids))
            
            # Get model predictions
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_indices = torch.argmax(logits, dim=1)
            
            # Process each sample in the batch
            for i in range(len(input_ids)):
                total_samples += 1
                source_category = source_categories[i]

                # Skip if no source category available
                if source_category is None or source_category == "":
                    continue

                valid_transitions += 1

                # Get predicted target category
                pred_idx = predicted_indices[i].item()
                target_category = label_encoder.inverse_transform([pred_idx])[0]

                # Extract just the codes (before the pipe)
                source_code = _code(source_category)
                target_code = _code(target_category)

                # Initialize source category in transitions dict if not exists
                if source_code not in transitions:
                    transitions[source_code] = {}

                # Initialize target category count if not exists
                if target_code not in transitions[source_code]:
                    transitions[source_code][target_code] = 0

                # Increment count
                transitions[source_code][target_code] += 1
    
    print(f"Processed {total_samples} total samples, {valid_transitions} with valid source categories")
    print(f"Found transitions from {len(transitions)} source categories")
    
    # Convert to DataFrame with proper normalization
    # Extract codes from label encoder classes
    all_category_codes = sorted(set(_code(c) for c in label_encoder.classes_))

    # Create empty matrix
    matrix_data = []
    source_categories_with_data = []

    for source_code in sorted(transitions.keys()):
        row = []
        total_transitions = sum(transitions[source_code].values())

        if total_transitions > 0:
            source_categories_with_data.append(source_code)
            for target_code in all_category_codes:
                count = transitions[source_code].get(target_code, 0)
                probability = count / total_transitions
                row.append(probability)
            matrix_data.append(row)

    # Create DataFrame
    predicted_matrix = pd.DataFrame(
        matrix_data,
        index=source_categories_with_data,
        columns=all_category_codes
    )
    
    # Save if path provided
    if save_path:
        save_predicted_transitions_matrix(predicted_matrix, save_path, experiment_name, total_samples, valid_transitions)
    
    return predicted_matrix


def save_predicted_transitions_matrix(predicted_matrix, base_save_path, experiment_name=None, total_samples=0, valid_transitions=0):
    """
    Save predicted transitions matrix as both CSV and PNG with timestamp and organized folder structure.
    
    Args:
        predicted_matrix: DataFrame with predicted transitions
        base_save_path: Base path for saving (without extension)
        experiment_name: Optional experiment name for organized folders
        total_samples: Total number of samples processed
        valid_transitions: Number of samples with valid transitions
    """
    import os
    from datetime import datetime
    
    # Add timestamp to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create organized folder structure
    if experiment_name:
        transitions_dir = "predicted_transitions"
        experiment_dir = os.path.join(transitions_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Extract filename from base_save_path and create new path with timestamp
        filename = os.path.basename(base_save_path)
        
        # Create experiment-specific filename with timestamp
        if not filename.startswith(experiment_name):
            filename = f"{experiment_name}_{filename}"
        
        # Add timestamp to filename
        filename_with_timestamp = f"{filename}_{timestamp}"
        
        base_path = os.path.join(experiment_dir, filename_with_timestamp)
        print(f"Saving predicted transitions matrix to organized path: {base_path}")
    else:
        # Fallback to basic folder structure with timestamp
        transitions_dir = "predicted_transitions"
        os.makedirs(transitions_dir, exist_ok=True)
        
        filename = os.path.basename(base_save_path)
        filename_with_timestamp = f"{filename}_{timestamp}"
        
        base_path = os.path.join(transitions_dir, filename_with_timestamp)
        print(f"Saving predicted transitions matrix to: {base_path}")
    
    # Save as CSV
    csv_path = f"{base_path}.csv"
    predicted_matrix.to_csv(csv_path)
    print(f"Predicted transitions matrix CSV saved to: {csv_path}")
    print(f"Matrix shape: {predicted_matrix.shape}")
    
    # Save as PNG visualization
    png_path = f"{base_path}.png"
    _create_transitions_heatmap(predicted_matrix, png_path, total_samples, valid_transitions)
    
    # Also save as PDF for better quality
    pdf_path = f"{base_path}.pdf"
    _create_transitions_heatmap(predicted_matrix, pdf_path, total_samples, valid_transitions, format='pdf')


def _create_transitions_heatmap(predicted_matrix, save_path, total_samples=0, valid_transitions=0, format='png'):
    """
    Create a heatmap visualization of the predicted transitions matrix.
    
    Args:
        predicted_matrix: DataFrame with predicted transitions
        save_path: Path to save the visualization
        total_samples: Total number of samples processed
        valid_transitions: Number of samples with valid transitions
        format: Output format ('png' or 'pdf')
    """
    def extract_short_category(full_category):
        """Extract short category name from full format."""
        if isinstance(full_category, str) and '|' in full_category:
            return full_category.split('|')[0].strip()
        return str(full_category)
    
    # Get short names for better readability
    short_source_names = [extract_short_category(name) for name in predicted_matrix.index]
    short_target_names = [extract_short_category(name) for name in predicted_matrix.columns]
    
    # Create figure with appropriate size
    fig_width = max(16, len(predicted_matrix.columns) * 0.4)
    fig_height = max(12, len(predicted_matrix.index) * 0.4)
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap
    sns.heatmap(predicted_matrix.values, 
                annot=False,  # Don't show numbers to avoid clutter
                fmt='.3f',
                cmap='Blues',
                xticklabels=short_target_names,
                yticklabels=short_source_names,
                cbar_kws={'label': 'Predicted Transition Probability'})
    
    # Customize the plot
    title = f'Predicted Transitions Matrix\n({valid_transitions} valid transitions from {total_samples} samples)'
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Target Category (Predicted)', fontsize=12)
    plt.ylabel('Source Category', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    if format == 'pdf':
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Predicted transitions matrix PDF saved to: {save_path}")
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predicted transitions matrix PNG saved to: {save_path}")
    
    plt.close()


def get_valid_categories_cumulative(transition_probs, cumulative_threshold=0.8):
    """
    Get valid target categories based on cumulative probability mass.

    This adaptive approach takes the top-K categories that together account for
    cumulative_threshold% of the transition probability mass.

    Benefits:
    - Focused distributions: selects fewer categories (e.g., 1-2)
    - Scattered distributions: selects more categories (e.g., 5-8)
    - No arbitrary threshold

    Args:
        transition_probs: Series or array of transition probabilities for each target category
        cumulative_threshold: Cumulative mass to cover (default: 0.8 = 80%)

    Returns:
        set: Set of valid category indices/names
    """
    if isinstance(transition_probs, pd.Series):
        sorted_indices = np.argsort(transition_probs.values)[::-1]
        sorted_probs = transition_probs.values[sorted_indices]
        category_names = transition_probs.index[sorted_indices]
    else:
        sorted_indices = np.argsort(transition_probs)[::-1]
        sorted_probs = transition_probs[sorted_indices]
        category_names = sorted_indices

    # Find how many categories needed to reach cumulative threshold
    cumsum = np.cumsum(sorted_probs)
    n_valid = np.searchsorted(cumsum, cumulative_threshold) + 1
    n_valid = min(n_valid, len(sorted_probs))  # Don't exceed total categories

    return set(category_names[:n_valid])


def evaluate_cumulative_mass_metrics(logits: torch.Tensor,
                                     targets: torch.Tensor,
                                     source_categories: list,
                                     transition_matrix: pd.DataFrame,
                                     label_encoder,
                                     cumulative_thresholds: list = [0.7, 0.8, 0.9]) -> dict:
    """
    Evaluate next category prediction using cumulative mass approach.

    This provides three complementary metrics:
    1. pred_accuracy: Prediction in statistically likely transitions
    2. true_coverage: Ground truth in statistically likely transitions
    3. pred_accuracy_plus_true: Prediction in (likely transitions + ground truth)

    Args:
        logits: Model output logits
        targets: True target categories (indices)
        source_categories: List of source category names
        transition_matrix: DataFrame with transition probabilities
        label_encoder: Encoder for category names/indices
        cumulative_thresholds: List of cumulative mass thresholds to evaluate

    Returns:
        dict: Dictionary with metrics for each threshold
    """
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    # Normalize transition matrix to code level
    code_tm = normalize_tm_to_codes(transition_matrix, label_encoder)

    # Helper to map class index -> code
    def idx_to_code(idx: int) -> str:
        name = label_encoder.inverse_transform([idx])[0]
        return _code(name)

    results = {}

    for cum_thresh in cumulative_thresholds:
        pred_in_valid = 0
        true_in_valid = 0
        pred_in_valid_plus_true = 0
        valid_source_count = 0

        for pred, target, source_cat in zip(preds, targets, source_categories):
            src_code = _code(source_cat) if source_cat is not None else None
            if not src_code or src_code not in code_tm.index:
                continue

            valid_source_count += 1
            transitions = code_tm.loc[src_code]

            # Get category codes
            target_code = idx_to_code(target.item())
            pred_code = idx_to_code(pred.item())

            # Get valid categories based on cumulative mass
            valid_cats_cumulative = get_valid_categories_cumulative(transitions, cum_thresh)

            # Valid categories plus true (for fairness)
            valid_cats_plus_true = valid_cats_cumulative.copy()
            valid_cats_plus_true.add(target_code)

            # Metric 1: Prediction in cumulative valid?
            if pred_code in valid_cats_cumulative:
                pred_in_valid += 1

            # Metric 2: Ground truth in cumulative valid?
            if target_code in valid_cats_cumulative:
                true_in_valid += 1

            # Metric 3: Prediction in (cumulative + true)?
            if pred_code in valid_cats_plus_true:
                pred_in_valid_plus_true += 1

        # Calculate percentages
        if valid_source_count > 0:
            results[f'cumulative{int(cum_thresh*100)}_pred_accuracy'] = pred_in_valid / valid_source_count
            results[f'cumulative{int(cum_thresh*100)}_true_coverage'] = true_in_valid / valid_source_count
            results[f'cumulative{int(cum_thresh*100)}_pred_accuracy_plus_true'] = pred_in_valid_plus_true / valid_source_count
        else:
            results[f'cumulative{int(cum_thresh*100)}_pred_accuracy'] = 0.0
            results[f'cumulative{int(cum_thresh*100)}_true_coverage'] = 0.0
            results[f'cumulative{int(cum_thresh*100)}_pred_accuracy_plus_true'] = 0.0

    results['valid_source_count_cumulative'] = valid_source_count

    return results


def extract_source_category_from_history(conversation_history):
    """
    Extract the source category (last speaker's category) from conversation history.

    Args:
        conversation_history: String containing the conversation history

    Returns:
        str: Source category name or None if not found
    """
    if not conversation_history or not isinstance(conversation_history, str):
        return None

    # Split by newlines to get individual turns
    turns = conversation_history.strip().split('\n')

    if not turns:
        return None

    # Get the last turn that has a category
    for turn in reversed(turns):
        if '(' in turn and '|' in turn:
            # Extract category from format: "K (K-FA-*-*-A-* | Anrede): text"
            try:
                # Find the part between parentheses
                start = turn.find('(') + 1
                end = turn.find(')', start)
                if start > 0 and end > start:
                    category_part = turn[start:end]
                    # Split by | and take the first part (the category code)
                    if '|' in category_part:
                        source_category = category_part.split('|')[0].strip()
                        return source_category
            except:
                continue

    return None


def compare_predicted_vs_actual_transitions(predicted_matrix, actual_matrix, save_comparison_path=None):
    """
    Compare predicted transitions matrix with actual transitions matrix.
    
    Args:
        predicted_matrix: DataFrame with predicted transitions
        actual_matrix: DataFrame with actual transitions (ground truth)
        save_comparison_path: Optional path to save comparison results
        
    Returns:
        dict: Comparison metrics including MSE, correlation, etc.
    """
    # Align matrices to have same rows and columns
    common_rows = sorted(set(predicted_matrix.index) & set(actual_matrix.index))
    common_cols = sorted(set(predicted_matrix.columns) & set(actual_matrix.columns))
    
    if not common_rows or not common_cols:
        print("Warning: No common categories found between predicted and actual matrices")
        return {}
    
    # Subset matrices to common categories
    pred_aligned = predicted_matrix.loc[common_rows, common_cols]
    actual_aligned = actual_matrix.loc[common_rows, common_cols]
    
    # Calculate comparison metrics
    pred_values = pred_aligned.values.flatten()
    actual_values = actual_aligned.values.flatten()
    
    # Mean Squared Error
    mse = np.mean((pred_values - actual_values) ** 2)
    
    # Correlation
    correlation = np.corrcoef(pred_values, actual_values)[0, 1] if len(pred_values) > 1 else 0.0
    
    # KL Divergence (element-wise)
    kl_divergences = []
    for i in range(len(common_rows)):
        pred_row = pred_aligned.iloc[i].values + 1e-12  # Add small epsilon
        actual_row = actual_aligned.iloc[i].values + 1e-12
        
        # Normalize to ensure they sum to 1
        pred_row = pred_row / pred_row.sum()
        actual_row = actual_row / actual_row.sum()
        
        kl_div = np.sum(actual_row * np.log(actual_row / pred_row))
        kl_divergences.append(kl_div)
    
    mean_kl_divergence = np.mean(kl_divergences)
    
    comparison_results = {
        "common_source_categories": len(common_rows),
        "common_target_categories": len(common_cols),
        "mean_squared_error": mse,
        "correlation": correlation,
        "mean_kl_divergence": mean_kl_divergence,
        "predicted_matrix_shape": predicted_matrix.shape,
        "actual_matrix_shape": actual_matrix.shape
    }
    
    # Save comparison if path provided
    if save_comparison_path:
        import json
        with open(save_comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"Comparison results saved to: {save_comparison_path}")
    
    return comparison_results
