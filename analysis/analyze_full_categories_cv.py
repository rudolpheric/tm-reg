#!/usr/bin/env python3
"""
Analysis script for full_categories (60-label) experiments with 5-fold cross-validation.

Aggregates results across CV folds, computes mean and confidence intervals.
Compares TM regularization effect on 60 labels vs 28 labels (client-only).

Usage:
    python analyze_full_categories_cv.py
    python analyze_full_categories_cv.py --data path/to/wandb_export.csv
"""

import argparse
import re
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Reference results from client_categories (28 labels) - from paper
CLIENT_28_RESULTS = {
    # Mean F1 across all BERT configurations per tm_weight
    'tm_weight_means': {
        0.0: 0.2246,
        0.2: 0.2549,
        0.5: 0.2646,
        1.0: 0.2505,
        1.5: 0.2530,
    },
    'best_f1': 0.292,
    'best_model': 'GBERT-large',
    'best_tm_weight': 0.2,
    'optimal_lambda': 0.5,  # For mean F1
    'relative_improvement': 0.178,  # 17.8% from lambda=0 to lambda=0.5
}

# Number of folds
N_FOLDS = 5


def load_data(data_path: Path = None) -> pd.DataFrame:
    """Load the most recent wandb export CSV."""
    if data_path:
        return pd.read_csv(data_path)

    # Find most recent export
    export_dir = Path(__file__).parent
    csvs = list(export_dir.glob("results/wandb_export_full_categories_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            "No wandb export found. Run: python export_wandb_logs.py --project full_categories_"
        )
    latest = max(csvs, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest}")
    return pd.read_csv(latest)


def extract_fold_from_name(name: str) -> int:
    """Extract fold number from experiment name (e.g., 'exp_name_fold0' -> 0)."""
    match = re.search(r'_fold(\d+)$', name)
    if match:
        return int(match.group(1))
    return -1


def extract_base_experiment_name(name: str) -> str:
    """Remove fold suffix from experiment name."""
    return re.sub(r'_fold\d+$', '', name)


def extract_context_from_run_name(name: str) -> int:
    """Extract context length from run name (e.g., 'ctx_4_model_type' -> 4)."""
    match = re.search(r'ctx_(\d+)', str(name))
    if match:
        return int(match.group(1))
    return None


def filter_bert_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to BERT models only (exclude RNN baselines, hierarchy variants)."""
    # Keep only model_type == 'bert' (exclude 'history', 'hierarchy', etc.)
    bert_df = df[df['model_type'] == 'bert'].copy()

    # Also filter out transition_matrix baseline and RNN models
    excluded_models = ['transition_matrix', 'simple_rnn', 'hierarchical_rnn',
                       'minimal_rnn', 'tanaka', 'unknown']
    for excl in excluded_models:
        bert_df = bert_df[~bert_df['pretrained_model'].str.contains(excl, case=False, na=False)]

    return bert_df


def filter_valid_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid models: 'bert', 'history', and 'history_easy' model types.

    Excludes RNN baselines and other non-transformer models.
    """
    # Keep 'bert', 'history', and 'history_easy' model types
    valid_df = df[df['model_type'].isin(['bert', 'history', 'history_easy'])].copy()

    # Filter out RNN baselines and unknown models
    excluded_models = ['transition_matrix', 'simple_rnn', 'hierarchical_rnn',
                       'minimal_rnn', 'tanaka', 'unknown']
    for excl in excluded_models:
        valid_df = valid_df[~valid_df['pretrained_model'].str.contains(excl, case=False, na=False)]

    return valid_df


def compute_cv_statistics(values: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Compute mean, std, and confidence interval for CV results.

    Args:
        values: Array of values across folds
        confidence: Confidence level for interval (default 0.95 for 95% CI)

    Returns:
        Dict with mean, std, se, ci_lower, ci_upper
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std with Bessel's correction
    se = std / np.sqrt(n)  # Standard error

    # t-distribution for small samples
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_margin = t_crit * se

    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': mean - ci_margin,
        'ci_upper': mean + ci_margin,
        'n_folds': n,
    }


def aggregate_cv_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results across CV folds.

    Groups experiments by base name (without fold suffix) and computes
    mean, std, and confidence intervals across folds.
    """
    finished = df[df['state'] == 'finished'].copy()

    if finished.empty:
        print("WARNING: No finished runs found!")
        return pd.DataFrame()

    # Extract fold info (handle both 'Name' and 'run_name' column names)
    name_col = 'run_name' if 'run_name' in finished.columns else 'Name'
    finished['fold'] = finished[name_col].apply(extract_fold_from_name)
    finished['base_exp_name'] = finished[name_col].apply(extract_base_experiment_name)

    # Filter to only experiments with fold info
    with_folds = finished[finished['fold'] >= 0].copy()
    print(f"Experiments with fold info: {len(with_folds)}")

    # Group by base experiment name
    aggregated = []
    for base_name, group in with_folds.groupby('base_exp_name'):
        folds_present = sorted(group['fold'].unique())

        # Skip if not all folds present
        n_folds = len(folds_present)

        # Get experiment config from first run
        first_run = group.iloc[0]

        # Compute statistics for val_f1
        f1_values = group['val_f1'].dropna().values
        if len(f1_values) == 0:
            continue

        f1_stats = compute_cv_statistics(f1_values)

        # Compute statistics for other metrics
        top3_values = group['top_k_acc_3'].dropna().values
        top3_stats = compute_cv_statistics(top3_values) if len(top3_values) > 0 else {}

        acc_values = group['val_accuracy'].dropna().values
        acc_stats = compute_cv_statistics(acc_values) if len(acc_values) > 0 else {}

        # Cumulative@70 statistics
        cumul70_col = 'val_cumulative70_pred_accuracy_plus_true'
        cumul70_values = group[cumul70_col].dropna().values if cumul70_col in group.columns else np.array([])
        cumul70_stats = compute_cv_statistics(cumul70_values) if len(cumul70_values) > 0 else {}

        # Macro F1 statistics
        macro_f1_col = 'class_macro_avg_f1'
        macro_f1_values = group[macro_f1_col].dropna().values if macro_f1_col in group.columns else np.array([])
        macro_f1_stats = compute_cv_statistics(macro_f1_values) if len(macro_f1_values) > 0 else {}

        # JS Divergence statistics
        js_col = 'mean_js_divergence'
        js_values = group[js_col].dropna().values if js_col in group.columns else np.array([])
        js_stats = compute_cv_statistics(js_values) if len(js_values) > 0 else {}

        # Get context length - from column if available, otherwise from run name
        context_len = first_run.get('max_context_utterances')
        if pd.isna(context_len) or context_len is None:
            context_len = extract_context_from_run_name(base_name)

        aggregated.append({
            'base_exp_name': base_name,
            'pretrained_model': first_run.get('pretrained_model'),
            'model_type': first_run.get('model_type'),
            'tm_weight': first_run.get('tm_weight'),
            'max_context_utterances': context_len,

            # F1 statistics
            'val_f1_mean': f1_stats['mean'],
            'val_f1_std': f1_stats['std'],
            'val_f1_se': f1_stats['se'],
            'val_f1_ci_lower': f1_stats['ci_lower'],
            'val_f1_ci_upper': f1_stats['ci_upper'],

            # Top-3 accuracy statistics
            'top_k_acc_3_mean': top3_stats.get('mean'),
            'top_k_acc_3_std': top3_stats.get('std'),
            'top_k_acc_3_ci_lower': top3_stats.get('ci_lower'),
            'top_k_acc_3_ci_upper': top3_stats.get('ci_upper'),

            # Accuracy statistics
            'val_accuracy_mean': acc_stats.get('mean'),
            'val_accuracy_std': acc_stats.get('std'),

            # Cumulative@70 statistics
            'cumul70_mean': cumul70_stats.get('mean'),
            'cumul70_std': cumul70_stats.get('std'),
            'cumul70_ci_lower': cumul70_stats.get('ci_lower'),
            'cumul70_ci_upper': cumul70_stats.get('ci_upper'),

            # Macro F1 statistics
            'macro_f1_mean': macro_f1_stats.get('mean'),
            'macro_f1_std': macro_f1_stats.get('std'),
            'macro_f1_ci_lower': macro_f1_stats.get('ci_lower'),
            'macro_f1_ci_upper': macro_f1_stats.get('ci_upper'),

            # JS Divergence statistics
            'js_divergence_mean': js_stats.get('mean'),
            'js_divergence_std': js_stats.get('std'),

            # Fold info
            'n_folds': n_folds,
            'folds_present': str(folds_present),
            'complete': n_folds == N_FOLDS,
        })

    return pd.DataFrame(aggregated)


def analyze_tm_effect_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze effect of tm_weight on F1 with CV aggregation.

    First aggregates results per experiment across folds, then
    computes statistics across experiments for each tm_weight.
    """
    # First aggregate across folds
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = cv_df[cv_df['complete'] == True].copy()
    print(f"Complete CV experiments: {len(complete_cv)}")

    # Group by tm_weight
    tm_stats = []
    for tm_weight, group in complete_cv.groupby('tm_weight'):
        f1_means = group['val_f1_mean'].values
        top3_means = group['top_k_acc_3_mean'].dropna().values

        # Mean of means (grand mean)
        f1_grand_stats = compute_cv_statistics(f1_means)

        # For intervals, use pooled variance across folds
        # Average the per-experiment std
        avg_fold_std = group['val_f1_std'].mean()

        tm_stats.append({
            'tm_weight': tm_weight,
            'val_f1_mean': f1_grand_stats['mean'],
            'val_f1_std': f1_grand_stats['std'],  # Std across experiments
            'val_f1_fold_std': avg_fold_std,  # Avg std within experiments
            'val_f1_ci_lower': f1_grand_stats['ci_lower'],
            'val_f1_ci_upper': f1_grand_stats['ci_upper'],
            'top_k_acc_3_mean': np.mean(top3_means) if len(top3_means) > 0 else None,
            'top_k_acc_3_std': np.std(top3_means) if len(top3_means) > 0 else None,
            'n_experiments': len(group),
            'n_folds_avg': group['n_folds'].mean(),
        })

    return pd.DataFrame(tm_stats).sort_values('tm_weight')


def analyze_per_model_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results per pretrained model with CV aggregation."""
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = cv_df[cv_df['complete'] == True].copy()

    results = []
    for model in complete_cv['pretrained_model'].dropna().unique():
        model_df = complete_cv[complete_cv['pretrained_model'] == model]

        if model_df.empty:
            continue

        # Best result for this model (by mean F1)
        best_idx = model_df['val_f1_mean'].idxmax()
        best = model_df.loc[best_idx]

        # Baseline (tm_weight=0)
        baseline = model_df[model_df['tm_weight'] == 0.0]
        baseline_f1 = baseline['val_f1_mean'].mean() if not baseline.empty else None

        # Relative improvement
        rel_improvement = None
        if baseline_f1 and baseline_f1 > 0:
            rel_improvement = (best['val_f1_mean'] - baseline_f1) / baseline_f1

        results.append({
            'pretrained_model': model,
            'best_f1_mean': best['val_f1_mean'],
            'best_f1_std': best['val_f1_std'],
            'best_f1_ci': f"[{best['val_f1_ci_lower']:.4f}, {best['val_f1_ci_upper']:.4f}]",
            'best_tm_weight': best['tm_weight'],
            'baseline_f1_mean': baseline_f1,
            'relative_improvement': rel_improvement,
            'top_k_acc_3': best.get('top_k_acc_3_mean'),
            'n_experiments': len(model_df),
        })

    return pd.DataFrame(results).sort_values('best_f1_mean', ascending=False)


def analyze_encoder_by_macro_f1_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze results per encoder with macro-F1 as the primary metric.

    For each encoder, finds the configuration with best macro-F1 and reports
    all metrics (W-F1, Top-3, Cum70, JS) at that configuration.

    Note: Excludes label smoothing ablation runs to focus on TM regularization.
    """
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = cv_df[cv_df['complete'] == True].copy()

    # Exclude label smoothing ablation runs (they should be in appendix, not main results)
    # This ensures the encoder comparison focuses on TM regularization effect
    complete_cv = complete_cv[~complete_cv['base_exp_name'].str.contains('label_smoothing', case=False, na=False)]

    # Short display names for encoders
    encoder_display_names = {
        'gbert_base': 'GBERT-base',
        'gbert_large': 'GBERT-large',
        'gelectra_base': 'GELECTRa-base',
        'modern_gbert_134M': 'ModernGBERT-134M',
        'modern_gbert_1B': 'ModernGBERT-1B',
        'Eurobert_210m': 'EuroBERT-210M',
        'Eurobert_610m': 'EuroBERT-610M',
    }

    results = []
    for model in complete_cv['pretrained_model'].dropna().unique():
        model_df = complete_cv[complete_cv['pretrained_model'] == model]

        if model_df.empty:
            continue

        # Filter to rows with valid macro_f1
        model_df_valid = model_df[model_df['macro_f1_mean'].notna()]
        if model_df_valid.empty:
            continue

        # Best result for this model (by macro-F1)
        best_idx = model_df_valid['macro_f1_mean'].idxmax()
        best = model_df_valid.loc[best_idx]

        # Get short encoder name
        short_name = model.split('/')[-1] if '/' in str(model) else str(model)
        display_name = encoder_display_names.get(short_name, short_name)

        # Extract best config info (model_type and context length)
        model_type = best.get('model_type', 'unknown')
        context_len = best.get('max_context_utterances')

        # Create compact config string: B/H/E for bert/history/history_easy + context
        model_type_abbrev = {'bert': 'B', 'history': 'H', 'history_easy': 'E'}.get(model_type, '?')
        if pd.notna(context_len):
            best_config = f"{model_type_abbrev}{int(context_len)}"
        else:
            best_config = model_type_abbrev

        results.append({
            'encoder': display_name,
            'encoder_key': short_name,
            'macro_f1_mean': best['macro_f1_mean'],
            'macro_f1_std': best.get('macro_f1_std', 0),
            'best_tm_weight': best['tm_weight'],
            'best_config': best_config,
            'val_f1_mean': best.get('val_f1_mean'),
            'val_f1_std': best.get('val_f1_std', 0),
            'top_k_acc_3_mean': best.get('top_k_acc_3_mean'),
            'top_k_acc_3_std': best.get('top_k_acc_3_std', 0),
            'cumul70_mean': best.get('cumul70_mean'),
            'cumul70_std': best.get('cumul70_std', 0),
            'js_divergence_mean': best.get('js_divergence_mean'),
            'js_divergence_std': best.get('js_divergence_std', 0),
        })

    return pd.DataFrame(results).sort_values('macro_f1_mean', ascending=False)


def generate_encoder_macro_f1_latex_table(encoder_stats: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for encoder results with macro-F1 as primary metric."""
    if encoder_stats.empty:
        return

    latex_lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\begin{tabular}{@{}lccccccl@{}}",
        "    \\toprule",
        "    \\textbf{Encoder} & \\textbf{Macro-F1} & \\textbf{$\\lambda$} & \\textbf{W-F1} & \\textbf{Top-3} & \\textbf{Cum70} & \\textbf{JS} & \\textbf{Cfg} \\\\",
        "    \\midrule",
    ]

    # Find best macro-F1 for bolding
    best_macro_f1 = encoder_stats['macro_f1_mean'].max()

    for _, row in encoder_stats.iterrows():
        encoder = row['encoder']
        macro_f1 = row['macro_f1_mean']
        macro_f1_std = row.get('macro_f1_std', 0) or 0
        tm_weight = row['best_tm_weight']
        best_config = row.get('best_config', '-')
        w_f1 = row.get('val_f1_mean')
        top3 = row.get('top_k_acc_3_mean')
        cum70 = row.get('cumul70_mean')
        js = row.get('js_divergence_mean')

        # Format macro-F1 with std
        macro_f1_str = f".{int(macro_f1*1000):03d}\\,{{\\scriptsize$\\pm$.{int(macro_f1_std*1000):03d}}}"
        if abs(macro_f1 - best_macro_f1) < 0.001:
            macro_f1_str = "\\textbf{" + macro_f1_str + "}"

        # Format other metrics
        w_f1_str = f".{int(w_f1*1000):03d}" if pd.notna(w_f1) else "-"
        top3_str = f".{int(top3*1000):03d}" if pd.notna(top3) else "-"
        cum70_str = f".{int(cum70*1000):03d}" if pd.notna(cum70) else "-"
        js_str = f".{int(js*1000):03d}" if pd.notna(js) else "-"

        latex_lines.append(
            f"    {encoder} & {macro_f1_str} & {tm_weight:.1f} & {w_f1_str} & {top3_str} & {cum70_str} & {js_str} & {best_config} \\\\"
        )

    # Add mean row (no config for mean)
    mean_macro = encoder_stats['macro_f1_mean'].mean()
    mean_wf1 = encoder_stats['val_f1_mean'].mean()
    mean_top3 = encoder_stats['top_k_acc_3_mean'].mean()
    mean_cum70 = encoder_stats['cumul70_mean'].mean()
    mean_js = encoder_stats['js_divergence_mean'].mean()

    latex_lines.append("    \\midrule")
    latex_lines.append(
        f"    \\textit{{Mean}} & .{int(mean_macro*1000):03d} & -- & .{int(mean_wf1*1000):03d} & "
        f".{int(mean_top3*1000):03d} & .{int(mean_cum70*1000):03d} & .{int(mean_js*1000):03d} & -- \\\\"
    )

    latex_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Results by encoder (60 categories, 5-fold CV). Best configuration selected by macro-F1. Cfg = best architecture (B=BERT, H=History) + context length.}",
        "  \\label{tab:encoder_results}",
        "\\end{table}",
    ])

    output_path = output_dir / 'encoder_results_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved LaTeX table: {output_path}")


def analyze_by_model_type_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze results by model_type (bert vs history) with CV aggregation.

    Returns statistics for each (model_type, tm_weight) combination.
    """
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = cv_df[cv_df['complete'] == True].copy()

    results = []
    for model_type in complete_cv['model_type'].dropna().unique():
        type_df = complete_cv[complete_cv['model_type'] == model_type]

        for tm_weight in sorted(type_df['tm_weight'].dropna().unique()):
            tw_df = type_df[type_df['tm_weight'] == tm_weight]
            if len(tw_df) == 0:
                continue

            # Weighted F1
            f1_vals = tw_df['val_f1_mean'].values
            f1_stats = compute_cv_statistics(f1_vals) if len(f1_vals) > 1 else {
                'mean': f1_vals[0] if len(f1_vals) > 0 else None,
                'std': 0, 'se': 0, 'ci_lower': f1_vals[0] if len(f1_vals) > 0 else None,
                'ci_upper': f1_vals[0] if len(f1_vals) > 0 else None
            }

            # Macro F1
            macro_vals = tw_df['macro_f1_mean'].dropna().values
            macro_stats = compute_cv_statistics(macro_vals) if len(macro_vals) > 1 else {
                'mean': macro_vals[0] if len(macro_vals) > 0 else None,
                'std': 0
            }

            # Top-3 accuracy
            top3_vals = tw_df['top_k_acc_3_mean'].dropna().values
            top3_stats = compute_cv_statistics(top3_vals) if len(top3_vals) > 1 else {
                'mean': top3_vals[0] if len(top3_vals) > 0 else None,
                'std': 0
            }

            # Cumul@70
            cumul70_vals = tw_df['cumul70_mean'].dropna().values
            cumul70_stats = compute_cv_statistics(cumul70_vals) if len(cumul70_vals) > 1 else {
                'mean': cumul70_vals[0] if len(cumul70_vals) > 0 else None,
                'std': 0
            }

            # JS divergence
            js_vals = tw_df['js_divergence_mean'].dropna().values
            js_stats = compute_cv_statistics(js_vals) if len(js_vals) > 1 else {
                'mean': js_vals[0] if len(js_vals) > 0 else None,
                'std': 0
            }

            results.append({
                'model_type': model_type,
                'tm_weight': tm_weight,
                'val_f1_mean': f1_stats.get('mean'),
                'val_f1_std': f1_stats.get('std'),
                'macro_f1_mean': macro_stats.get('mean'),
                'macro_f1_std': macro_stats.get('std'),
                'top_k_acc_3_mean': top3_stats.get('mean'),
                'top_k_acc_3_std': top3_stats.get('std'),
                'cumul70_mean': cumul70_stats.get('mean'),
                'cumul70_std': cumul70_stats.get('std'),
                'js_divergence_mean': js_stats.get('mean'),
                'js_divergence_std': js_stats.get('std'),
                'n_experiments': len(tw_df),
            })

    return pd.DataFrame(results).sort_values(['model_type', 'tm_weight'])


def analyze_tm_effect_all_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze effect of tm_weight on F1 with CV aggregation, including both model types.

    Returns statistics for each tm_weight, averaging across all experiments
    (both bert and history model types).
    """
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = cv_df[cv_df['complete'] == True].copy()
    print(f"Complete CV experiments (all model types): {len(complete_cv)}")

    # Group by tm_weight
    tm_stats = []
    for tm_weight, group in complete_cv.groupby('tm_weight'):
        # Weighted F1
        f1_means = group['val_f1_mean'].values
        f1_grand_stats = compute_cv_statistics(f1_means) if len(f1_means) > 1 else {}

        # Macro F1
        macro_means = group['macro_f1_mean'].dropna().values
        macro_stats = compute_cv_statistics(macro_means) if len(macro_means) > 1 else {}

        # Top-3 accuracy
        top3_means = group['top_k_acc_3_mean'].dropna().values
        top3_stats = compute_cv_statistics(top3_means) if len(top3_means) > 1 else {}

        # Cumul@70
        cumul70_means = group['cumul70_mean'].dropna().values
        cumul70_stats = compute_cv_statistics(cumul70_means) if len(cumul70_means) > 1 else {}

        # JS divergence
        js_means = group['js_divergence_mean'].dropna().values
        js_stats = compute_cv_statistics(js_means) if len(js_means) > 1 else {}

        # Average the per-experiment std (fold std)
        avg_fold_std = group['val_f1_std'].mean()

        tm_stats.append({
            'tm_weight': tm_weight,
            'val_f1_mean': f1_grand_stats.get('mean', np.mean(f1_means) if len(f1_means) > 0 else None),
            'val_f1_std': f1_grand_stats.get('std', 0),
            'val_f1_fold_std': avg_fold_std,
            'val_f1_ci_lower': f1_grand_stats.get('ci_lower'),
            'val_f1_ci_upper': f1_grand_stats.get('ci_upper'),
            'macro_f1_mean': macro_stats.get('mean', np.mean(macro_means) if len(macro_means) > 0 else None),
            'macro_f1_std': macro_stats.get('std', 0),
            'top_k_acc_3_mean': top3_stats.get('mean', np.mean(top3_means) if len(top3_means) > 0 else None),
            'top_k_acc_3_std': top3_stats.get('std', 0),
            'cumul70_mean': cumul70_stats.get('mean', np.mean(cumul70_means) if len(cumul70_means) > 0 else None),
            'cumul70_std': cumul70_stats.get('std', 0),
            'js_divergence_mean': js_stats.get('mean', np.mean(js_means) if len(js_means) > 0 else None),
            'js_divergence_std': js_stats.get('std', 0),
            'n_experiments': len(group),
            'n_folds_avg': group['n_folds'].mean(),
        })

    return pd.DataFrame(tm_stats).sort_values('tm_weight')


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> dict:
    """
    Perform paired bootstrap significance test.

    Tests H0: mean(scores_a) = mean(scores_b) using paired bootstrap resampling.

    Args:
        scores_a: Scores from condition A (e.g., baseline)
        scores_b: Scores from condition B (e.g., with TM regularization)
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed for reproducibility

    Returns:
        Dict with observed_diff, p_value, ci_lower, ci_upper, significant
    """
    np.random.seed(random_state)

    n = len(scores_a)
    assert len(scores_b) == n, "Arrays must have same length for paired test"

    # Observed difference
    observed_diff = np.mean(scores_b) - np.mean(scores_a)

    # Paired differences
    paired_diffs = scores_b - scores_a

    # Bootstrap the paired differences
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = paired_diffs[indices]
        bootstrap_diffs.append(np.mean(bootstrap_sample))

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-tailed p-value: proportion of bootstrap samples with opposite sign
    # or more extreme than observed
    if observed_diff >= 0:
        p_value = np.mean(bootstrap_diffs <= 0) * 2
    else:
        p_value = np.mean(bootstrap_diffs >= 0) * 2
    p_value = min(p_value, 1.0)

    # 95% confidence interval for the difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
    }


def get_fold_scores(df: pd.DataFrame, base_exp_name: str, metric: str = 'class_macro_avg_f1') -> np.ndarray:
    """
    Get per-fold scores for a given experiment.

    Args:
        df: Raw wandb export DataFrame with fold-level results
        base_exp_name: Base experiment name (without _foldN suffix)
        metric: Metric column to extract

    Returns:
        Array of scores for each fold
    """
    name_col = 'run_name' if 'run_name' in df.columns else 'Name'
    finished = df[df['state'] == 'finished'].copy()

    # Find runs matching this experiment
    pattern = f"^{re.escape(base_exp_name)}_fold\\d+$"
    matching = finished[finished[name_col].str.match(pattern, na=False)]

    if matching.empty:
        return np.array([])

    # Extract fold number and sort
    matching['fold'] = matching[name_col].apply(extract_fold_from_name)
    matching = matching.sort_values('fold')

    return matching[metric].dropna().values


def run_significance_tests(df: pd.DataFrame, tm_weights_to_test: list = None) -> pd.DataFrame:
    """
    Run paired bootstrap significance tests for key comparisons.

    Key comparisons:
    1. λ=0.0 vs λ=0.2, λ=0.5, λ=1.0, λ=1.5 (TM effect at multiple λ values)
    2. BERT vs History architecture (at optimal λ)

    Excludes: label_smoothing runs, history_easy architecture

    Args:
        df: Raw wandb export DataFrame
        tm_weights_to_test: List of λ values to test against baseline (default: [0.2, 0.5, 1.0, 1.5])

    Returns:
        DataFrame with test results for each comparison
    """
    if tm_weights_to_test is None:
        tm_weights_to_test = [0.2, 0.5, 1.0, 1.5]

    results = []
    finished = df[df['state'] == 'finished'].copy()
    name_col = 'run_name' if 'run_name' in finished.columns else 'Name'

    # Extract experiment info
    finished['fold'] = finished[name_col].apply(extract_fold_from_name)
    finished['base_exp_name'] = finished[name_col].apply(extract_base_experiment_name)

    # Filter to fold-annotated runs
    with_folds = finished[finished['fold'] >= 0].copy()

    # Exclude label smoothing runs
    with_folds = with_folds[~with_folds['base_exp_name'].str.contains('label_smoothing', case=False, na=False)]

    # Exclude history_easy model type - only keep bert and history
    with_folds = with_folds[with_folds['model_type'].isin(['bert', 'history'])]

    print(f"\nFiltered to {len(with_folds)} runs (excluding label_smoothing and history_easy)")

    # Extract context from run name
    with_folds['context_len'] = with_folds['base_exp_name'].apply(extract_context_from_name)

    # Group by base experiment name to count folds and get metadata
    exp_fold_counts = with_folds.groupby('base_exp_name').agg({
        'fold': 'count',
        'pretrained_model': 'first',
        'model_type': 'first',
        'tm_weight': 'first',
        'context_len': 'first',
        'class_macro_avg_f1': 'mean'
    }).reset_index()

    # Only keep complete experiments (5 folds)
    complete_exps = exp_fold_counts[exp_fold_counts['fold'] == N_FOLDS].copy()
    print(f"Found {len(complete_exps)} complete 5-fold experiments")

    # Test 1: TM regularization effect (λ=0.0 vs λ=X) for each encoder/model_type/context combo
    # Test against multiple λ values to show consistent improvement
    for tm_weight in tm_weights_to_test:
        print(f"\n--- Test: TM Regularization Effect (λ=0.0 vs λ={tm_weight}) ---")
        for model_type in ['bert', 'history']:
            for encoder in complete_exps['pretrained_model'].unique():
                for ctx in complete_exps['context_len'].dropna().unique():
                    baseline_exps = complete_exps[
                        (complete_exps['pretrained_model'] == encoder) &
                        (complete_exps['model_type'] == model_type) &
                        (complete_exps['tm_weight'] == 0.0) &
                        (complete_exps['context_len'] == ctx)
                    ]
                    tm_exps = complete_exps[
                        (complete_exps['pretrained_model'] == encoder) &
                        (complete_exps['model_type'] == model_type) &
                        (complete_exps['tm_weight'] == tm_weight) &
                        (complete_exps['context_len'] == ctx)
                    ]

                    if baseline_exps.empty or tm_exps.empty:
                        continue

                    baseline_name = baseline_exps['base_exp_name'].iloc[0]
                    tm_name = tm_exps['base_exp_name'].iloc[0]

                    baseline_scores = get_fold_scores(df, baseline_name)
                    tm_scores = get_fold_scores(df, tm_name)

                    if len(baseline_scores) == N_FOLDS and len(tm_scores) == N_FOLDS:
                        test_result = paired_bootstrap_test(baseline_scores, tm_scores)

                        short_encoder = encoder.split('/')[-1] if '/' in str(encoder) else str(encoder)
                        results.append({
                            'comparison': f'TM_effect_0.0_vs_{tm_weight}',
                            'tm_weight_tested': tm_weight,
                            'encoder': short_encoder,
                            'model_type': model_type,
                            'context': int(ctx),
                            'condition_a': 'λ=0.0',
                            'condition_b': f'λ={tm_weight}',
                            'mean_a': np.mean(baseline_scores),
                            'mean_b': np.mean(tm_scores),
                            'diff': test_result['observed_diff'],
                            'ci_lower': test_result['ci_lower'],
                            'ci_upper': test_result['ci_upper'],
                            'p_value': test_result['p_value'],
                            'significant_05': test_result['significant_05'],
                            'significant_01': test_result['significant_01'],
                        })

    # Test 2: Architecture effect (BERT vs History) at λ=0.5
    print("\n--- Test: Architecture Effect (BERT vs History at λ=0.5) ---")
    for encoder in complete_exps['pretrained_model'].unique():
        for ctx in complete_exps['context_len'].dropna().unique():
            bert_exps = complete_exps[
                (complete_exps['pretrained_model'] == encoder) &
                (complete_exps['model_type'] == 'bert') &
                (complete_exps['tm_weight'] == 0.5) &
                (complete_exps['context_len'] == ctx)
            ]
            history_exps = complete_exps[
                (complete_exps['pretrained_model'] == encoder) &
                (complete_exps['model_type'] == 'history') &
                (complete_exps['tm_weight'] == 0.5) &
                (complete_exps['context_len'] == ctx)
            ]

            if bert_exps.empty or history_exps.empty:
                continue

            bert_name = bert_exps['base_exp_name'].iloc[0]
            history_name = history_exps['base_exp_name'].iloc[0]

            bert_scores = get_fold_scores(df, bert_name)
            history_scores = get_fold_scores(df, history_name)

            if len(bert_scores) == N_FOLDS and len(history_scores) == N_FOLDS:
                test_result = paired_bootstrap_test(bert_scores, history_scores)

                short_encoder = encoder.split('/')[-1] if '/' in str(encoder) else str(encoder)
                results.append({
                    'comparison': 'Arch_BERT_vs_History',
                    'encoder': short_encoder,
                    'model_type': 'comparison',
                    'context': int(ctx),
                    'condition_a': 'BERT',
                    'condition_b': 'History',
                    'mean_a': np.mean(bert_scores),
                    'mean_b': np.mean(history_scores),
                    'diff': test_result['observed_diff'],
                    'ci_lower': test_result['ci_lower'],
                    'ci_upper': test_result['ci_upper'],
                    'p_value': test_result['p_value'],
                    'significant_05': test_result['significant_05'],
                    'significant_01': test_result['significant_01'],
                })

    return pd.DataFrame(results)


def apply_multiple_testing_correction(sig_results: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multiple testing corrections (Bonferroni and Benjamini-Hochberg FDR).

    Args:
        sig_results: DataFrame with p_value column

    Returns:
        DataFrame with additional columns for corrected p-values and significance
    """
    if sig_results.empty:
        return sig_results

    results = sig_results.copy()

    # Apply corrections per comparison type
    for comparison in results['comparison'].unique():
        mask = results['comparison'] == comparison
        p_values = results.loc[mask, 'p_value'].values
        n_tests = len(p_values)

        # Bonferroni correction
        bonferroni_p = np.minimum(p_values * n_tests, 1.0)
        results.loc[mask, 'p_bonferroni'] = bonferroni_p
        results.loc[mask, 'sig_bonferroni_05'] = bonferroni_p < 0.05
        results.loc[mask, 'sig_bonferroni_01'] = bonferroni_p < 0.01

        # Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        ranks = np.arange(1, n_tests + 1)

        # BH adjusted p-values
        bh_adjusted = np.minimum.accumulate((sorted_p * n_tests / ranks)[::-1])[::-1]
        bh_adjusted = np.minimum(bh_adjusted, 1.0)

        # Map back to original order
        bh_p = np.empty(n_tests)
        bh_p[sorted_indices] = bh_adjusted

        results.loc[mask, 'p_fdr'] = bh_p
        results.loc[mask, 'sig_fdr_05'] = bh_p < 0.05
        results.loc[mask, 'sig_fdr_01'] = bh_p < 0.01

    return results


def print_significance_summary(sig_results: pd.DataFrame):
    """Print a summary of significance test results with multiple testing corrections."""
    if sig_results.empty:
        print("No significance test results to summarize.")
        return

    # Apply multiple testing corrections
    results = apply_multiple_testing_correction(sig_results)

    print("\n" + "="*80)
    print("SIGNIFICANCE TEST RESULTS (Paired Bootstrap, 10000 iterations)")
    print("="*80)

    # Group by comparison type
    for comparison in results['comparison'].unique():
        subset = results[results['comparison'] == comparison]
        print(f"\n--- {comparison} ---")

        # Summary statistics
        n_tests = len(subset)
        n_sig_05 = subset['significant_05'].sum()
        n_sig_01 = subset['significant_01'].sum()
        n_sig_bonf_05 = subset['sig_bonferroni_05'].sum()
        n_sig_fdr_05 = subset['sig_fdr_05'].sum()
        mean_diff = subset['diff'].mean()
        mean_p = subset['p_value'].mean()

        print(f"  Tests run: {n_tests}")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Mean p-value: {mean_p:.4f}")
        print(f"\n  Uncorrected:")
        print(f"    Significant at p<0.05: {n_sig_05}/{n_tests} ({100*n_sig_05/n_tests:.0f}%)")
        print(f"    Significant at p<0.01: {n_sig_01}/{n_tests} ({100*n_sig_01/n_tests:.0f}%)")
        print(f"\n  Bonferroni corrected (α/{n_tests}):")
        print(f"    Significant at p<0.05: {n_sig_bonf_05}/{n_tests} ({100*n_sig_bonf_05/n_tests:.0f}%)")
        print(f"\n  Benjamini-Hochberg FDR corrected:")
        print(f"    Significant at FDR<0.05: {n_sig_fdr_05}/{n_tests} ({100*n_sig_fdr_05/n_tests:.0f}%)")

        # Show individual results with corrections
        print(f"\n  {'Encoder':<18} {'A→B':<12} {'Diff':>7} {'p':>7} {'p_BH':>7} {'p_Bonf':>8} {'Sig'}")
        print("  " + "-"*75)
        for _, row in subset.iterrows():
            # Significance markers based on FDR-corrected p-values
            if row['sig_fdr_01']:
                sig_marker = "***"  # FDR < 0.01
            elif row['sig_fdr_05']:
                sig_marker = "**"   # FDR < 0.05
            elif row['significant_05']:
                sig_marker = "*"    # uncorrected < 0.05
            else:
                sig_marker = ""
            direction = f"{row['condition_a']}→{row['condition_b']}"
            print(f"  {row['encoder']:<18} {direction:<12} {row['diff']:>+.4f} {row['p_value']:>7.4f} {row['p_fdr']:>7.4f} {row['p_bonferroni']:>8.4f} {sig_marker}")

    return results


def summarize_significance_by_lambda(sig_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize significance test results by λ value.

    Shows how many comparisons are significant at each λ value to demonstrate
    consistent improvement across regularization strengths.
    """
    if sig_results.empty:
        return pd.DataFrame()

    # Ensure corrections are applied
    if 'p_fdr' not in sig_results.columns:
        sig_results = apply_multiple_testing_correction(sig_results)

    # Only TM effect comparisons
    tm_results = sig_results[sig_results['comparison'].str.startswith('TM_effect')].copy()

    if tm_results.empty:
        return pd.DataFrame()

    summary = []
    for tm_weight in sorted(tm_results['tm_weight_tested'].unique()):
        subset = tm_results[tm_results['tm_weight_tested'] == tm_weight]
        n_tests = len(subset)
        n_positive = (subset['diff'] > 0).sum()
        n_sig_fdr = subset['sig_fdr_05'].sum()
        n_sig_01 = subset['sig_fdr_01'].sum()
        mean_diff = subset['diff'].mean()
        median_diff = subset['diff'].median()

        summary.append({
            'tm_weight': tm_weight,
            'n_comparisons': n_tests,
            'n_positive': n_positive,
            'pct_positive': 100 * n_positive / n_tests if n_tests > 0 else 0,
            'n_sig_fdr_05': n_sig_fdr,
            'pct_sig_fdr_05': 100 * n_sig_fdr / n_tests if n_tests > 0 else 0,
            'n_sig_fdr_01': n_sig_01,
            'pct_sig_fdr_01': 100 * n_sig_01 / n_tests if n_tests > 0 else 0,
            'mean_diff': mean_diff,
            'median_diff': median_diff,
        })

    return pd.DataFrame(summary)


def print_lambda_summary(sig_results: pd.DataFrame):
    """Print a summary of significance results by λ value."""
    summary = summarize_significance_by_lambda(sig_results)

    if summary.empty:
        print("No λ summary to print.")
        return

    print("\n" + "="*80)
    print("SIGNIFICANCE BY λ VALUE (TM Regularization Effect)")
    print("="*80)
    print("\n{:<8} {:>6} {:>8} {:>10} {:>10} {:>12} {:>12}".format(
        "λ", "Tests", "Pos%", "FDR<.05", "FDR<.01", "Mean Δ", "Median Δ"))
    print("-"*80)

    for _, row in summary.iterrows():
        print("{:<8.1f} {:>6d} {:>7.0f}% {:>9.0f}% {:>9.0f}% {:>+11.4f} {:>+11.4f}".format(
            row['tm_weight'],
            int(row['n_comparisons']),
            row['pct_positive'],
            row['pct_sig_fdr_05'],
            row['pct_sig_fdr_01'],
            row['mean_diff'],
            row['median_diff']
        ))

    # Overall interpretation
    print("\n" + "-"*80)
    all_positive_pct = summary['pct_positive'].mean()
    all_sig_pct = summary['pct_sig_fdr_05'].mean()
    print(f"Overall: {all_positive_pct:.0f}% of comparisons show positive effect across all λ values")
    print(f"         {all_sig_pct:.0f}% are significant after FDR correction on average")

    return summary


def generate_significance_latex_table(sig_results: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for significance test results with FDR correction."""
    if sig_results.empty:
        return

    # Ensure corrections are applied
    if 'p_fdr' not in sig_results.columns:
        sig_results = apply_multiple_testing_correction(sig_results)

    # Focus on TM effect comparison - now handle multiple λ values
    tm_results = sig_results[sig_results['comparison'].str.startswith('TM_effect')].copy()

    if tm_results.empty:
        return

    # Generate summary table by λ value
    summary = summarize_significance_by_lambda(sig_results)
    if not summary.empty:
        latex_summary = [
            "\\begin{table}[h]",
            "\\centering",
            "\\footnotesize",
            "\\begin{tabular}{rrrrrrrr}",
            "\\toprule",
            "\\textbf{$\\lambda$} & \\textbf{Tests} & \\textbf{Pos.} & \\textbf{\\%Pos.} & \\textbf{Sig.(FDR)} & \\textbf{\\%Sig.} & \\textbf{Mean $\\Delta$} & \\textbf{Median $\\Delta$} \\\\",
            "\\midrule",
        ]

        for _, row in summary.iterrows():
            latex_summary.append(
                f"    {row['tm_weight']:.1f} & {int(row['n_comparisons'])} & {int(row['n_positive'])} & "
                f"{row['pct_positive']:.0f}\\% & {int(row['n_sig_fdr_05'])} & {row['pct_sig_fdr_05']:.0f}\\% & "
                f"{row['mean_diff']:+.4f} & {row['median_diff']:+.4f} \\\\"
            )

        latex_summary.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Summary of TM regularization significance tests by $\\lambda$ value. Tests: number of paired comparisons (encoder × architecture × context). Pos.: positive effect (improvement over $\\lambda$=0). Sig.(FDR): significant after Benjamini-Hochberg correction at $\\alpha$=0.05.}",
            "\\label{tab:significance_by_lambda}",
            "\\end{table}",
        ])

        output_path = output_dir / 'significance_by_lambda_table.tex'
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex_summary))
        print(f"\nSaved λ summary LaTeX table: {output_path}")

    # Also generate detailed table for λ=0.5 (for backward compatibility)
    tm_05_results = tm_results[tm_results['tm_weight_tested'] == 0.5].copy()

    if not tm_05_results.empty:
        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\footnotesize",
            "\\begin{tabular}{llcccccc}",
            "\\toprule",
            "\\textbf{Encoder} & \\textbf{Arch} & \\textbf{$\\lambda$=0} & \\textbf{$\\lambda$=0.5} & \\textbf{$\\Delta$} & \\textbf{95\\% CI} & \\textbf{p} & \\textbf{$p_{\\text{FDR}}$} \\\\",
            "\\midrule",
        ]

        for _, row in tm_05_results.iterrows():
            encoder = row['encoder']
            arch = row['model_type'].capitalize()
            mean_a = f".{int(row['mean_a']*1000):03d}"
            mean_b = f".{int(row['mean_b']*1000):03d}"
            diff = f"+.{int(row['diff']*1000):03d}" if row['diff'] >= 0 else f"-.{int(abs(row['diff'])*1000):03d}"
            ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            p_val = f"{row['p_value']:.3f}"
            p_fdr = f"{row['p_fdr']:.3f}"

            # Add significance markers based on FDR-corrected p-values
            if row['sig_fdr_01']:
                p_fdr += "$^{***}$"
            elif row['sig_fdr_05']:
                p_fdr += "$^{**}$"
            elif row['significant_05']:
                p_fdr += "$^{*}$"

            latex_lines.append(f"    {encoder} & {arch} & {mean_a} & {mean_b} & {diff} & {ci} & {p_val} & {p_fdr} \\\\")

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Paired bootstrap significance tests for TM regularization effect ($\\lambda$=0 vs $\\lambda$=0.5). Macro-F1 shown. $p_{\\text{FDR}}$: Benjamini-Hochberg corrected. $^{*}$uncorrected $p<0.05$, $^{**}$FDR$<0.05$, $^{***}$FDR$<0.01$.}",
            "\\label{tab:significance_tm}",
            "\\end{table}",
        ])

        output_path = output_dir / 'significance_tests_table.tex'
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"\nSaved LaTeX table: {output_path}")


def compare_with_client_28_cv(tm_stats: pd.DataFrame):
    """Compare full_categories CV results with client_categories (28 labels)."""
    print("\n" + "="*70)
    print("COMPARISON: Full Categories (60) vs Client-Only (28) - 5-Fold CV")
    print("="*70)

    if tm_stats.empty:
        print("No data to compare!")
        return

    print("\n{:<10} {:>14} {:>20} {:>12}".format(
        "tm_weight", "Full(60)", "95% CI", "Client(28)"))
    print("-"*60)

    for _, row in tm_stats.iterrows():
        tw = row['tm_weight']
        full_f1 = row['val_f1_mean']
        ci = f"[{row['val_f1_ci_lower']:.4f}, {row['val_f1_ci_upper']:.4f}]"
        client_f1 = CLIENT_28_RESULTS['tm_weight_means'].get(tw, None)

        if client_f1:
            print(f"{tw:<10.1f} {full_f1:>14.4f} {ci:>20} {client_f1:>12.4f}")
        else:
            print(f"{tw:<10.1f} {full_f1:>14.4f} {ci:>20} {'N/A':>12}")

    # Compute key comparisons
    if 0.0 in tm_stats['tm_weight'].values and 0.5 in tm_stats['tm_weight'].values:
        full_baseline = tm_stats[tm_stats['tm_weight'] == 0.0]['val_f1_mean'].iloc[0]
        full_optimal = tm_stats[tm_stats['tm_weight'] == 0.5]['val_f1_mean'].iloc[0]
        full_improvement = (full_optimal - full_baseline) / full_baseline

        print(f"\nRelative improvement (lambda=0 -> lambda=0.5):")
        print(f"  Full (60 labels):   {full_improvement:+.1%}")
        print(f"  Client (28 labels): {CLIENT_28_RESULTS['relative_improvement']:+.1%}")

    # Find optimal lambda for full categories
    best_row = tm_stats.loc[tm_stats['val_f1_mean'].idxmax()]
    print(f"\nOptimal lambda:")
    print(f"  Full (60 labels):   {best_row['tm_weight']:.1f} (F1={best_row['val_f1_mean']:.4f} +/- {best_row['val_f1_std']:.4f})")
    print(f"  Client (28 labels): {CLIENT_28_RESULTS['optimal_lambda']:.1f} (F1={CLIENT_28_RESULTS['tm_weight_means'][0.5]:.4f})")


def plot_tm_effect_cv(tm_stats: pd.DataFrame, output_dir: Path):
    """Plot tm_weight effect with confidence intervals."""
    if tm_stats.empty:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Full categories tm_weight effect with CI
    ax1 = axes[0]

    # Plot with confidence interval band
    ax1.fill_between(
        tm_stats['tm_weight'],
        tm_stats['val_f1_ci_lower'],
        tm_stats['val_f1_ci_upper'],
        alpha=0.3, color='steelblue', label='95% CI'
    )
    ax1.plot(tm_stats['tm_weight'], tm_stats['val_f1_mean'],
             'o-', color='steelblue', linewidth=2, markersize=8, label='Full (60 labels)')

    # Add error bars for fold std
    ax1.errorbar(
        tm_stats['tm_weight'], tm_stats['val_f1_mean'],
        yerr=tm_stats['val_f1_fold_std'],
        fmt='none', color='steelblue', capsize=4, alpha=0.7
    )

    # Add client_28 reference
    client_tws = sorted(CLIENT_28_RESULTS['tm_weight_means'].keys())
    client_f1s = [CLIENT_28_RESULTS['tm_weight_means'][tw] for tw in client_tws]
    ax1.plot(client_tws, client_f1s, 's--', color='gray', alpha=0.7,
             markersize=6, label='Client (28 labels)')

    ax1.set_xlabel('Transition Loss Weight ($\\lambda$)', fontsize=12)
    ax1.set_ylabel('Weighted F1', fontsize=12)
    ax1.set_title('TM Regularization Effect (5-Fold CV)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative improvement from baseline with CI
    ax2 = axes[1]
    baseline_f1 = tm_stats[tm_stats['tm_weight'] == 0.0]['val_f1_mean'].iloc[0] if 0.0 in tm_stats['tm_weight'].values else tm_stats['val_f1_mean'].iloc[0]
    baseline_ci_lower = tm_stats[tm_stats['tm_weight'] == 0.0]['val_f1_ci_lower'].iloc[0] if 0.0 in tm_stats['tm_weight'].values else 0
    baseline_ci_upper = tm_stats[tm_stats['tm_weight'] == 0.0]['val_f1_ci_upper'].iloc[0] if 0.0 in tm_stats['tm_weight'].values else 0

    rel_improvements = (tm_stats['val_f1_mean'] - baseline_f1) / baseline_f1 * 100
    # Propagate uncertainty to relative improvement
    rel_ci_lower = (tm_stats['val_f1_ci_lower'] - baseline_f1) / baseline_f1 * 100
    rel_ci_upper = (tm_stats['val_f1_ci_upper'] - baseline_f1) / baseline_f1 * 100

    # Bar plot with error bars
    bars = ax2.bar(tm_stats['tm_weight'].astype(str), rel_improvements,
                   color='steelblue', alpha=0.7, edgecolor='navy')

    # Add CI as error bars
    yerr_lower = rel_improvements - rel_ci_lower
    yerr_upper = rel_ci_upper - rel_improvements
    ax2.errorbar(range(len(tm_stats)), rel_improvements,
                 yerr=[yerr_lower, yerr_upper],
                 fmt='none', color='navy', capsize=4)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Transition Loss Weight ($\\lambda$)', fontsize=12)
    ax2.set_ylabel('Relative Improvement (%)', fontsize=12)
    ax2.set_title('Improvement vs Baseline ($\\lambda=0$) with 95% CI', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'full_categories_cv_tm_effect.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    plt.close()


def plot_fold_variability(df: pd.DataFrame, output_dir: Path):
    """Plot variability across folds for different experiments."""
    finished = df[df['state'] == 'finished'].copy()
    name_col = 'run_name' if 'run_name' in finished.columns else 'Name'
    finished['fold'] = finished[name_col].apply(extract_fold_from_name)
    finished['base_exp_name'] = finished[name_col].apply(extract_base_experiment_name)

    with_folds = finished[finished['fold'] >= 0].copy()

    if with_folds.empty:
        print("No data with fold info to plot variability!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Box plot of F1 per fold
    ax1 = axes[0]
    fold_data = [with_folds[with_folds['fold'] == f]['val_f1'].dropna() for f in range(N_FOLDS)]
    bp = ax1.boxplot(fold_data, labels=[f'Fold {i}' for i in range(N_FOLDS)],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Weighted F1', fontsize=12)
    ax1.set_title('F1 Distribution per Fold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Fold variance distribution
    ax2 = axes[1]

    # Compute per-experiment fold variance
    variances = []
    for base_name, group in with_folds.groupby('base_exp_name'):
        if len(group) >= 2:
            var = group['val_f1'].var()
            if not np.isnan(var):
                variances.append(var)

    if variances:
        ax2.hist(variances, bins=30, color='steelblue', alpha=0.7, edgecolor='navy')
        ax2.axvline(np.mean(variances), color='red', linestyle='--',
                    label=f'Mean: {np.mean(variances):.6f}')
        ax2.axvline(np.median(variances), color='orange', linestyle='--',
                    label=f'Median: {np.median(variances):.6f}')
        ax2.set_xlabel('Variance in F1 across Folds', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Cross-Fold Variance', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'full_categories_cv_fold_variability.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def compute_metrics_by_architecture_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean metrics per architecture and TM weight with CV aggregation.

    First aggregates across folds for each experiment, then computes
    mean and CI across experiments for each (architecture, tm_weight) combination.
    """
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = cv_df[cv_df['complete'] == True].copy()

    # Extract architecture from pretrained_model
    def get_arch_name(model_path):
        if pd.isna(model_path):
            return 'unknown'
        # Extract just the model name from path
        name = model_path.split('/')[-1] if '/' in str(model_path) else str(model_path)
        return name

    complete_cv['architecture'] = complete_cv['pretrained_model'].apply(get_arch_name)

    results = []
    for arch in complete_cv['architecture'].unique():
        arch_df = complete_cv[complete_cv['architecture'] == arch]

        for tw in sorted(arch_df['tm_weight'].dropna().unique()):
            tw_df = arch_df[arch_df['tm_weight'] == tw]
            if len(tw_df) == 0:
                continue

            # Get mean and CI for each metric
            weighted_f1_vals = tw_df['val_f1_mean'].values  # Already CV-aggregated means

            # Macro F1 values (class_macro_avg_f1)
            macro_f1_vals = tw_df['macro_f1_mean'].dropna().values if 'macro_f1_mean' in tw_df.columns else np.array([])

            # JS divergence values
            js_vals = tw_df['js_divergence_mean'].dropna().values if 'js_divergence_mean' in tw_df.columns else np.array([])

            results.append({
                'architecture': arch,
                'tm_weight': tw,
                'val_f1_mean': np.mean(weighted_f1_vals),
                'val_f1_std': np.std(weighted_f1_vals, ddof=1) if len(weighted_f1_vals) > 1 else 0,
                'val_f1_fold_std': tw_df['val_f1_std'].mean(),  # Average fold std
                'macro_f1_mean': np.mean(macro_f1_vals) if len(macro_f1_vals) > 0 else None,
                'macro_f1_fold_std': tw_df['macro_f1_std'].mean() if 'macro_f1_std' in tw_df.columns else None,
                'top_k_acc_3_mean': tw_df['top_k_acc_3_mean'].mean(),
                'val_accuracy_mean': tw_df['val_accuracy_mean'].mean(),
                'js_divergence_mean': np.mean(js_vals) if len(js_vals) > 0 else None,
                'cumul70_mean': tw_df['cumul70_mean'].mean() if 'cumul70_mean' in tw_df.columns else None,
                'cumul70_fold_std': tw_df['cumul70_std'].mean() if 'cumul70_std' in tw_df.columns else None,
                'n_experiments': len(tw_df),
            })

    return pd.DataFrame(results)


def compute_relative_effects_cv(df: pd.DataFrame) -> dict:
    """Compute relative effect sizes as percentages using CV-aggregated metrics."""
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return {'λ_tm': 0, 'Arch': 0, 'optimal_tw': 0.5}

    complete_cv = cv_df[cv_df['complete'] == True].copy()

    # Extract architecture
    def get_arch_name(model_path):
        if pd.isna(model_path):
            return 'unknown'
        name = model_path.split('/')[-1] if '/' in str(model_path) else str(model_path)
        return name

    complete_cv['architecture'] = complete_cv['pretrained_model'].apply(get_arch_name)

    # λ_tm effect: mean improvement from λ=0 to best λ across all architectures
    tm_improvements = []

    for arch in complete_cv['architecture'].unique():
        arch_df = complete_cv[complete_cv['architecture'] == arch]
        arch_by_tw = arch_df.groupby('tm_weight')['val_f1_mean'].mean()

        if len(arch_by_tw) > 0 and 0.0 in arch_by_tw.index:
            baseline = arch_by_tw[0.0]
            best_val = arch_by_tw.max()

            if baseline > 0:
                improvement = ((best_val - baseline) / baseline) * 100
                tm_improvements.append(improvement)

    tm_effect_pct = np.mean(tm_improvements) if tm_improvements else 0

    # Architecture effect: spread between best and worst architecture at optimal λ
    overall_by_tw = complete_cv.groupby('tm_weight')['val_f1_mean'].mean()
    optimal_tw = overall_by_tw.idxmax() if len(overall_by_tw) > 0 else 0.5

    arch_at_optimal = complete_cv[complete_cv['tm_weight'] == optimal_tw].groupby('architecture')['val_f1_mean'].mean()
    if len(arch_at_optimal) > 1:
        arch_effect_pct = ((arch_at_optimal.max() - arch_at_optimal.min()) / arch_at_optimal.min()) * 100
    else:
        arch_effect_pct = 0

    return {
        'λ_tm': abs(tm_effect_pct),
        'Arch': abs(arch_effect_pct),
        'optimal_tw': optimal_tw
    }


def plot_architecture_figure_cv(df: pd.DataFrame, output_dir: Path):
    """
    Create the combined figure showing per-architecture results with CV confidence intervals.

    Similar to full_categories_figure.png but with error bands for CV uncertainty.
    5 panels: F1, Top-3 Acc, Cumul@70, JS Div, TM Effect by Encoder

    ACL-optimized: compact size with horizontal bar chart for encoder effects.
    """
    # Compute metrics by architecture
    arch_metrics = compute_metrics_by_architecture_cv(df)

    if arch_metrics.empty:
        print("No data for architecture figure!")
        return

    # Compute effect sizes
    effect_sizes = compute_relative_effects_cv(df)

    # ACL-style: use serif font, normal weight
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    # Wider figure to accommodate 7 lines, with column chart
    fig, axes = plt.subplots(1, 5, figsize=(9.5, 2.6))
    plt.subplots_adjust(wspace=0.28, top=0.80, bottom=0.18, left=0.05, right=0.98)

    # Colors and markers for each architecture
    arch_styles = {
        'gbert_base': {'color': '#2171B5', 'marker': 'o'},       # Blue
        'gbert_large': {'color': '#238B45', 'marker': 's'},      # Green
        'gelectra_base': {'color': '#D94801', 'marker': '^'},    # Orange
        'modern_gbert_134M': {'color': '#6A51A3', 'marker': 'D'}, # Purple
        'modern_gbert_1B': {'color': '#CB181D', 'marker': 'v'},   # Red
        'Eurobert_210m': {'color': '#17BECF', 'marker': '<'},     # Cyan/Teal
        'Eurobert_610m': {'color': '#B8860B', 'marker': '>'},     # Gold
    }

    # Short display names for legend
    arch_display_names = {
        'gbert_base': 'GB-base',
        'gbert_large': 'GB-large',
        'gelectra_base': 'GELECTRa',
        'modern_gbert_134M': 'MGB-134M',
        'modern_gbert_1B': 'MGB-1B',
        'Eurobert_210m': 'EB-210M',
        'Eurobert_610m': 'EB-610M',
    }

    # Metrics to plot (4 panels) - short titles like reference
    metrics_config = [
        ('macro_f1_mean', 'F1'),
        ('top_k_acc_3_mean', 'Top-3 Acc'),
        ('cumul70_mean', 'Cumul@70'),
        ('js_divergence_mean', 'JS Div'),
    ]

    # Plot each metric panel
    for idx, (metric_key, title) in enumerate(metrics_config):
        ax = axes[idx]

        for arch, style in arch_styles.items():
            arch_data = arch_metrics[arch_metrics['architecture'] == arch].sort_values('tm_weight')
            if len(arch_data) == 0:
                continue

            # Plot line with markers
            ax.plot(arch_data['tm_weight'], arch_data[metric_key],
                   marker=style['marker'], color=style['color'],
                   linewidth=1.5, markersize=5, label=arch_display_names.get(arch, arch),
                   markeredgewidth=0.5, markeredgecolor='white')

        ax.set_xlabel(r'$\lambda$_tm', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'], fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.locator_params(axis='y', nbins=4)

        # Use 2 decimal places for Top-3 Acc panel
        if 'top_k' in metric_key:
            from matplotlib.ticker import FormatStrFormatter
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Add percentage improvement annotation
        improvements = []
        for arch in arch_styles.keys():
            arch_data = arch_metrics[arch_metrics['architecture'] == arch]
            if len(arch_data) >= 2:
                y0 = arch_data[arch_data['tm_weight'] == 0.0][metric_key].values
                y_best = arch_data[metric_key].max() if 'js' not in metric_key.lower() else arch_data[metric_key].min()
                if len(y0) > 0 and y0[0] > 0:
                    if 'js' in metric_key.lower():
                        pct = ((y0[0] - y_best) / y0[0]) * 100  # JS: lower is better
                    else:
                        pct = ((y_best - y0[0]) / y0[0]) * 100
                    improvements.append(pct)
        if improvements:
            mean_pct = np.mean(improvements)
            ax.text(0.95, 0.95, f'+{mean_pct:.0f}%', transform=ax.transAxes,
                   ha='right', va='top', fontsize=8, color='#1f77b4')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    # Panel 5: Vertical column chart for TM effect by encoder with angled labels
    ax_impr = axes[4]
    improvements_by_arch = []
    for arch in arch_styles.keys():
        arch_data = arch_metrics[arch_metrics['architecture'] == arch]
        if len(arch_data) >= 2:
            baseline = arch_data[arch_data['tm_weight'] == 0.0]['val_f1_mean'].values
            best = arch_data['val_f1_mean'].max()
            if len(baseline) > 0 and baseline[0] > 0:
                pct = ((best - baseline[0]) / baseline[0]) * 100
                improvements_by_arch.append({'arch': arch, 'improvement': pct})

    if improvements_by_arch:
        impr_df = pd.DataFrame(improvements_by_arch).sort_values('improvement', ascending=False)
        colors = [arch_styles.get(a, {'color': 'gray'})['color'] for a in impr_df['arch']]
        bars = ax_impr.bar(range(len(impr_df)), impr_df['improvement'],
                          color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax_impr.set_xticks(range(len(impr_df)))
        short_labels = [arch_display_names.get(a, a) for a in impr_df['arch']]
        ax_impr.set_xticklabels(short_labels, fontsize=7, rotation=45, ha='right')
        ax_impr.set_ylabel('Impr. (%)', fontsize=8)
        ax_impr.set_title('TM Effect', fontsize=9)
        ax_impr.axhline(y=0, color='black', linewidth=0.8)
        ax_impr.tick_params(axis='y', labelsize=8)
        ax_impr.spines['top'].set_visible(True)
        ax_impr.spines['right'].set_visible(True)

    # Add legend at top - compact horizontal
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7,
              bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=7,
              handlelength=1.2, handletextpad=0.3, columnspacing=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80, wspace=0.28)

    # Save figures
    png_path = output_dir / 'full_categories_cv_figure.png'
    pdf_path = output_dir / 'full_categories_cv_figure.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    plt.close()

    # Reset rcParams to defaults
    plt.rcParams.update(plt.rcParamsDefault)


def generate_latex_table_cv(tm_stats: pd.DataFrame, model_stats: pd.DataFrame, output_dir: Path):
    """Generate LaTeX-ready table with CV statistics."""
    if tm_stats.empty:
        return

    # Table 1: TM weight effect with CI
    latex_lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    $\\lambda_{\\text{tm}}$ & F1 (Mean $\\pm$ Std) & 95\\% CI & Top-3 Acc & N \\\\",
        "    \\midrule",
    ]

    for _, row in tm_stats.iterrows():
        f1_mean = row['val_f1_mean']
        f1_std = row['val_f1_std']
        ci = f"[{row['val_f1_ci_lower']:.3f}, {row['val_f1_ci_upper']:.3f}]"
        top3 = f"{row['top_k_acc_3_mean']:.3f}" if pd.notna(row.get('top_k_acc_3_mean')) else '-'
        n = int(row['n_experiments'])
        latex_lines.append(f"    {row['tm_weight']:.1f} & {f1_mean:.4f} $\\pm$ {f1_std:.4f} & {ci} & {top3} & {n} \\\\")

    latex_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Effect of transition loss weight on full categories (60 labels) with 5-fold CV.}",
        "  \\label{tab:full_categories_cv_tm}",
        "\\end{table}",
    ])

    output_path = output_dir / 'full_categories_cv_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved LaTeX table: {output_path}")

    # Table 2: Per-model results
    if not model_stats.empty:
        latex_lines2 = [
            "\\begin{table}[t]",
            "  \\centering",
            "  \\begin{tabular}{lccccc}",
            "    \\toprule",
            "    Model & Best F1 & 95\\% CI & Best $\\lambda$ & Baseline F1 & Impr. \\\\",
            "    \\midrule",
        ]

        for _, row in model_stats.iterrows():
            model = row['pretrained_model'].split('/')[-1]  # Short name
            best_f1 = f"{row['best_f1_mean']:.4f}"
            ci = row['best_f1_ci']
            best_tw = f"{row['best_tm_weight']:.1f}"
            baseline = f"{row['baseline_f1_mean']:.4f}" if pd.notna(row['baseline_f1_mean']) else '-'
            impr = f"{row['relative_improvement']*100:+.1f}\\%" if pd.notna(row['relative_improvement']) else '-'
            latex_lines2.append(f"    {model} & {best_f1} & {ci} & {best_tw} & {baseline} & {impr} \\\\")

        latex_lines2.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Per-model results with 5-fold CV for full categories.}",
            "  \\label{tab:full_categories_cv_models}",
            "\\end{table}",
        ])

        output_path2 = output_dir / 'full_categories_cv_models_table.tex'
        with open(output_path2, 'w') as f:
            f.write('\n'.join(latex_lines2))
        print(f"Saved LaTeX table: {output_path2}")


def check_cv_completeness(df: pd.DataFrame):
    """Check how many experiments have complete 5-fold CV."""
    finished = df[df['state'] == 'finished'].copy()
    name_col = 'run_name' if 'run_name' in finished.columns else 'Name'
    finished['fold'] = finished[name_col].apply(extract_fold_from_name)
    finished['base_exp_name'] = finished[name_col].apply(extract_base_experiment_name)

    with_folds = finished[finished['fold'] >= 0]

    fold_counts = with_folds.groupby('base_exp_name')['fold'].nunique()
    complete = (fold_counts == N_FOLDS).sum()
    partial = ((fold_counts < N_FOLDS) & (fold_counts > 0)).sum()

    print("\n" + "="*60)
    print("CV COMPLETENESS CHECK")
    print("="*60)
    print(f"Total unique base experiments: {len(fold_counts)}")
    print(f"Complete (all {N_FOLDS} folds): {complete}")
    print(f"Partial (some folds missing): {partial}")

    # Show fold distribution
    print(f"\nFold count distribution:")
    for n in range(1, N_FOLDS + 1):
        count = (fold_counts == n).sum()
        print(f"  {n} folds: {count} experiments")


def extract_context_from_name(name: str) -> int:
    """Extract context length from experiment name (e.g., 'exp_ctx_4_...' -> 4)."""
    match = re.search(r'ctx_(\d+)', str(name))
    if match:
        return int(match.group(1))
    return None


def analyze_context_tm_cross_cv(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze context window size × TM weight cross-effect with CV aggregation.

    Returns:
        Tuple of (mean_df, std_df) pivot tables for weighted F1.
    """
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Extract context length from base experiment name
    cv_df['context_len'] = cv_df['base_exp_name'].apply(extract_context_from_name)

    # Filter to experiments with context info
    with_context = cv_df[cv_df['context_len'].notna()].copy()

    if with_context.empty:
        print("No experiments with context length info found.")
        return pd.DataFrame(), pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = with_context[with_context['complete'] == True].copy()
    print(f"Complete CV experiments with context info: {len(complete_cv)}")

    # Create pivot table for mean macro-F1
    pivot_mean = complete_cv.pivot_table(
        values='macro_f1_mean',
        index='context_len',
        columns='tm_weight',
        aggfunc='mean'
    ).round(3)

    # Create pivot table for std (average fold std)
    pivot_std = complete_cv.pivot_table(
        values='macro_f1_std',
        index='context_len',
        columns='tm_weight',
        aggfunc='mean'
    ).round(3)

    return pivot_mean, pivot_std


def generate_context_tm_latex_table(mean_df: pd.DataFrame, std_df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for context × TM weight analysis."""
    if mean_df.empty:
        return

    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{l" + "c" * len(mean_df.columns) + "}",
        "\\toprule",
        "& \\multicolumn{" + str(len(mean_df.columns)) + "}{c}{\\textbf{TM Weight ($\\lambda$)}} \\\\",
        "\\cmidrule(lr){2-" + str(len(mean_df.columns) + 1) + "}",
        "\\textbf{Context} & " + " & ".join([f"{col:.1f}" for col in mean_df.columns]) + " \\\\",
        "\\midrule",
    ]

    # Find best value
    best_val = mean_df.max().max()

    for ctx in sorted(mean_df.index):
        row_vals = []
        for col in mean_df.columns:
            mean_val = mean_df.loc[ctx, col]
            std_val = std_df.loc[ctx, col] if ctx in std_df.index and col in std_df.columns else 0

            formatted = f".{int(mean_val*1000):03d}\\,{{\\scriptsize$\\pm$.{int(std_val*1000):03d}}}"
            if abs(mean_val - best_val) < 0.001:
                formatted = "\\textbf{" + formatted + "}"
            row_vals.append(formatted)

        latex_lines.append(f"{int(ctx)} & " + " & ".join(row_vals) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Macro-F1 by context window size and TM weight (mean $\\pm$ std across 5-fold CV). Best performance at 4 utterances with $\\lambda$=1.5.}",
        "\\label{tab:context_ablation}",
        "\\end{table}",
    ])

    output_path = output_dir / 'context_tm_ablation_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved LaTeX table: {output_path}")


def extract_ls_value_from_name(name: str) -> float:
    """Extract label smoothing value from experiment name."""
    match = re.search(r'ls_(\d+\.?\d*)', str(name))
    return float(match.group(1)) if match else None


def analyze_label_smoothing_vs_tm_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare label smoothing vs transition-matrix regularization.

    For each encoder, compares:
    - Label smoothing with ε ∈ {0.0, 0.1, 0.2}
    - Best TM regularization config (excluding label smoothing runs)

    Returns DataFrame with comparison results.
    """
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame()

    # Only use complete CV experiments
    complete_cv = cv_df[cv_df['complete'] == True].copy()

    # Identify label smoothing runs
    complete_cv['is_label_smoothing'] = complete_cv['base_exp_name'].str.contains(
        'label_smoothing', case=False, na=False
    )
    complete_cv['ls_value'] = complete_cv['base_exp_name'].apply(extract_ls_value_from_name)

    # Encoder display names
    encoder_display_names = {
        'gbert_base': 'GBERT-base',
        'gbert_large': 'GBERT-large',
        'gelectra_base': 'GELECTRa-base',
        'modern_gbert_134M': 'ModernGBERT-134M',
        'modern_gbert_1B': 'ModernGBERT-1B',
        'Eurobert_210m': 'EuroBERT-210M',
        'Eurobert_610m': 'EuroBERT-610M',
    }

    results = []
    for model in complete_cv['pretrained_model'].dropna().unique():
        # Skip unknown/invalid models
        if 'unknown' in str(model).lower():
            continue

        model_df = complete_cv[complete_cv['pretrained_model'] == model]

        if model_df.empty:
            continue

        short_name = model.split('/')[-1] if '/' in str(model) else str(model)
        display_name = encoder_display_names.get(short_name, short_name)

        # Get label smoothing results
        ls_results = {}
        for ls_val in [0.0, 0.1, 0.2]:
            ls_subset = model_df[(model_df['is_label_smoothing']) & (model_df['ls_value'] == ls_val)]
            if not ls_subset.empty:
                ls_results[ls_val] = {
                    'macro_f1_mean': ls_subset['macro_f1_mean'].mean(),
                    'macro_f1_std': ls_subset['macro_f1_std'].mean(),
                }

        # Get best TM regularization result (excluding label smoothing)
        tm_df = model_df[~model_df['is_label_smoothing']]
        tm_df_valid = tm_df[tm_df['macro_f1_mean'].notna()]

        if tm_df_valid.empty:
            continue

        best_tm_idx = tm_df_valid['macro_f1_mean'].idxmax()
        best_tm = tm_df_valid.loc[best_tm_idx]

        # Calculate improvement over best label smoothing
        best_ls_f1 = max([ls_results.get(v, {}).get('macro_f1_mean', 0) for v in [0.0, 0.1, 0.2]])
        delta = best_tm['macro_f1_mean'] - best_ls_f1 if best_ls_f1 > 0 else None

        results.append({
            'encoder': display_name,
            'encoder_key': short_name,
            'ls_0.0': ls_results.get(0.0, {}).get('macro_f1_mean'),
            'ls_0.0_std': ls_results.get(0.0, {}).get('macro_f1_std'),
            'ls_0.1': ls_results.get(0.1, {}).get('macro_f1_mean'),
            'ls_0.1_std': ls_results.get(0.1, {}).get('macro_f1_std'),
            'ls_0.2': ls_results.get(0.2, {}).get('macro_f1_mean'),
            'ls_0.2_std': ls_results.get(0.2, {}).get('macro_f1_std'),
            'tm_best': best_tm['macro_f1_mean'],
            'tm_best_std': best_tm.get('macro_f1_std', 0),
            'tm_best_lambda': best_tm['tm_weight'],
            'delta': delta,
        })

    return pd.DataFrame(results).sort_values('tm_best', ascending=False)


def generate_label_smoothing_latex_table(ls_stats: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table comparing label smoothing vs TM regularization."""
    if ls_stats.empty:
        return

    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "& \\multicolumn{3}{c}{\\textbf{Label Smoothing}} & \\multicolumn{2}{c}{\\textbf{TM Reg.}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-6}",
        "\\textbf{Encoder} & $\\epsilon$=0.0 & $\\epsilon$=0.1 & $\\epsilon$=0.2 & Best & $\\Delta$ \\\\",
        "\\midrule",
    ]

    for _, row in ls_stats.iterrows():
        # Format label smoothing values
        ls_0 = f".{int(row['ls_0.0']*1000):03d}" if pd.notna(row['ls_0.0']) else "--"
        ls_1 = f".{int(row['ls_0.1']*1000):03d}" if pd.notna(row['ls_0.1']) else "--"
        ls_2 = f".{int(row['ls_0.2']*1000):03d}" if pd.notna(row['ls_0.2']) else "--"

        # Find best LS value for bolding
        ls_values = [row['ls_0.0'], row['ls_0.1'], row['ls_0.2']]
        valid_ls = [v for v in ls_values if pd.notna(v)]
        best_ls = max(valid_ls) if valid_ls else None

        if best_ls is not None:
            if pd.notna(row['ls_0.0']) and row['ls_0.0'] == best_ls:
                ls_0 = f"\\textbf{{{ls_0}}}"
            if pd.notna(row['ls_0.1']) and row['ls_0.1'] == best_ls:
                ls_1 = f"\\textbf{{{ls_1}}}"
            if pd.notna(row['ls_0.2']) and row['ls_0.2'] == best_ls:
                ls_2 = f"\\textbf{{{ls_2}}}"

        # Format TM best
        tm_best = f".{int(row['tm_best']*1000):03d}" if pd.notna(row['tm_best']) else "--"

        # Format delta
        if pd.notna(row['delta']):
            delta_val = row['delta']
            if delta_val > 0:
                delta = f"+.{int(abs(delta_val)*1000):03d}"
            else:
                delta = f"--.{int(abs(delta_val)*1000):03d}"
        else:
            delta = "--"

        # Bold TM if better than all LS
        if pd.notna(row['tm_best']) and best_ls is not None and row['tm_best'] > best_ls:
            tm_best = f"\\textbf{{{tm_best}}}"

        latex_lines.append(f"{row['encoder']} & {ls_0} & {ls_1} & {ls_2} & {tm_best} & {delta} \\\\")

    # Add mean row
    mean_ls_0 = ls_stats['ls_0.0'].mean()
    mean_ls_1 = ls_stats['ls_0.1'].mean()
    mean_ls_2 = ls_stats['ls_0.2'].mean()
    mean_tm = ls_stats['tm_best'].mean()
    mean_delta = ls_stats['delta'].mean()

    latex_lines.extend([
        "\\midrule",
        f"\\textit{{Mean}} & .{int(mean_ls_0*1000):03d} & .{int(mean_ls_1*1000):03d} & .{int(mean_ls_2*1000):03d} & \\textbf{{.{int(mean_tm*1000):03d}}} & +.{int(mean_delta*1000):03d} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Label smoothing vs.\\ TM regularization (macro-F1, 60 categories, 5-fold CV). TM regularization outperforms label smoothing on all encoders. $\\Delta$ = TM best $-$ best LS.}",
        "\\label{tab:ablation_ls_full}",
        "\\end{table}",
    ])

    output_path = output_dir / 'label_smoothing_vs_tm_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved LaTeX table: {output_path}")

    # Also save CSV
    csv_path = output_dir / 'label_smoothing_vs_tm_results.csv'
    ls_stats.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze full_categories CV results')
    parser.add_argument('--data', type=str, help='Path to wandb export CSV')
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    # Load data
    data_path = Path(args.data) if args.data else None
    df = load_data(data_path)

    print(f"Total runs loaded: {len(df)}")
    print(f"States: {df['state'].value_counts().to_dict()}")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")

    # Check CV completeness
    check_cv_completeness(df)

    # Filter to valid models (bert + history)
    valid_df = filter_valid_models(df)
    print(f"\nValid model runs (bert + history): {len(valid_df)}")
    print(f"  BERT: {len(valid_df[valid_df['model_type'] == 'bert'])}")
    print(f"  History: {len(valid_df[valid_df['model_type'] == 'history'])}")
    print(f"Finished runs: {len(valid_df[valid_df['state'] == 'finished'])}")

    # Analyze TM effect with CV (all model types)
    print("\n" + "="*60)
    print("TM WEIGHT EFFECT (ALL model types, 5-fold CV)")
    print("="*60)
    tm_stats_all = analyze_tm_effect_all_cv(valid_df)
    if not tm_stats_all.empty:
        print(tm_stats_all.to_string(index=False))

    # Analyze by model_type (bert vs history)
    print("\n" + "="*60)
    print("RESULTS BY MODEL TYPE (bert vs history, 5-fold CV)")
    print("="*60)
    model_type_stats = analyze_by_model_type_cv(valid_df)
    if not model_type_stats.empty:
        print(model_type_stats.to_string(index=False))

    # Filter to BERT only for per-model analysis
    bert_df = filter_bert_only(df)
    print(f"\nBERT-only runs: {len(bert_df)}")

    # Analyze TM effect with CV (BERT only - for backward compatibility)
    print("\n" + "="*60)
    print("TM WEIGHT EFFECT (BERT models only, 5-fold CV)")
    print("="*60)
    tm_stats = analyze_tm_effect_cv(bert_df)
    if not tm_stats.empty:
        print(tm_stats.to_string(index=False))

    # Analyze per model with CV
    print("\n" + "="*60)
    print("RESULTS PER PRETRAINED MODEL (5-fold CV)")
    print("="*60)
    model_stats = analyze_per_model_cv(bert_df)
    if not model_stats.empty:
        print(model_stats.to_string(index=False))

    # Analyze encoder results by macro-F1 (all model types)
    print("\n" + "="*60)
    print("ENCODER RESULTS BY MACRO-F1 (5-fold CV)")
    print("="*60)
    encoder_macro_stats = analyze_encoder_by_macro_f1_cv(valid_df)
    if not encoder_macro_stats.empty:
        print(encoder_macro_stats.to_string(index=False))
        # Generate LaTeX table
        generate_encoder_macro_f1_latex_table(encoder_macro_stats, output_dir)
        # Save CSV
        encoder_macro_stats.to_csv(output_dir / 'encoder_macro_f1_results.csv', index=False)

    # Compare with client_28
    compare_with_client_28_cv(tm_stats)

    # Analyze context × TM weight cross-effect
    print("\n" + "="*60)
    print("CONTEXT LENGTH × TM WEIGHT CROSS-ANALYSIS (5-fold CV)")
    print("="*60)
    context_tm_mean, context_tm_std = analyze_context_tm_cross_cv(valid_df)
    if not context_tm_mean.empty:
        print("\nMacro-F1 (mean):")
        print(context_tm_mean.to_string())
        print("\nMacro-F1 (std across folds):")
        print(context_tm_std.to_string())

        # Generate LaTeX table
        generate_context_tm_latex_table(context_tm_mean, context_tm_std, output_dir)

        # Save CSV
        context_tm_mean.to_csv(output_dir / 'context_tm_cross_mean.csv')
        context_tm_std.to_csv(output_dir / 'context_tm_cross_std.csv')

    # Generate outputs
    if not tm_stats.empty:
        plot_tm_effect_cv(tm_stats, output_dir)
        plot_fold_variability(bert_df, output_dir)
        plot_architecture_figure_cv(bert_df, output_dir)
        generate_latex_table_cv(tm_stats, model_stats, output_dir)

        # Save CSV summaries
        tm_stats.to_csv(output_dir / 'full_categories_cv_tm_summary.csv', index=False)
        if not model_stats.empty:
            model_stats.to_csv(output_dir / 'full_categories_cv_model_summary.csv', index=False)

        # Save full aggregated results (BERT only)
        cv_df = aggregate_cv_results(bert_df)
        if not cv_df.empty:
            cv_df.to_csv(output_dir / 'full_categories_cv_all_experiments.csv', index=False)

        # Save TM stats for all model types
        if not tm_stats_all.empty:
            tm_stats_all.to_csv(output_dir / 'full_categories_cv_tm_summary_all.csv', index=False)

        # Save model_type comparison
        if not model_type_stats.empty:
            model_type_stats.to_csv(output_dir / 'full_categories_cv_model_type_summary.csv', index=False)

        # Save full aggregated results for ALL model types
        cv_df_all = aggregate_cv_results(valid_df)
        if not cv_df_all.empty:
            cv_df_all.to_csv(output_dir / 'full_categories_cv_all_experiments_all.csv', index=False)

        print(f"\nSaved CSV summaries to {output_dir}")

    # Label smoothing vs TM regularization comparison
    print("\n" + "="*60)
    print("LABEL SMOOTHING VS TM REGULARIZATION (5-fold CV)")
    print("="*60)
    ls_stats = analyze_label_smoothing_vs_tm_cv(df)
    if not ls_stats.empty:
        print(ls_stats.to_string())
        generate_label_smoothing_latex_table(ls_stats, output_dir)
    else:
        print("No label smoothing data found.")

    # Run significance tests (now testing multiple λ values: 0.2, 0.5, 1.0, 1.5)
    print("\n" + "="*60)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS (Multiple λ values)")
    print("="*60)
    sig_results = run_significance_tests(df)
    if not sig_results.empty:
        corrected_results = print_significance_summary(sig_results)

        # Print summary by λ value (shows consistency across regularization strengths)
        lambda_summary = print_lambda_summary(corrected_results)

        # Generate LaTeX tables
        generate_significance_latex_table(corrected_results, output_dir)

        # Save results
        corrected_results.to_csv(output_dir / 'significance_tests_results.csv', index=False)
        if lambda_summary is not None and not lambda_summary.empty:
            lambda_summary.to_csv(output_dir / 'significance_by_lambda_summary.csv', index=False)
        print(f"\nSaved significance test results to {output_dir / 'significance_tests_results.csv'}")
    else:
        print("No significance tests could be run.")

    print("\n" + "="*60)
    print("CV ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
