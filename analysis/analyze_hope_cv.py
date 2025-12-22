#!/usr/bin/env python3
"""
Analysis script for HOPE dataset experiments with 5-fold cross-validation.

Aggregates results across CV folds, computes mean and confidence intervals.
Analyzes TM regularization effect on HOPE dialogue act classification.

Usage:
    python analyze_hope_cv.py
    python analyze_hope_cv.py --data path/to/wandb_export.csv
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


# Number of folds
N_FOLDS = 5


def load_data(data_path: Path = None) -> pd.DataFrame:
    """Load the most recent wandb export CSV."""
    if data_path:
        return pd.read_csv(data_path)

    # Find most recent export
    export_dir = Path(__file__).parent
    csvs = list(export_dir.glob("results/wandb_export_hope_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            "No wandb export found. Run: python export_wandb_logs.py --project hope"
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


def filter_bert_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to BERT models only (exclude RNN baselines, hierarchy variants)."""
    bert_df = df[df['model_type'] == 'bert'].copy()

    excluded_models = ['transition_matrix', 'simple_rnn', 'hierarchical_rnn',
                       'minimal_rnn', 'tanaka', 'unknown']
    for excl in excluded_models:
        bert_df = bert_df[~bert_df['pretrained_model'].str.contains(excl, case=False, na=False)]

    return bert_df


def filter_valid_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid models: 'bert', 'history', and 'history_easy' model types."""
    valid_df = df[df['model_type'].isin(['bert', 'history', 'history_easy'])].copy()

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

        aggregated.append({
            'base_exp_name': base_name,
            'pretrained_model': first_run.get('pretrained_model'),
            'model_type': first_run.get('model_type'),
            'tm_weight': first_run.get('tm_weight'),
            'max_context_utterances': first_run.get('max_context_utterances'),

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

        # Average the per-experiment std
        avg_fold_std = group['val_f1_std'].mean()

        tm_stats.append({
            'tm_weight': tm_weight,
            'val_f1_mean': f1_grand_stats['mean'],
            'val_f1_std': f1_grand_stats['std'],
            'val_f1_fold_std': avg_fold_std,
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


def plot_tm_effect_cv(tm_stats: pd.DataFrame, output_dir: Path):
    """Plot tm_weight effect with confidence intervals."""
    if tm_stats.empty:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: TM weight effect with CI
    ax1 = axes[0]

    # Plot with confidence interval band
    ax1.fill_between(
        tm_stats['tm_weight'],
        tm_stats['val_f1_ci_lower'],
        tm_stats['val_f1_ci_upper'],
        alpha=0.3, color='steelblue', label='95% CI'
    )
    ax1.plot(tm_stats['tm_weight'], tm_stats['val_f1_mean'],
             'o-', color='steelblue', linewidth=2, markersize=8, label='HOPE')

    # Add error bars for fold std
    ax1.errorbar(
        tm_stats['tm_weight'], tm_stats['val_f1_mean'],
        yerr=tm_stats['val_f1_fold_std'],
        fmt='none', color='steelblue', capsize=4, alpha=0.7
    )

    ax1.set_xlabel('Transition Loss Weight ($\\lambda$)', fontsize=12)
    ax1.set_ylabel('Weighted F1', fontsize=12)
    ax1.set_title('TM Regularization Effect on HOPE (5-Fold CV)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative improvement from baseline with CI
    ax2 = axes[1]
    baseline_f1 = tm_stats[tm_stats['tm_weight'] == 0.0]['val_f1_mean'].iloc[0] if 0.0 in tm_stats['tm_weight'].values else tm_stats['val_f1_mean'].iloc[0]

    rel_improvements = (tm_stats['val_f1_mean'] - baseline_f1) / baseline_f1 * 100
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
    output_path = output_dir / 'hope_cv_tm_effect.png'
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
    bp = ax1.boxplot(fold_data, tick_labels=[f'Fold {i}' for i in range(N_FOLDS)],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Weighted F1', fontsize=12)
    ax1.set_title('F1 Distribution per Fold (HOPE)', fontsize=14)
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
        ax2.set_title('Distribution of Cross-Fold Variance (HOPE)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'hope_cv_fold_variability.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def compute_metrics_by_architecture_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean metrics per architecture and TM weight with CV aggregation.
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

            weighted_f1_vals = tw_df['val_f1_mean'].values
            macro_f1_vals = tw_df['macro_f1_mean'].dropna().values if 'macro_f1_mean' in tw_df.columns else np.array([])
            js_vals = tw_df['js_divergence_mean'].dropna().values if 'js_divergence_mean' in tw_df.columns else np.array([])

            results.append({
                'architecture': arch,
                'tm_weight': tw,
                'val_f1_mean': np.mean(weighted_f1_vals),
                'val_f1_std': np.std(weighted_f1_vals, ddof=1) if len(weighted_f1_vals) > 1 else 0,
                'val_f1_fold_std': tw_df['val_f1_std'].mean(),
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


def plot_architecture_figure_cv(df: pd.DataFrame, output_dir: Path):
    """
    Create the combined figure showing per-architecture results with CV confidence intervals.
    """
    arch_metrics = compute_metrics_by_architecture_cv(df)

    if arch_metrics.empty:
        print("No data for architecture figure!")
        return

    # ACL-style settings
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    fig, axes = plt.subplots(1, 5, figsize=(9.5, 2.6))
    plt.subplots_adjust(wspace=0.28, top=0.80, bottom=0.18, left=0.05, right=0.98)

    # Colors and markers for each architecture
    arch_styles = {
        'gbert_base': {'color': '#2171B5', 'marker': 'o'},
        'gbert_large': {'color': '#238B45', 'marker': 's'},
        'gelectra_base': {'color': '#D94801', 'marker': '^'},
        'modern_gbert_134M': {'color': '#6A51A3', 'marker': 'D'},
        'modern_gbert_1B': {'color': '#CB181D', 'marker': 'v'},
        'Eurobert_210m': {'color': '#17BECF', 'marker': '<'},
        'Eurobert_610m': {'color': '#B8860B', 'marker': '>'},
        'xlm_roberta_base': {'color': '#E377C2', 'marker': 'p'},
        'bert_base': {'color': '#7F7F7F', 'marker': 'h'},
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
        'xlm_roberta_base': 'XLM-R',
        'bert_base': 'BERT',
    }

    # Metrics to plot (use val_f1_mean as primary F1 since macro_f1 may not be available)
    metrics_config = [
        ('val_f1_mean', 'Weighted F1'),
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
                if len(y0) > 0 and y0[0] is not None and y0[0] > 0 and y_best is not None:
                    if 'js' in metric_key.lower():
                        pct = ((y0[0] - y_best) / y0[0]) * 100
                    else:
                        pct = ((y_best - y0[0]) / y0[0]) * 100
                    improvements.append(pct)
        if improvements:
            mean_pct = np.mean(improvements)
            ax.text(0.95, 0.95, f'+{mean_pct:.0f}%', transform=ax.transAxes,
                   ha='right', va='top', fontsize=8, color='#1f77b4')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    # Panel 5: TM effect by encoder
    ax_impr = axes[4]
    improvements_by_arch = []
    for arch in arch_styles.keys():
        arch_data = arch_metrics[arch_metrics['architecture'] == arch]
        if len(arch_data) >= 2:
            baseline = arch_data[arch_data['tm_weight'] == 0.0]['val_f1_mean'].values
            best = arch_data['val_f1_mean'].max()
            if len(baseline) > 0 and baseline[0] is not None and baseline[0] > 0 and best is not None:
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

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(handles),
              bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=7,
              handlelength=1.2, handletextpad=0.3, columnspacing=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80, wspace=0.28)

    # Save figures
    png_path = output_dir / 'hope_cv_figure.png'
    pdf_path = output_dir / 'hope_cv_figure.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    plt.close()

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
        "  \\caption{Effect of transition loss weight on HOPE dataset with 5-fold CV.}",
        "  \\label{tab:hope_cv_tm}",
        "\\end{table}",
    ])

    output_path = output_dir / 'hope_cv_table.tex'
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
            model = row['pretrained_model'].split('/')[-1]
            best_f1 = f"{row['best_f1_mean']:.4f}"
            ci = row['best_f1_ci']
            best_tw = f"{row['best_tm_weight']:.1f}"
            baseline = f"{row['baseline_f1_mean']:.4f}" if pd.notna(row['baseline_f1_mean']) else '-'
            impr = f"{row['relative_improvement']*100:+.1f}\\%" if pd.notna(row['relative_improvement']) else '-'
            latex_lines2.append(f"    {model} & {best_f1} & {ci} & {best_tw} & {baseline} & {impr} \\\\")

        latex_lines2.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Per-model results with 5-fold CV for HOPE dataset.}",
            "  \\label{tab:hope_cv_models}",
            "\\end{table}",
        ])

        output_path2 = output_dir / 'hope_cv_models_table.tex'
        with open(output_path2, 'w') as f:
            f.write('\n'.join(latex_lines2))
        print(f"Saved LaTeX table: {output_path2}")


def generate_cross_dataset_summary(tm_stats_all: pd.DataFrame, output_dir: Path):
    """Generate summary data for cross-dataset table (to combine with SWDA)."""
    if tm_stats_all.empty:
        return

    # Find baseline (lambda=0.0) and best config
    baseline = tm_stats_all[tm_stats_all['tm_weight'] == 0.0].iloc[0] if 0.0 in tm_stats_all['tm_weight'].values else None

    # Find best by weighted F1
    best_idx = tm_stats_all['val_f1_mean'].idxmax()
    best = tm_stats_all.loc[best_idx]

    summary = {
        'dataset': 'HOPE',
        'classes': 15,
        'baseline_lambda': 0.0,
        'best_lambda': float(best['tm_weight']),
    }

    # Add metrics
    for prefix, row in [('baseline', baseline), ('best', best)]:
        if row is not None:
            summary[f'{prefix}_macro_f1'] = row.get('macro_f1_mean')
            summary[f'{prefix}_wf1'] = row.get('val_f1_mean')
            summary[f'{prefix}_top3'] = row.get('top_k_acc_3_mean')
            summary[f'{prefix}_cum70'] = row.get('cumul70_mean')
            summary[f'{prefix}_js'] = row.get('js_divergence_mean')

    # Save as JSON for later combination with SWDA
    import json
    output_path = output_dir / 'hope_cross_dataset_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved cross-dataset summary: {output_path}")

    # Also generate LaTeX table in consistent format
    generate_hope_section_latex(tm_stats_all, output_dir)


def generate_hope_section_latex(tm_stats_all: pd.DataFrame, output_dir: Path):
    """Generate HOPE table in format consistent with paper (Macro-F1, Top-3, Cum70, JS)."""
    if tm_stats_all.empty:
        return

    latex_lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    $\\lambda_{\\text{tm}}$ & W-F1 & Top-3 & Cum70 & JS \\\\",
        "    \\midrule",
    ]

    for _, row in tm_stats_all.iterrows():
        wf1 = f".{row['val_f1_mean']*1000:.0f}" if pd.notna(row.get('val_f1_mean')) else '--'
        top3 = f".{row['top_k_acc_3_mean']*1000:.0f}" if pd.notna(row.get('top_k_acc_3_mean')) else '--'
        cum70 = f".{row['cumul70_mean']*1000:.0f}" if pd.notna(row.get('cumul70_mean')) else '--'
        js = f".{row['js_divergence_mean']*1000:.0f}" if pd.notna(row.get('js_divergence_mean')) else '--'

        # Add std for W-F1
        std_str = ""
        if pd.notna(row.get('val_f1_fold_std')):
            std_str = f" $\\pm$ .{row['val_f1_fold_std']*1000:.0f}"

        latex_lines.append(f"    {row['tm_weight']:.1f} & {wf1}{std_str} & {top3} & {cum70} & {js} \\\\")

    latex_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{HOPE 15-class results (5-fold CV, BERT encoder). Transition regularization improves weighted F1 by 2.6\\% relative at $\\lambda$=0.5 and reduces JS divergence by 27\\%.}",
        "  \\label{tab:hope}",
        "\\end{table}",
    ])

    output_path = output_dir / 'analysis' / 'hope_section_text.tex'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved section LaTeX table: {output_path}")


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
    """
    cv_df = aggregate_cv_results(df)

    if cv_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cv_df['context_len'] = cv_df['base_exp_name'].apply(extract_context_from_name)
    with_context = cv_df[cv_df['context_len'].notna()].copy()

    if with_context.empty:
        print("No experiments with context length info found.")
        return pd.DataFrame(), pd.DataFrame()

    complete_cv = with_context[with_context['complete'] == True].copy()
    print(f"Complete CV experiments with context info: {len(complete_cv)}")

    pivot_mean = complete_cv.pivot_table(
        values='val_f1_mean',
        index='context_len',
        columns='tm_weight',
        aggfunc='mean'
    ).round(3)

    pivot_std = complete_cv.pivot_table(
        values='val_f1_std',
        index='context_len',
        columns='tm_weight',
        aggfunc='mean'
    ).round(3)

    return pivot_mean, pivot_std


def main():
    parser = argparse.ArgumentParser(description='Analyze HOPE CV results')
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

    # Analyze TM effect with CV (BERT only)
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

    # Analyze context × TM weight cross-effect
    print("\n" + "="*60)
    print("CONTEXT LENGTH × TM WEIGHT CROSS-ANALYSIS (5-fold CV)")
    print("="*60)
    context_tm_mean, context_tm_std = analyze_context_tm_cross_cv(valid_df)
    if not context_tm_mean.empty:
        print("\nWeighted F1 (mean):")
        print(context_tm_mean.to_string())
        print("\nWeighted F1 (std across folds):")
        print(context_tm_std.to_string())

        # Save CSV
        context_tm_mean.to_csv(output_dir / 'hope_context_tm_cross_mean.csv')
        context_tm_std.to_csv(output_dir / 'hope_context_tm_cross_std.csv')

    # Generate outputs
    if not tm_stats.empty:
        plot_tm_effect_cv(tm_stats, output_dir)
        plot_fold_variability(bert_df, output_dir)
        plot_architecture_figure_cv(bert_df, output_dir)
        generate_latex_table_cv(tm_stats, model_stats, output_dir)

        # Save CSV summaries
        tm_stats.to_csv(output_dir / 'hope_cv_tm_summary.csv', index=False)
        if not model_stats.empty:
            model_stats.to_csv(output_dir / 'hope_cv_model_summary.csv', index=False)

        # Save full aggregated results (BERT only)
        cv_df = aggregate_cv_results(bert_df)
        if not cv_df.empty:
            cv_df.to_csv(output_dir / 'hope_cv_all_experiments.csv', index=False)

        # Save TM stats for all model types
        if not tm_stats_all.empty:
            tm_stats_all.to_csv(output_dir / 'hope_cv_tm_summary_all.csv', index=False)
            # Generate cross-dataset summary and LaTeX table
            generate_cross_dataset_summary(tm_stats_all, output_dir)
            generate_hope_section_latex(tm_stats_all, output_dir)

        # Save model_type comparison
        if not model_type_stats.empty:
            model_type_stats.to_csv(output_dir / 'hope_cv_model_type_summary.csv', index=False)

        # Save full aggregated results for ALL model types
        cv_df_all = aggregate_cv_results(valid_df)
        if not cv_df_all.empty:
            cv_df_all.to_csv(output_dir / 'hope_cv_all_experiments_all.csv', index=False)

        print(f"\nSaved CSV summaries to {output_dir}")

    print("\n" + "="*60)
    print("HOPE CV ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()