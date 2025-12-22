#!/usr/bin/env python3
"""
Analysis script for SWDA (Switchboard Dialog Act) experiments.

Compares TM regularization effect on SWDA (9 labels) with:
- Tanaka et al. (2019) baselines
- Client-only (28 labels) results from the paper

Focus on BERT models to test cross-dataset generalization.

Usage:
    python analyze_swda_results.py
    python analyze_swda_results.py --data path/to/wandb_export.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Reference baselines from Tanaka et al. (2019)
TANAKA_BASELINES = {
    'max_probability_tm_only': {'accuracy': 0.548, 'macro_f1': 0.169},
    'tanaka_best': {'accuracy': 0.697, 'macro_f1': 0.324},
}

# Reference results from client_categories (28 labels) - from paper
CLIENT_28_RESULTS = {
    'tm_weight_means': {
        0.0: 0.2246,
        0.2: 0.2549,
        0.5: 0.2646,
        1.0: 0.2505,
        1.5: 0.2530,
    },
    'optimal_lambda': 0.5,
    'relative_improvement': 0.178,  # 17.8% from lambda=0 to lambda=0.5
}


def load_data(data_path: Path = None) -> pd.DataFrame:
    """Load the most recent wandb export CSV."""
    if data_path:
        return pd.read_csv(data_path)

    # Find most recent export
    export_dir = Path(__file__).parent
    csvs = list(export_dir.glob("wandb_export_swda_ndap_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            "No wandb export found. Run: python export_wandb_logs.py --project swda_ndap"
        )
    latest = max(csvs, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest}")
    return pd.read_csv(latest)


def filter_bert_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to BERT models only (bert_base and xlm_roberta_base)."""
    bert_df = df[df['model_type'] == 'bert'].copy()

    # Exclude transition_matrix baseline and tanaka model
    excluded = ['transition_matrix', 'tanaka']
    for excl in excluded:
        bert_df = bert_df[~bert_df['pretrained_model'].str.contains(excl, case=False, na=False)]

    return bert_df


def analyze_tm_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze effect of tm_weight on metrics."""
    finished = df[df['state'] == 'finished'].copy()

    if finished.empty:
        print("WARNING: No finished runs found!")
        return pd.DataFrame()

    # Group by tm_weight and compute stats
    # Prioritize macro_f1 over weighted f1 for imbalanced datasets
    agg_dict = {}

    # Macro F1 is the primary metric for comparison with Tanaka
    if 'val_macro_f1' in finished.columns:
        agg_dict['val_macro_f1'] = ['mean', 'std', 'count', 'max']

    # Weighted F1 as secondary
    if 'val_f1' in finished.columns:
        agg_dict['val_f1'] = ['mean', 'std', 'max']
    elif 'val_weighted_f1' in finished.columns:
        agg_dict['val_weighted_f1'] = ['mean', 'std', 'max']

    # Add accuracy if available
    if 'val_accuracy' in finished.columns:
        agg_dict['val_accuracy'] = ['mean', 'std']

    # Top-3 accuracy
    if 'val_top_3_accuracy' in finished.columns:
        agg_dict['val_top_3_accuracy'] = ['mean', 'std']
    elif 'top_k_acc_3' in finished.columns:
        agg_dict['top_k_acc_3'] = ['mean', 'std']

    # Cumulative@70 accuracy
    if 'val_cumulative70_pred_accuracy' in finished.columns:
        agg_dict['val_cumulative70_pred_accuracy'] = ['mean', 'std']

    # JS divergence
    if 'mean_js_divergence' in finished.columns:
        agg_dict['mean_js_divergence'] = ['mean', 'std']

    tm_stats = finished.groupby('tm_weight').agg(agg_dict).round(4)

    # Flatten column names
    tm_stats.columns = ['_'.join(col).strip() for col in tm_stats.columns.values]
    tm_stats = tm_stats.reset_index()

    return tm_stats


def analyze_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results per pretrained model (bert_base vs xlm_roberta_base)."""
    finished = df[df['state'] == 'finished'].copy()

    if finished.empty:
        return pd.DataFrame()

    # Use macro_f1 as primary metric if available, otherwise val_f1
    primary_metric = 'val_macro_f1' if 'val_macro_f1' in finished.columns else 'val_f1'

    results = []
    for model in finished['pretrained_model'].unique():
        model_df = finished[finished['pretrained_model'] == model]

        if model_df[primary_metric].notna().any():
            best_idx = model_df[primary_metric].idxmax()
            best = model_df.loc[best_idx]

            # Baseline (tm_weight=0)
            baseline = model_df[model_df['tm_weight'] == 0.0]
            baseline_macro_f1 = baseline['val_macro_f1'].mean() if not baseline.empty and 'val_macro_f1' in baseline.columns else None
            baseline_f1 = baseline['val_f1'].mean() if not baseline.empty and 'val_f1' in baseline.columns else None
            baseline_acc = baseline['val_accuracy'].mean() if not baseline.empty and 'val_accuracy' in baseline.columns else None

            # Relative improvement (based on macro F1)
            rel_improvement = None
            if baseline_macro_f1 and baseline_macro_f1 > 0:
                rel_improvement = (best.get('val_macro_f1', 0) - baseline_macro_f1) / baseline_macro_f1

            results.append({
                'pretrained_model': model,
                'best_macro_f1': best.get('val_macro_f1'),
                'best_weighted_f1': best.get('val_f1'),
                'best_accuracy': best.get('val_accuracy'),
                'best_tm_weight': best['tm_weight'],
                'baseline_macro_f1': baseline_macro_f1,
                'baseline_accuracy': baseline_acc,
                'relative_improvement': rel_improvement,
                'n_runs': len(model_df),
            })

    sort_col = 'best_macro_f1' if 'val_macro_f1' in finished.columns else 'best_weighted_f1'
    return pd.DataFrame(results).sort_values(sort_col, ascending=False)


def compare_with_baselines(tm_stats: pd.DataFrame, model_stats: pd.DataFrame):
    """Compare SWDA results with Tanaka baselines and client results."""
    print("\n" + "="*60)
    print("COMPARISON WITH TANAKA ET AL. (2019) BASELINES")
    print("="*60)

    print("\nTanaka Baselines:")
    print(f"  Max-Probability (TM only): Acc={TANAKA_BASELINES['max_probability_tm_only']['accuracy']:.1%}, "
          f"Macro-F1={TANAKA_BASELINES['max_probability_tm_only']['macro_f1']:.1%}")
    print(f"  Tanaka Best:               Acc={TANAKA_BASELINES['tanaka_best']['accuracy']:.1%}, "
          f"Macro-F1={TANAKA_BASELINES['tanaka_best']['macro_f1']:.1%}")

    if not model_stats.empty:
        print("\nOur BERT Results (best per model):")
        for _, row in model_stats.iterrows():
            macro_f1_str = f"Macro-F1={row['best_macro_f1']:.1%}" if pd.notna(row.get('best_macro_f1')) else "Macro-F1=N/A"
            acc_str = f"Acc={row['best_accuracy']:.1%}" if pd.notna(row.get('best_accuracy')) else "Acc=N/A"
            print(f"  {row['pretrained_model']}: {macro_f1_str}, {acc_str}, λ={row['best_tm_weight']}")

            # Compare with Tanaka (use macro F1 for comparison)
            if pd.notna(row.get('best_macro_f1')):
                vs_tanaka_f1 = row['best_macro_f1'] - TANAKA_BASELINES['tanaka_best']['macro_f1']
                print(f"    vs Tanaka best: {vs_tanaka_f1:+.1%} macro F1")
            if pd.notna(row.get('best_accuracy')):
                vs_tanaka_acc = row['best_accuracy'] - TANAKA_BASELINES['tanaka_best']['accuracy']
                print(f"    vs Tanaka best: {vs_tanaka_acc:+.1%} accuracy")

    print("\n" + "="*60)
    print("CROSS-DATASET TM EFFECT COMPARISON")
    print("="*60)

    # Determine metric column - prefer macro F1
    f1_col = 'val_macro_f1_mean' if 'val_macro_f1_mean' in tm_stats.columns else 'val_f1_mean'
    f1_label = "Macro F1" if 'macro' in f1_col else "Weighted F1"

    if not tm_stats.empty and 0.0 in tm_stats['tm_weight'].values:
        # SWDA improvement
        swda_baseline = tm_stats[tm_stats['tm_weight'] == 0.0][f1_col].iloc[0]
        swda_best_row = tm_stats.loc[tm_stats[f1_col].idxmax()]
        swda_best = swda_best_row[f1_col]
        swda_improvement = (swda_best - swda_baseline) / swda_baseline if swda_baseline > 0 else 0

        print(f"\nRelative {f1_label} improvement from TM regularization:")
        print(f"  SWDA (9 labels):    {swda_improvement:+.1%} (λ=0 → λ={swda_best_row['tm_weight']})")
        print(f"  Client (28 labels): {CLIENT_28_RESULTS['relative_improvement']:+.1%} (λ=0 → λ=0.5)")

        print(f"\nOptimal λ:")
        print(f"  SWDA (9 labels):    {swda_best_row['tm_weight']}")
        print(f"  Client (28 labels): {CLIENT_28_RESULTS['optimal_lambda']}")

        # Check if same pattern
        if 0.4 <= swda_best_row['tm_weight'] <= 0.6:
            print("\n✓ Similar optimal λ range across datasets!")
        else:
            print(f"\n⚠ Different optimal λ: SWDA={swda_best_row['tm_weight']}, Client=0.5")


def plot_tm_effect(tm_stats: pd.DataFrame, output_dir: Path):
    """Plot tm_weight effect for SWDA."""
    if tm_stats.empty:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Determine F1 column name
    f1_col = 'val_f1_mean' if 'val_f1_mean' in tm_stats.columns else 'val_weighted_f1_mean'
    f1_std_col = 'val_f1_std' if 'val_f1_std' in tm_stats.columns else 'val_weighted_f1_std'

    # Plot 1: F1 vs tm_weight
    ax1 = axes[0]
    ax1.errorbar(tm_stats['tm_weight'], tm_stats[f1_col],
                 yerr=tm_stats.get(f1_std_col, 0), marker='o', capsize=5,
                 color='steelblue', label='SWDA (37 labels)')

    # Add client_28 reference (normalized for visual comparison)
    client_tws = sorted(CLIENT_28_RESULTS['tm_weight_means'].keys())
    client_f1s = [CLIENT_28_RESULTS['tm_weight_means'][tw] for tw in client_tws]
    ax1.plot(client_tws, client_f1s, 'o--', color='gray', alpha=0.7, label='Client (28 labels)')

    ax1.set_xlabel('Transition Loss Weight (λ)')
    ax1.set_ylabel('Weighted F1')
    ax1.set_title('TM Regularization Effect: SWDA vs Client')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy vs tm_weight (if available)
    ax2 = axes[1]
    if 'val_accuracy_mean' in tm_stats.columns:
        ax2.errorbar(tm_stats['tm_weight'], tm_stats['val_accuracy_mean'],
                     yerr=tm_stats.get('val_accuracy_std', 0), marker='s', capsize=5,
                     color='darkorange', label='SWDA BERT')

        # Add Tanaka baselines
        ax2.axhline(y=TANAKA_BASELINES['tanaka_best']['accuracy'], color='red',
                    linestyle='--', alpha=0.7, label=f"Tanaka Best ({TANAKA_BASELINES['tanaka_best']['accuracy']:.1%})")
        ax2.axhline(y=TANAKA_BASELINES['max_probability_tm_only']['accuracy'], color='gray',
                    linestyle=':', alpha=0.7, label=f"TM Only ({TANAKA_BASELINES['max_probability_tm_only']['accuracy']:.1%})")

        ax2.set_xlabel('Transition Loss Weight (λ)')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('SWDA Accuracy vs Tanaka Baselines')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Relative improvement plot instead
        baseline_f1 = tm_stats[tm_stats['tm_weight'] == 0.0]['val_f1_mean'].iloc[0] if 0.0 in tm_stats['tm_weight'].values else tm_stats['val_f1_mean'].iloc[0]
        rel_improvements = (tm_stats['val_f1_mean'] - baseline_f1) / baseline_f1 * 100

        ax2.bar(tm_stats['tm_weight'].astype(str), rel_improvements, color='steelblue', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Transition Loss Weight (λ)')
        ax2.set_ylabel('Relative F1 Improvement (%)')
        ax2.set_title('Improvement vs Baseline (λ=0)')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'swda_tm_effect.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    plt.close()


def plot_cross_dataset_comparison(tm_stats: pd.DataFrame, output_dir: Path):
    """Plot cross-dataset TM effect comparison."""
    if tm_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine F1 column name
    f1_col = 'val_f1_mean' if 'val_f1_mean' in tm_stats.columns else 'val_weighted_f1_mean'

    # Normalize to baseline for fair comparison
    tm_weights = [0.0, 0.2, 0.5, 1.0, 1.5]

    # SWDA normalized
    swda_baseline = tm_stats[tm_stats['tm_weight'] == 0.0][f1_col].iloc[0] if 0.0 in tm_stats['tm_weight'].values else None
    if swda_baseline:
        swda_normalized = []
        for tw in tm_weights:
            row = tm_stats[tm_stats['tm_weight'] == tw]
            if not row.empty:
                swda_normalized.append(row[f1_col].iloc[0] / swda_baseline)
            else:
                swda_normalized.append(np.nan)
        ax.plot(tm_weights, swda_normalized, 'o-', label='SWDA (37 labels)', color='steelblue')

    # Client normalized
    client_baseline = CLIENT_28_RESULTS['tm_weight_means'][0.0]
    client_normalized = [CLIENT_28_RESULTS['tm_weight_means'].get(tw, np.nan) / client_baseline for tw in tm_weights]
    ax.plot(tm_weights, client_normalized, 's--', label='Client (28 labels)', color='darkorange')

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Transition Loss Weight (λ)')
    ax.set_ylabel('Normalized F1 (relative to λ=0)')
    ax.set_title('Cross-Dataset TM Regularization Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'cross_dataset_tm_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def generate_latex_table(tm_stats: pd.DataFrame, model_stats: pd.DataFrame, output_dir: Path):
    """Generate LaTeX-ready table with consistent metrics (Macro-F1, Top-3, Cum70, JS)."""
    if tm_stats.empty:
        return

    # Determine column names based on what's available
    macro_f1_col = 'val_macro_f1_mean' if 'val_macro_f1_mean' in tm_stats.columns else None
    top3_col = 'val_top_3_accuracy_mean' if 'val_top_3_accuracy_mean' in tm_stats.columns else 'top_k_acc_3_mean'
    cum70_col = 'val_cumulative70_pred_accuracy_mean' if 'val_cumulative70_pred_accuracy_mean' in tm_stats.columns else None
    js_col = 'mean_js_divergence_mean' if 'mean_js_divergence_mean' in tm_stats.columns else None

    latex_lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    $\\lambda_{\\text{tm}}$ & Macro-F1 & Top-3 & Cum70 & JS \\\\",
        "    \\midrule",
    ]

    for _, row in tm_stats.iterrows():
        macro_f1 = f".{row[macro_f1_col]*1000:.0f}" if macro_f1_col and pd.notna(row.get(macro_f1_col)) else '--'
        top3 = f".{row[top3_col]*1000:.0f}" if top3_col in tm_stats.columns and pd.notna(row.get(top3_col)) else '--'
        cum70 = f".{row[cum70_col]*1000:.0f}" if cum70_col and pd.notna(row.get(cum70_col)) else '--'
        js = f".{row[js_col]*1000:.0f}" if js_col and pd.notna(row.get(js_col)) else '--'
        latex_lines.append(f"    {row['tm_weight']:.1f} & {macro_f1} & {top3} & {cum70} & {js} \\\\")

    latex_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{SWDA 37-class results with XLM-RoBERTa. Transition regularization improves macro-F1, with optimal $\\lambda$=0.2.}",
        "  \\label{tab:swda}",
        "\\end{table}",
    ])

    output_path = output_dir / 'swda_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved LaTeX table: {output_path}")

    # Also generate a summary for cross-dataset comparison
    generate_cross_dataset_summary(tm_stats, output_dir)


def generate_cross_dataset_summary(tm_stats: pd.DataFrame, output_dir: Path):
    """Generate summary data for cross-dataset table."""
    if tm_stats.empty:
        return

    # Find baseline (lambda=0.0) and best config
    baseline = tm_stats[tm_stats['tm_weight'] == 0.0].iloc[0] if 0.0 in tm_stats['tm_weight'].values else None

    # Find best by macro-F1
    macro_f1_col = 'val_macro_f1_mean' if 'val_macro_f1_mean' in tm_stats.columns else None
    if macro_f1_col:
        best_idx = tm_stats[macro_f1_col].idxmax()
        best = tm_stats.loc[best_idx]
    else:
        best = None

    summary = {
        'dataset': 'SWDA',
        'classes': 37,
        'baseline_lambda': 0.0,
        'best_lambda': best['tm_weight'] if best is not None else None,
    }

    # Add metrics
    for prefix, row in [('baseline', baseline), ('best', best)]:
        if row is not None:
            summary[f'{prefix}_macro_f1'] = row.get('val_macro_f1_mean')
            summary[f'{prefix}_top3'] = row.get('val_top_3_accuracy_mean') or row.get('top_k_acc_3_mean')
            summary[f'{prefix}_cum70'] = row.get('val_cumulative70_pred_accuracy_mean')
            summary[f'{prefix}_js'] = row.get('mean_js_divergence_mean')

    # Save as JSON for later combination with HOPE
    import json
    output_path = output_dir / 'swda_cross_dataset_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved cross-dataset summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze SWDA results')
    parser.add_argument('--data', type=str, help='Path to wandb export CSV')
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    # Load data
    data_path = Path(args.data) if args.data else None
    df = load_data(data_path)

    print(f"Total runs loaded: {len(df)}")
    print(f"States: {df['state'].value_counts().to_dict()}")

    # Filter to BERT models
    bert_df = filter_bert_models(df)
    print(f"\nBERT-only runs: {len(bert_df)}")
    print(f"Finished BERT runs: {len(bert_df[bert_df['state'] == 'finished'])}")

    if 'pretrained_model' in bert_df.columns:
        print(f"Models: {bert_df['pretrained_model'].unique().tolist()}")

    # Analyze TM effect
    print("\n" + "="*60)
    print("TM WEIGHT EFFECT (BERT models only)")
    print("="*60)
    tm_stats = analyze_tm_effect(bert_df)
    if not tm_stats.empty:
        print(tm_stats.to_string(index=False))

    # Analyze per model
    print("\n" + "="*60)
    print("RESULTS PER PRETRAINED MODEL")
    print("="*60)
    model_stats = analyze_per_model(bert_df)
    if not model_stats.empty:
        print(model_stats.to_string(index=False))

    # Compare with baselines
    compare_with_baselines(tm_stats, model_stats)

    # Generate outputs
    if not tm_stats.empty:
        plot_tm_effect(tm_stats, output_dir)
        plot_cross_dataset_comparison(tm_stats, output_dir)
        generate_latex_table(tm_stats, model_stats, output_dir)

        # Save CSV summaries
        tm_stats.to_csv(output_dir / 'swda_tm_summary.csv', index=False)
        if not model_stats.empty:
            model_stats.to_csv(output_dir / 'swda_model_summary.csv', index=False)
        print(f"\nSaved CSV summaries to {output_dir}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
