#!/usr/bin/env python3
"""
Create a small example transition matrix heatmap with translated labels
for the ACL paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read the transition matrix
tm_path = Path(__file__).parent.parent / "data/oncoco/cv_splits/fold_0/transition_matrix.csv"
df = pd.read_csv(tm_path, index_col=0)

# Clean up the index and columns (remove the arrow prefix like "2→")
df.index = [idx.split("→")[-1] if "→" in str(idx) else idx for idx in df.index]

# Create short English translations for selected categories
# Using actual category names from OnCoCo taxonomy (with line breaks)
translations = {
    # Counselor categories (B-) - 3 key acts
    "B-WF-AK-RS-FPP-*": "C: Precise\nInquiry",       # punktuelle, präzise Nachfrage
    "B-WF-AK-RS-ERx-*": "C: Simple\nReflection",    # Einfache Reflexion
    "B-FZ-*-*-V-*": "C: Farewell",                  # Verabschiedung
    # Client categories (K-) - 3 key acts
    "K-WF-AKP-*-PDar-*": "Cl: Problem\nDescription", # Problemdarstellung
    "K-WF-AKP-*-Zust-*": "Cl: Agreement",           # Zustimmung
    "K-FZ-*-*-F-*": "Cl: Formal\nClosing",          # Formales zum Abschluss
}

# Get the subset of categories that we want to visualize
selected_cats = list(translations.keys())

# Filter the dataframe to only include selected categories
df_subset = df.loc[
    [c for c in selected_cats if c in df.index],
    [c for c in selected_cats if c in df.columns]
]

# Rename using translations
df_subset.index = [translations.get(c, c) for c in df_subset.index]
df_subset.columns = [translations.get(c, c) for c in df_subset.columns]

print(f"Matrix shape: {df_subset.shape}")
print(f"Categories: {list(df_subset.index)}")

# Create the heatmap - compact size for ACL column
fig, ax = plt.subplots(figsize=(4.5, 3.8))

# Use a professional blue colormap suitable for academic papers
cmap = sns.color_palette("Blues", as_cmap=True)

# Create annotation labels - hide zeros
annot_labels = df_subset.applymap(lambda x: f"{x:.2f}" if x > 0.005 else "")

# Create heatmap
heatmap = sns.heatmap(
    df_subset,
    annot=annot_labels,
    fmt="",
    cmap=cmap,
    vmin=0,
    vmax=0.6,
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    annot_kws={"size": 8},
    cbar_kws={"label": "P(next|current)", "shrink": 0.7}
)

# Rotate labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# Labels
ax.set_xlabel("Next Act", fontsize=9)
ax.set_ylabel("Current Act", fontsize=9)

plt.tight_layout()

# Save the figure
output_path = Path(__file__).parent.parent / "latex/figures/transition_matrix_example.pdf"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved to {output_path}")

# Also save as PNG for preview
output_png = output_path.with_suffix('.png')
plt.savefig(output_png, bbox_inches='tight', dpi=300)
print(f"Saved to {output_png}")

plt.close()
