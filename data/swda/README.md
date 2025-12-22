# Switchboard Dialogue Act Corpus (SWDA)

## About
The Switchboard Dialogue Act Corpus extends the original Switchboard corpus with dialogue act annotations using the DAMSL tag set.

## Preprocessing

Run the preprocessing script to download and prepare the SWDA dataset:

```bash
python preprocess_swda.py
```

This will:
1. Download SWDA from HuggingFace (`swda` dataset)
2. Filter to 37 categories (excluding rare ones with < 50 occurrences)
3. Create train/val/test splits (80/10/10) at conversation level
4. Compute transition matrix for TM regularization
5. Save preprocessed files to `swda_preprocessed/`

## Output Files
- `train.json`, `val.json`, `test.json`: Preprocessed splits
- `transition_matrix.csv`: Empirical transition probabilities
- `label_encoder.pkl`: Label encoding mapping
- `split_info.json`: Split statistics

## Citation
```bibtex
@article{stolcke2000dialogue,
  title={Dialogue act modeling for automatic tagging and recognition of conversational speech},
  author={Stolcke, Andreas and Ries, Klaus and Coccaro, Noah and Shriberg, Elizabeth and Bates, Rebecca and Jurafsky, Daniel and Taylor, Paul and Martin, Rachel and Van Ess-Dykema, Carol and Meteer, Marie},
  journal={Computational linguistics},
  volume={26},
  number={3},
  pages={339--373},
  year={2000}
}
```
