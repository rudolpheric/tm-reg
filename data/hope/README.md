# HOPE Counselling Dataset

## About
The HOPE dataset contains counselling conversations annotated with 15 dialogue act categories for mental health support conversations.

## Prerequisites

Download the HOPE dataset from the original source:
- Paper: Malhotra et al. (2022) "Speaker and Time-aware Joint Contextual Learning for Dialogue-act Classification in Counselling Conversations"
- The dataset should be placed in this directory with the following structure:
  ```
  hope/
  ├── Train/
  ├── Validation/
  └── Test/
  ```

## Preprocessing

After downloading the dataset, run the preprocessing script:

```bash
python preprocess_hope.py
```

This will:
1. Load conversations from Train/Validation/Test directories
2. Map speaker labels (T → Counselor, P → Client)
3. Apply sliding window preprocessing (window size 5)
4. Compute transition matrix for TM regularization
5. Save preprocessed files to `hope_preprocessed/`

## Output Files
- `train.json`, `val.json`, `test.json`: Preprocessed splits
- `transition_matrix.csv`: Empirical transition probabilities (12x12)
- `label_encoder.pkl`: Label encoding mapping
- `split_info.json`: Split statistics

## Dialogue Act Categories
The dataset uses 12 canonical dialogue act labels:
- `id`: Information Delivery
- `irq`: Information Request (Yes/No)
- `gc`: General Chat
- `crq`: Clarification Request
- `cd`: Clarification Delivery
- `yq`: Yes/No Question
- `ack`: Acknowledgement
- `op`: Opinion
- `gt`: Greeting
- `on`: Other Neutral
- `od`: Other Directive
- `orq`: Open Request

## Citation
```bibtex
@inproceedings{malhotra2022speaker,
  title={Speaker and Time-aware Joint Contextual Learning for Dialogue-act Classification in Counselling Conversations},
  author={Malhotra, Ganeshan and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
