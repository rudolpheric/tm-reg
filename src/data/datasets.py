import pandas as pd
import torch
import pickle
import random
import re
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

class DialogueNextSentenceDataset(Dataset):
    """
    Loads preprocessed conversation pairs for next category and speaker prediction.
    
    This implementation includes a simplified label format that replaces cryptic codes with
    human-readable descriptions in the conversation history:
    
    Original format:
        'K (K-WF-AKP-*-PPers-* | Preisgeben persönlicher Daten): Hallo hier ist Elke'
        'B (B-FA-*-*-*-* | Gesprächseröffnung): Hallo Elke. Ich bin Marie und Beraterin.'
    
    Simplified format:
        '(Preisgeben persönlicher Daten) Klient: Hallo hier ist Elke'
        '(Gesprächseröffnung) Berater: Hallo Elke. Ich bin Marie und Beraterin.'
    
    Each item in the dataset focuses on:
      - Input: conversation history (optionally using only the last x sentences)
      - Target: the next utterance's category and a binary speaker change indicator.
    """
    def __init__(self, dataset_dir, tokenizer, max_input_length=512, num_history_sentences=None,
                 min_category_count=None, sample_fraction=None, random_seed=42):
        """
        Args:
            dataset_dir (str or Path): Path to the directory with dataset files.
            tokenizer: Tokenizer to be used for converting text to tokens.
            max_input_length (int): Maximum length for tokenization.
            num_history_sentences (int or None): If set, only use the last x sentences from history.
            min_category_count (int or None): Filter out categories with less than this count.
            sample_fraction (float in (0,1] or None): Use only a fraction of the dataset if provided.
            random_seed (int): Seed used for reproducible sampling.
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.num_history_sentences = num_history_sentences  # New hyperparameter

        # Load conversation pairs and label encoder from pickle files
        dataset_dir = Path(dataset_dir)
        with open(dataset_dir / "conversation_pairs.pkl", "rb") as f:
            self.pairs = pickle.load(f)
        
        with open(dataset_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # If a minimum category count is specified, filter the dataset
        if min_category_count is not None:
            self.filter_by_category_frequency(min_count=min_category_count)

        # Optionally sample only a fraction of the dataset for testing purposes
        if sample_fraction is not None and 0 < sample_fraction < 1:
            # For reproducibility, set the random seed
            random.seed(random_seed)
            total_samples = len(self.pairs)
            sample_size = int(total_samples * sample_fraction)
            # You might choose to shuffle prior to sampling to avoid any ordering bias.
            random.shuffle(self.pairs)
            self.pairs = self.pairs[:sample_size]
            print(f"Dataset sampled: {sample_size} samples selected out of {total_samples}")

    def filter_by_category_frequency(self, min_count):
        """Filters out samples with target categories occurring less than min_count times and refits the label encoder."""
        # Count category occurrences
        category_counts = {}
        for pair in self.pairs:
            cat = pair["target_category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Determine valid and invalid categories
        valid_categories = {cat for cat, count in category_counts.items() if count >= min_count}
        invalid_categories = {cat for cat, count in category_counts.items() if count < min_count}
        print(f"Categories with at least {min_count} occurrences: {valid_categories}")
        print(f"Categories with less than {min_count} occurrences: {invalid_categories}")

        # Filter pairs based on valid categories
        filtered_pairs = [pair for pair in self.pairs if pair["target_category"] in valid_categories]
        print(f"Dataset reduced from {len(self.pairs)} to {len(filtered_pairs)} samples after filtering by category.")
        self.pairs = filtered_pairs

        # Update the label encoder to only include the valid categories
        new_label_encoder = LabelEncoder()
        new_label_encoder.fit(list(valid_categories))
        self.label_encoder = new_label_encoder

    def __len__(self):
        return len(self.pairs)

    def transform_history(self, history):
        """
        Transform the history format to simplify the labels.
        
        Converts the format from:
            'K (K-WF-AKP-*-PPers-* | Preisgeben persönlicher Daten): Hallo hier ist Elke'
        to:
            '(Preisgeben persönlicher Daten) Klient: Hallo hier ist Elke'
        
        This makes the conversation more readable and focuses on the meaningful description
        rather than the cryptic code, while preserving the speaker role information.
        
        Args:
            history (str): Original conversation history with cryptic labels
            
        Returns:
            str: Transformed conversation history with simplified human-readable labels
        """
        # Process line by line
        lines = history.split("\n")
        transformed_lines = []
        
        for line in lines:
            # This regex will match patterns like "K (K-WF-AKP-*-PPers-* | Preisgeben persönlicher Daten):"
            regex = r'([KB]) \(([^|]+)\| ([^)]+)\): (.*)'
            match = re.match(regex, line)
            
            if match:
                speaker, code, description, text = match.groups()
                speaker_type = "Klient" if speaker == "K" else "Berater"
                transformed_line = f"({description.strip()}) {speaker_type}: {text}"
                transformed_lines.append(transformed_line)
            else:
                # If the line doesn't match our pattern, keep it as is
                transformed_lines.append(line)
                
        return "\n".join(transformed_lines)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        history_old = pair["history"]

        # Transform the history format before any other processing
        # history = self.transform_history(history_old)
        history = history_old
        # If only a subset of sentences is required, only use the last num_history_sentences
        if self.num_history_sentences is not None:
            sentences = history.split("\n")
            if len(sentences) > self.num_history_sentences:
                history = "\n".join(sentences[-self.num_history_sentences:])

        self.tokenizer.truncation_side = 'left'
        inputs = self.tokenizer(
            history,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_input_length,
            truncation=True,
        )

        # Get classification label
        target_category_encoded = self.label_encoder.transform([pair["target_category"]])[0]
        target_category_tensor = torch.tensor(target_category_encoded, dtype=torch.long)
        target_speaker_change_tensor = torch.tensor(pair["target_speaker_change"], dtype=torch.float)

        last_utterance = history_old.strip().split("\n")[-1] # use old history for source category extraction

        # Extract the category code and description using two-step regex approach
        # Step 1: Extract code (everything before |)
        # Pattern matches: "K (K-WF-AKP-*-PPers-* | ...)"
        code_match = re.search(r'[KB] \(([^|]+)\|', last_utterance)
        if code_match:
            code = code_match.group(1).strip()
            # Step 2: Extract description (everything between | and ):)
            # This handles descriptions with parentheses like "(weitere) Nutzung ..."
            desc_match = re.search(r'\|\s*(.+?)\):', last_utterance)
            if desc_match:
                description = desc_match.group(1).strip()
                source_category = f"{code} | {description}"
            else:
                source_category = None  # fallback if description parsing fails
        else:
            source_category = None  # fallback if code parsing fails

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "raw_history": history,  # Now contains the transformed history
            "target_category_name": pair["target_category"],
            "category_label": target_category_tensor,
            "speaker_change_label": target_speaker_change_tensor,
            "source_category": source_category  # Now this field contains the simplified category
        }
        
class CategoryHierarchyDataset(Dataset):
    def __init__(self, base_dataset, hierarchy, max_utterances=5, max_length=128):
        self.base_dataset = base_dataset
        self.hierarchy = hierarchy
        # Ensure max_utterances is not None
        self.max_utterances = max_utterances if max_utterances is not None else 5
        self.max_length = max_length
        self.tokenizer = base_dataset.tokenizer
        self.label_encoder = base_dataset.label_encoder
    
    def __len__(self):
        return len(self.base_dataset)
    
    def parse_structured_code(self, code_text):
        """
        Parse a structured code like K-WF-HP-*-PosR-* and extract components.
        
        Returns dictionary with component indices based on the hierarchy.
        """
        # Clean up the code text (remove whitespace, etc.)
        clean_code = code_text.strip()
        
        # Check if this code exists in our hierarchy
        if clean_code in self.hierarchy["full_codes"]:
            return self.hierarchy["full_codes"][clean_code]
        
        # If not found, try to parse it manually
        components = clean_code.split("-")
        
        # Extract each level
        speaker = components[0] if len(components) > 0 else ""
        k1 = components[1] if len(components) > 1 and components[1] != "*" else ""
        k2 = components[2] if len(components) > 2 and components[2] != "*" else ""
        
        # Find the first non-* component after k2 for k3
        k3 = ""
        if len(components) > 3:
            for i in range(3, len(components)):
                if components[i] != "*":
                    k3 = components[i]
                    break
        
        # Find the second non-* component after k2 for k4
        k4 = ""
        found_k3 = False
        if len(components) > 3:
            for i in range(3, len(components)):
                if components[i] != "*":
                    if not found_k3:
                        found_k3 = True
                    else:
                        k4 = components[i]
                        break
        
        # Map to indices
        return {
            "speaker": self.hierarchy["speaker_types"].get(speaker, 0),
            "main_category": self.hierarchy["main_categories"].get(k1, 0),
            "sub_category": self.hierarchy["sub_categories"].get(k2, 0),
            "third_level": self.hierarchy["third_level"].get(k3, 0),
            "fourth_level": self.hierarchy["fourth_level"].get(k4, 0)
        }
    
    def parse_utterance(self, utterance):
        """
        Parse an utterance to extract code and text.
        """
        # Example: "K (K-WF-AKP-*-PDar-* | Problemdarstellung): Ich hatte einfach das Gefühl..."
        pattern = r'([KB])\s+\(([^|]+)(?:\|\s*([^)]+))?\):\s*(.+)'
        match = re.search(pattern, utterance)
        
        if match:
            speaker, code, description, text = match.groups()
            return {
                "speaker": speaker,
                "code": code.strip(),
                "description": description.strip() if description else "",
                "text": text.strip()
            }
        else:
            # Fallback
            return {
                "speaker": "K" if "Klient" in utterance else "B",
                "code": "",
                "description": "",
                "text": utterance
            }
    
    def __getitem__(self, idx):
        # Get base sample
        base_sample = self.base_dataset[idx]
        
        # Extract conversation history
        history = base_sample["raw_history"]
        utterance_texts = history.split("\n")
        
        # Take last max_utterances
        utterance_texts = utterance_texts[-self.max_utterances:] if len(utterance_texts) > self.max_utterances else utterance_texts
        
        # Process each utterance
        utterances = []
        structured_codes = []
        
        for utterance_text in utterance_texts:
            parsed = self.parse_utterance(utterance_text)
            
            # Tokenize utterance
            encoded = self.tokenizer(
                parsed["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            
            utterances.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0)
            })
            
            # Parse the structured code
            structured_code = self.parse_structured_code(parsed["code"])
            structured_codes.append(structured_code)
        
        # Pad if necessary
        while len(utterances) < self.max_utterances:
            # Create empty utterance
            empty = self.tokenizer(
                "",
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            
            utterances.append({
                "input_ids": empty["input_ids"].squeeze(0),
                "attention_mask": empty["attention_mask"].squeeze(0)
            })
            
            # Add empty structured code
            structured_codes.append({
                "speaker": 0,
                "main_category": 0,
                "sub_category": 0,
                "third_level": 0,
                "fourth_level": 0
            })
        
        # Stack utterances
        input_ids = torch.stack([u["input_ids"] for u in utterances])
        attention_mask = torch.stack([u["attention_mask"] for u in utterances])
        
        # Extract components for easy access
        speaker_ids = torch.tensor([sc["speaker"] for sc in structured_codes], dtype=torch.long)
        main_cat_ids = torch.tensor([sc["main_category"] for sc in structured_codes], dtype=torch.long)
        sub_cat_ids = torch.tensor([sc["sub_category"] for sc in structured_codes], dtype=torch.long)
        third_level_ids = torch.tensor([sc["third_level"] for sc in structured_codes], dtype=torch.long)
        fourth_level_ids = torch.tensor([sc["fourth_level"] for sc in structured_codes], dtype=torch.long)
        
        # Parse target code (extract components for multi-task learning)
        target_category_name = base_sample["target_category_name"]
        if " | " in target_category_name:
            target_code = target_category_name.split(" | ")[0]
        else:
            target_code = target_category_name
            
        target_components = self.parse_structured_code(target_code)
        
        # Add K1-K5 parameter names for refactored model compatibility
        result = {
            # Original format for compatibility 
            "input_ids": base_sample["input_ids"],
            "attention_mask": base_sample["attention_mask"],
            "category_label": base_sample["category_label"],
            "raw_history": base_sample["raw_history"],
            "target_category_name": base_sample["target_category_name"],
            "source_category": base_sample.get("source_category", None),
            
            # Hierarchical format
            "hierarchical_input_ids": input_ids,
            "hierarchical_attention_mask": attention_mask,
            
            # Legacy structured components (for backward compatibility)
            "speaker_ids": speaker_ids,
            "main_cat_ids": main_cat_ids,
            "sub_cat_ids": sub_cat_ids,
            "third_level_ids": third_level_ids,
            "fourth_level_ids": fourth_level_ids,
            
            # NEW: K1-K5 parameter names for refactored model
            "k1_ids": speaker_ids,        # K1 = Speaker
            "k2_ids": main_cat_ids,       # K2 = Main Category  
            "k3_ids": sub_cat_ids,        # K3 = Sub Category
            "k4_ids": third_level_ids,    # K4 = Third Level
            "k5_ids": fourth_level_ids,   # K5 = Fourth Level
            
            # Target component labels for multi-task learning (legacy names)
            "target_speaker": torch.tensor(target_components["speaker"], dtype=torch.long),
            "target_main_cat": torch.tensor(target_components["main_category"], dtype=torch.long),
            "target_sub_cat": torch.tensor(target_components["sub_category"], dtype=torch.long),
            "target_third_level": torch.tensor(target_components["third_level"], dtype=torch.long),
            "target_fourth_level": torch.tensor(target_components["fourth_level"], dtype=torch.long),
            
            # NEW: Target component labels with K1-K5 naming
            "target_k1": torch.tensor(target_components["speaker"], dtype=torch.long),
            "target_k2": torch.tensor(target_components["main_category"], dtype=torch.long),
            "target_k3": torch.tensor(target_components["sub_category"], dtype=torch.long),
            "target_k4": torch.tensor(target_components["third_level"], dtype=torch.long),
            "target_k5": torch.tensor(target_components["fourth_level"], dtype=torch.long)
        }

        return result


class OnCoCoDataset(Dataset):
    """
    Dataset for OnCoCo counselling conversations using JSON-based CV splits.

    This dataset loads conversation history and target labels from JSON files
    created during cross-validation split generation.

    Each sample contains:
        - conversation_history: The dialogue history as a string
        - label: The target category code (e.g., "K-WF-AKP-*-PDar-*")
    """

    def __init__(self, data: list, tokenizer, label_encoder, max_length: int = 512,
                 num_history_sentences: int = None):
        """
        Args:
            data: List of dictionaries with 'conversation_history' and 'label' keys
            tokenizer: HuggingFace tokenizer for encoding text
            label_encoder: Sklearn LabelEncoder for converting labels to indices
            max_length: Maximum sequence length for tokenization
            num_history_sentences: If set, only use the last N sentences from history
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.num_history_sentences = num_history_sentences

    def __len__(self):
        return len(self.data)

    def _extract_source_category(self, history: str) -> str:
        """Extract the source category from the last utterance in history."""
        if not history:
            return None

        lines = history.strip().split('\n')
        if not lines:
            return None

        last_line = lines[-1]

        # Pattern: "K (K-WF-AKP-*-PPers-* | Description): text"
        code_match = re.search(r'[KB] \(([^|]+)\|', last_line)
        if code_match:
            code = code_match.group(1).strip()
            desc_match = re.search(r'\|\s*(.+?)\):', last_line)
            if desc_match:
                description = desc_match.group(1).strip()
                return f"{code} | {description}"

        return None

    def __getitem__(self, idx):
        item = self.data[idx]
        history = item['conversation_history']
        label = item['label']

        # Optionally limit history length
        if self.num_history_sentences is not None:
            sentences = history.split('\n')
            if len(sentences) > self.num_history_sentences:
                history = '\n'.join(sentences[-self.num_history_sentences:])

        # Tokenize with left truncation to keep most recent context
        self.tokenizer.truncation_side = 'left'
        encoding = self.tokenizer(
            history,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )

        # Convert label to index
        label_idx = self.label_encoder.transform([label])[0]

        # Extract source category
        source_category = self._extract_source_category(item['conversation_history'])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_idx, dtype=torch.long),
            'source_categories': source_category,
            'raw_history': history,
            'target_category_name': label,
        }