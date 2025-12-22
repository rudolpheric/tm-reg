import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import pandas as pd
import math
from typing import Dict, Optional, List, Tuple


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy Loss with label smoothing and temperature scaling.
    """
    def __init__(self, smoothing: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply temperature scaling to logits before softmax
        scaled_pred = pred / self.temperature
        log_probs = torch.log_softmax(scaled_pred, dim=-1)
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance with temperature scaling.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. In Proceedings of the IEEE international
    conference on computer vision (pp. 2999-3007).
    
    Args:
        alpha (float or torch.Tensor): Weighting factor for rare class (default=1.0).
            If tensor, should have shape (num_classes,) for per-class weights.
        gamma (float): Focusing parameter to down-weight easy examples (default=2.0).
        temperature (float): Temperature scaling parameter (default=1.0).
        reduction (str): Specifies the reduction to apply to the output: 
            'none' | 'mean' | 'sum' (default='mean').
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, temperature: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for focal loss computation.
        
        Args:
            inputs: Predicted logits with shape (N, C) where N is batch size and C is number of classes
            targets: Ground truth class indices with shape (N,)
            
        Returns:
            Focal loss value
        """
        # Apply temperature scaling to logits
        scaled_inputs = inputs / self.temperature
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(scaled_inputs, targets, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            # Per-class alpha weights
            alpha_t = self.alpha[targets]
        else:
            # Single alpha value
            alpha_t = self.alpha
            
        focal_loss = alpha_t * focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class TransitionMatrixConstraint(nn.Module):
    """
    Constrains model predictions to only allow valid transitions from the transition matrix.

    This module masks out invalid transitions by setting their logits to a large negative value,
    effectively forcing the model to only predict categories that have been observed to follow
    the current category with probability above the threshold.
    """
    def __init__(self, transition_matrix, label_encoder, threshold=0.01, mask_value=-50.0):
        """
        Args:
            transition_matrix: pandas DataFrame with transition probabilities
            label_encoder: LabelEncoder to map category names to indices
            threshold: Minimum probability for a transition to be considered valid (default=0.01)
            mask_value: Value to set for invalid transitions (default=-1e9)
        """
        super().__init__()
        self.threshold = threshold
        self.mask_value = mask_value

        # Helper to extract code-only identifier: left of '|'
        def _code(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            parts = str(s).split('|', 1)
            return parts[0].strip()

        # Precompute target class codes for label encoder classes (columns)
        self.target_class_codes: List[str] = [_code(c) for c in label_encoder.classes_]

        # Build allowed target-code sets per source code from transition_matrix
        allowed_codes_by_source: Dict[str, set] = {}
        # Iterate over all rows/cols and collect codes above threshold
        for row_name in transition_matrix.index:
            src_code = _code(row_name)
            if not src_code:
                continue
            allowed = allowed_codes_by_source.setdefault(src_code, set())
            row = transition_matrix.loc[row_name]
            for col_name, prob in row.items():
                try:
                    p = float(prob)
                except Exception:
                    continue
                if p >= self.threshold:
                    tgt_code = _code(col_name)
                    if tgt_code:
                        allowed.add(tgt_code)

        # Convert to a mapping usable at runtime
        self.allowed_codes_by_source = allowed_codes_by_source

    def _build_batch_mask(self, source_categories: List[Optional[str]], device: torch.device) -> torch.Tensor:
        """Create a batch mask [batch, num_classes] where 1=allowed, 0=masked based on code-level mapping."""
        num_classes = len(self.target_class_codes)
        batch_mask = torch.zeros(len(source_categories), num_classes, device=device)

        for i, src in enumerate(source_categories):
            if src is None:
                # Unknown source → allow all
                batch_mask[i] = 1.0
                continue
            src_code = str(src).split('|', 1)[0].strip()
            allowed_codes = self.allowed_codes_by_source.get(src_code)
            if not allowed_codes:
                # If source code not in TM → allow all
                batch_mask[i] = 1.0
                continue
            # Mark allowed indices by matching target class codes
            for j, tgt_code in enumerate(self.target_class_codes):
                if tgt_code in allowed_codes:
                    batch_mask[i, j] = 1.0
        return batch_mask

    def forward(self, pred_logits, source_categories, label_encoder):
        """
        Apply transition constraints to prediction logits.

        Args:
            pred_logits: Model predictions [batch_size, num_classes]
            source_categories: List of source category names for each sample
            label_encoder: LabelEncoder to map category names to indices

        Returns:
            Masked logits with invalid transitions set to mask_value
        """
        batch_size = pred_logits.size(0)
        device = pred_logits.device

        # Handle case where source_categories is None or empty
        if source_categories is None or len(source_categories) == 0:
            return pred_logits  # Return unmasked logits

        # Create a mask for this batch using code-only mapping
        batch_mask = self._build_batch_mask(source_categories, device)

        # CRITICAL FIX: Check if any sample has NO valid transitions
        # If a sample has all zeros in batch_mask, it means no valid transitions exist
        # In this case, allow all transitions to prevent NaN loss
        num_valid_per_sample = batch_mask.sum(dim=1)
        zero_valid_mask = (num_valid_per_sample == 0)
        if zero_valid_mask.any():
            # For samples with no valid transitions, allow all transitions
            batch_mask[zero_valid_mask] = 1.0
            print(f"Warning: {zero_valid_mask.sum().item()} samples have no valid transitions at threshold {self.threshold}. Allowing all transitions for these samples.")

        # Apply mask: keep valid transitions, set invalid ones to mask_value
        masked_logits = pred_logits.clone()
        masked_logits[batch_mask == 0] = self.mask_value

        return masked_logits


class TransitionMatrixLoss(nn.Module):
    """
    Transition Matrix Loss that encourages predictions to follow observed transition patterns.

    This loss computes the KL divergence between model predictions and historical transition
    probabilities from the dataset, encouraging the model to respect conversation flow patterns.
    """
    def __init__(self, transition_matrix, label_encoder, temperature=1.0, epsilon=1e-8, kl_direction='forward',
                 use_prior=False, prior_weight=0.1, prior_path=None, use_entropy_weighting=False):
        """
        Args:
            transition_matrix: pandas DataFrame with transition probabilities
            label_encoder: LabelEncoder to map category names to indices
            temperature: Temperature for softmax (default=1.0)
            epsilon: Small value to avoid numerical issues (default=1e-8)
            kl_direction: Direction of KL divergence ('forward' for KL(target||pred), 'reverse' for KL(pred||target))
            use_prior: Whether to use a-priori probabilities (default=False)
            prior_weight: Weight for prior mixing in Option C (default=0.1, range [0,1])
            prior_path: Path to CSV file with category priors (required if use_prior=True)
            use_entropy_weighting: Whether to weight TM loss by source category entropy (default=False)
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.kl_direction = kl_direction
        self.use_prior = use_prior
        self.prior_weight = prior_weight
        self.prior_dist = None

        # Helper to extract code-only identifier
        def _code(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            parts = str(s).split('|', 1)
            return parts[0].strip()

        # Precompute:
        # - mapping from target class index -> code
        # - code-level distributions per source code over target class indices
        self.target_class_codes: List[str] = [_code(c) for c in label_encoder.classes_]

        # Build code-level distributions from transition_matrix
        # Aggregate columns by code, and rows by code (mean across duplicate rows), then map to target classes
        # Build column aggregation
        col_codes: Dict[str, List[str]] = {}
        for col in transition_matrix.columns:
            c = _code(col)
            if not c:
                continue
            col_codes.setdefault(c, []).append(col)

        # Aggregate rows by code
        rows_by_code: Dict[str, List[str]] = {}
        for row in transition_matrix.index:
            rc = _code(row)
            if not rc:
                continue
            rows_by_code.setdefault(rc, []).append(row)

        # Build source_code -> distribution over target class indices
        distrib_by_source: Dict[str, torch.Tensor] = {}
        for src_code, row_names in rows_by_code.items():
            # Average rows if multiple
            row_values = None
            for rn in row_names:
                series = transition_matrix.loc[rn]
                row_values = series.values if row_values is None else (row_values + series.values)
            if row_values is None:
                continue
            row_values = row_values / max(1, len(row_names))

            # Convert to dict col_name -> prob
            row_series = pd.Series(row_values, index=transition_matrix.columns)

            # Aggregate by code for columns
            code_prob: Dict[str, float] = {}
            for code, cols in col_codes.items():
                val = float(row_series[cols].sum()) if len(cols) > 1 else float(row_series[cols[0]])
                code_prob[code] = max(val, 0.0)

            # Map to target class indices via target codes
            vec = torch.zeros(len(self.target_class_codes), dtype=torch.float32)
            for j, tgt_code in enumerate(self.target_class_codes):
                p = code_prob.get(tgt_code, 0.0)
                vec[j] = p
            # Add epsilon and normalize
            vec = vec + epsilon
            if vec.sum() > 0:
                vec = vec / vec.sum()
            distrib_by_source[src_code] = vec

        # Store
        self.distrib_by_source = distrib_by_source

        # Entropy weighting: compute normalized entropy for each source category
        self.use_entropy_weighting = use_entropy_weighting
        self.source_entropy_weights = {}

        if use_entropy_weighting:
            for src_code, dist in self.distrib_by_source.items():
                # Compute entropy: H(p) = -sum(p * log(p))
                entropy = -torch.sum(dist * torch.log(dist + self.epsilon))
                max_entropy = torch.log(torch.tensor(float(len(dist))))
                # Normalize to [0, 1]: high entropy = weight 1, low entropy = weight ~0
                normalized_entropy = (entropy / max_entropy).item()
                self.source_entropy_weights[src_code] = normalized_entropy
            print(f"✓ Entropy weighting enabled. Weights range: "
                  f"{min(self.source_entropy_weights.values()):.3f} - "
                  f"{max(self.source_entropy_weights.values()):.3f}")

        # Load a-priori probabilities if requested
        if self.use_prior:
            if prior_path is None:
                raise ValueError("prior_path must be provided when use_prior=True")
            self.prior_dist = self._load_priors(prior_path, label_encoder, _code)
            print(f"✓ Loaded a-priori probabilities from {prior_path}")

    def _load_priors(self, prior_path: str, label_encoder, _code_fn):
        """
        Load and process a-priori category probabilities.

        Args:
            prior_path: Path to CSV file with priors
            label_encoder: LabelEncoder for category mapping
            _code_fn: Function to extract code from category name

        Returns:
            torch.Tensor: Prior distribution over target class indices
        """
        try:
            priors_df = pd.read_csv(prior_path, index_col=0)
        except Exception as e:
            raise RuntimeError(f"Failed to load priors from {prior_path}: {e}")

        # Extract code from category names and build code-to-probability mapping
        code_probs = {}
        for cat_name, row in priors_df.iterrows():
            code = _code_fn(cat_name)
            if code:
                prob = float(row['prior_probability'])
                # Aggregate by code if multiple descriptions exist
                if code in code_probs:
                    code_probs[code] += prob
                else:
                    code_probs[code] = prob

        # Map to target class indices
        prior_vec = torch.zeros(len(self.target_class_codes), dtype=torch.float32)

        for j, tgt_code in enumerate(self.target_class_codes):
            p = code_probs.get(tgt_code, 0.0)
            prior_vec[j] = p

        # Add epsilon and normalize
        prior_vec = prior_vec + self.epsilon
        prior_vec = prior_vec / prior_vec.sum()

        return prior_vec

    def _create_transition_tensor(self, transition_matrix, label_encoder):
        """Convert pandas transition matrix to PyTorch tensor with proper indexing."""
        num_classes = len(label_encoder.classes_)
        transition_tensor = torch.zeros(num_classes, num_classes)
        
        for i, source_cat in enumerate(label_encoder.classes_):
            if source_cat in transition_matrix.index:
                for j, target_cat in enumerate(label_encoder.classes_):
                    if target_cat in transition_matrix.columns:
                        transition_tensor[i, j] = transition_matrix.loc[source_cat, target_cat]
        
        # Add epsilon to avoid zero probabilities and normalize
        transition_tensor = transition_tensor + self.epsilon
        transition_tensor = transition_tensor / transition_tensor.sum(dim=1, keepdim=True)
        
        return transition_tensor
    
    def forward(self, pred_logits, source_categories, label_encoder):
        """
        Compute transition matrix loss.

        Args:
            pred_logits: Model predictions [batch_size, num_classes]
            source_categories: List of source category names for each sample
            label_encoder: LabelEncoder to map category names to indices

        Returns:
            Transition matrix loss (KL divergence)
        """
        batch_size = pred_logits.size(0)
        device = pred_logits.device

        # Handle case where source_categories is None or empty
        if source_categories is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Build target distributions per sample from code-level mapping
        tgt_dists = []
        for src in source_categories:
            if src is None:
                # Use prior or uniform if unknown source
                if self.use_prior and self.prior_dist is not None:
                    vec = self.prior_dist  # Option A: Use prior
                else:
                    vec = torch.ones(len(self.target_class_codes), dtype=torch.float32)
                    vec = vec / vec.sum()
            else:
                src_code = str(src).split('|', 1)[0].strip()
                vec = self.distrib_by_source.get(src_code)
                if vec is None:
                    # Fallback to prior or uniform if no distribution for this source code
                    if self.use_prior and self.prior_dist is not None:
                        vec = self.prior_dist  # Option A: Use prior as fallback
                    else:
                        vec = torch.ones(len(self.target_class_codes), dtype=torch.float32)
                        vec = vec / vec.sum()
                elif self.use_prior and self.prior_dist is not None and self.prior_weight > 0:
                    # Option C: Weighted combination of transition + prior
                    vec = (1.0 - self.prior_weight) * vec + self.prior_weight * self.prior_dist
            tgt_dists.append(vec)
        target_transitions = torch.stack(tgt_dists, dim=0).to(device)

        # CRITICAL FIX: Check for NaN/Inf in logits before computing loss
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            print(f"Warning: NaN or Inf detected in pred_logits before TransitionMatrixLoss!")
            print(f"  NaN count: {torch.isnan(pred_logits).sum().item()}")
            print(f"  Inf count: {torch.isinf(pred_logits).sum().item()}")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Clamp logits to prevent extreme values that cause numerical issues
        # -1e9 from masking can cause problems in softmax
        pred_logits_clamped = torch.clamp(pred_logits, min=-50.0, max=50.0)

        # Convert logits to probabilities with numerical stability
        pred_probs = F.softmax(pred_logits_clamped / self.temperature, dim=-1)

        # Check for NaN in probabilities
        if torch.isnan(pred_probs).any():
            print(f"Warning: NaN detected in pred_probs after softmax!")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Add epsilon to prevent log(0)
        pred_probs = torch.clamp(pred_probs, min=self.epsilon)
        target_transitions = torch.clamp(target_transitions, min=self.epsilon)

        # Compute KL divergence with numerical stability
        # Two directions available:
        # - 'forward': KL(target || pred) = mode-seeking, penalizes when pred is low where target is high
        # - 'reverse': KL(pred || target) = mean-seeking, penalizes when pred is high where target is low

        # Compute per-sample KL divergence (no reduction) for entropy weighting
        if self.kl_direction == 'reverse':
            # KL(pred || target) = sum_i pred_i * log(pred_i / target_i)
            log_target = target_transitions.log()
            kl_per_sample = (pred_probs * (pred_probs.log() - log_target)).sum(dim=-1)
        else:  # 'forward' (default/original behavior)
            # KL(target || pred) = sum_i target_i * log(target_i / pred_i)
            log_pred = pred_probs.log()
            kl_per_sample = (target_transitions * (target_transitions.log() - log_pred)).sum(dim=-1)

        # Apply entropy weighting if enabled
        if self.use_entropy_weighting and self.source_entropy_weights:
            weights = []
            for src in source_categories:
                src_code = str(src).split('|', 1)[0].strip() if src else None
                weight = self.source_entropy_weights.get(src_code, 1.0) if src_code else 1.0
                weights.append(weight)
            weights = torch.tensor(weights, device=device, dtype=pred_logits.dtype)
            kl_div = (weights * kl_per_sample).sum() / (weights.sum() + self.epsilon)
        else:
            kl_div = kl_per_sample.mean()

        # Final check for NaN in loss
        if torch.isnan(kl_div):
            print(f"Warning: NaN detected in KL divergence!")
            return torch.tensor(0.0, device=device, requires_grad=True)

        return kl_div


class CombinedLoss(nn.Module):
    """
    Combined loss that integrates cross-entropy loss with transition matrix loss.

    This loss combines:
    1. Standard cross-entropy loss for supervised learning
    2. Transition matrix loss to encourage following conversation flow patterns
    3. Optional transition constraints to mask invalid transitions
    """
    def __init__(self, transition_matrix, label_encoder, ce_weight=1.0, tm_weight=0.1,
                 use_label_smoothing=False, use_focal_loss=False, focal_alpha=1.0, focal_gamma=2.0,
                 class_weights=None, smoothing=0.1, ce_temperature=1.0, tm_temperature=1.0,
                 use_constraint=False, constraint_threshold=0.01, kl_direction='forward',
                 use_transition_loss_with_prior=False, prior_weight=0.1, prior_path=None):
        """
        Args:
            transition_matrix: pandas DataFrame with transition probabilities
            label_encoder: LabelEncoder to map category names to indices
            ce_weight: Weight for cross-entropy loss (default=1.0)
            tm_weight: Weight for transition matrix loss (default=0.1)
            use_label_smoothing: Whether to use label smoothing for CE loss
            use_focal_loss: Whether to use focal loss instead of CE loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            class_weights: Optional class weights for CE loss
            smoothing: Label smoothing parameter
            ce_temperature: Temperature scaling for cross-entropy loss (default=1.0)
            tm_temperature: Temperature scaling for transition matrix loss (default=1.0)
            use_constraint: Whether to apply transition matrix constraints to logits (default=False)
            constraint_threshold: Minimum probability for valid transitions (default=0.01)
            kl_direction: Direction of KL divergence ('forward' or 'reverse') (default='forward')
            use_transition_loss_with_prior: Whether to use a-priori probabilities in transition loss (default=False)
            prior_weight: Weight for prior mixing in weighted combination (default=0.1, range [0,1])
            prior_path: Path to CSV file with category priors (required if use_transition_loss_with_prior=True)
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.tm_weight = tm_weight
        self.ce_temperature = ce_temperature
        self.use_constraint = use_constraint

        # Initialize primary loss function with temperature scaling
        if use_focal_loss:
            self.primary_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, temperature=ce_temperature)
        elif use_label_smoothing:
            self.primary_loss = LabelSmoothingCrossEntropy(smoothing=smoothing, temperature=ce_temperature)
        else:
            # For standard CrossEntropyLoss, we'll apply temperature scaling manually in forward
            self.primary_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Initialize transition matrix loss with temperature scaling, KL direction, and prior support
        self.transition_loss = TransitionMatrixLoss(transition_matrix, label_encoder,
                                                    temperature=tm_temperature, kl_direction=kl_direction,
                                                    use_prior=use_transition_loss_with_prior,
                                                    prior_weight=prior_weight, prior_path=prior_path)

        # Initialize transition constraint if enabled
        if self.use_constraint:
            self.transition_constraint = TransitionMatrixConstraint(
                transition_matrix, label_encoder, threshold=constraint_threshold
            )
        
    def forward(self, pred_logits, targets, source_categories, label_encoder):
        """
        Compute combined loss.

        Args:
            pred_logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            source_categories: List of source category names for each sample
            label_encoder: LabelEncoder to map category names to indices

        Returns:
            Combined loss value and dictionary of loss components
        """
        # CRITICAL FIX: Check input for NaN/Inf
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            print(f"ERROR: NaN/Inf in pred_logits at start of CombinedLoss!")
            print(f"  NaN count: {torch.isnan(pred_logits).sum().item()}")
            print(f"  Inf count: {torch.isinf(pred_logits).sum().item()}")
            print(f"  Min: {pred_logits.min().item()}, Max: {pred_logits.max().item()}")
            # Return a dummy loss to prevent training crash
            dummy_loss = torch.tensor(1.0, device=pred_logits.device, requires_grad=True)
            return dummy_loss, {
                'primary_loss': 1.0,
                'transition_loss': 0.0,
                'total_loss': 1.0
            }

        # Apply transition constraint to logits if enabled
        constrained_logits = pred_logits
        if self.use_constraint and source_categories is not None:
            constrained_logits = self.transition_constraint(pred_logits, source_categories, label_encoder)

            # Check if constraint introduced NaN/Inf
            if torch.isnan(constrained_logits).any() or torch.isinf(constrained_logits).any():
                print(f"ERROR: NaN/Inf after applying constraints!")
                print(f"  NaN count: {torch.isnan(constrained_logits).sum().item()}")
                print(f"  Inf count: {torch.isinf(constrained_logits).sum().item()}")
                # Fallback to unconstrained logits
                constrained_logits = pred_logits
                print(f"  Falling back to unconstrained logits")

        # Clamp constrained logits for numerical stability
        # This prevents extreme values from causing NaN in loss computation
        constrained_logits = torch.clamp(constrained_logits, min=-50.0, max=50.0)

        # Compute primary loss (cross-entropy or focal) with constrained logits
        # Apply temperature scaling for standard CrossEntropyLoss
        try:
            if isinstance(self.primary_loss, nn.CrossEntropyLoss):
                scaled_logits = constrained_logits / self.ce_temperature
                primary_loss = self.primary_loss(scaled_logits, targets)
            else:
                # Temperature scaling is handled internally for other loss types
                primary_loss = self.primary_loss(constrained_logits, targets)

            # Check if primary loss is NaN
            if torch.isnan(primary_loss):
                print(f"ERROR: NaN in primary_loss!")
                primary_loss = torch.tensor(1.0, device=pred_logits.device, requires_grad=True)
        except Exception as e:
            print(f"ERROR computing primary loss: {e}")
            primary_loss = torch.tensor(1.0, device=pred_logits.device, requires_grad=True)

        # Compute transition matrix loss (temperature scaling handled internally)
        # Use constrained logits for consistency
        try:
            if source_categories is not None:
                transition_loss = self.transition_loss(constrained_logits, source_categories, label_encoder)
            else:
                # If source categories are not available, skip transition loss
                transition_loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

            # Check if transition loss is NaN
            if torch.isnan(transition_loss):
                print(f"ERROR: NaN in transition_loss!")
                transition_loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        except Exception as e:
            print(f"ERROR computing transition loss: {e}")
            transition_loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

        # Combine losses
        total_loss = self.ce_weight * primary_loss + self.tm_weight * transition_loss

        # Final NaN check
        if torch.isnan(total_loss):
            print(f"ERROR: NaN in total_loss!")
            print(f"  primary_loss: {primary_loss.item()}, transition_loss: {transition_loss.item()}")
            total_loss = torch.tensor(1.0, device=pred_logits.device, requires_grad=True)

        return total_loss, {
            'primary_loss': primary_loss.item() if not torch.isnan(primary_loss) else float('nan'),
            'transition_loss': transition_loss.item() if not torch.isnan(transition_loss) else float('nan'),
            'total_loss': total_loss.item() if not torch.isnan(total_loss) else float('nan')
        }


class ConstrainedModelWrapper(nn.Module):
    """
    Wrapper that applies transition matrix constraints to any model's predictions.

    This wrapper allows you to take any existing model and apply transition constraints
    at inference time without modifying the model itself. Useful for evaluation and
    comparing constrained vs unconstrained predictions.
    """
    def __init__(self, model, transition_matrix, label_encoder, threshold=0.01, apply_constraint=True):
        """
        Args:
            model: The underlying prediction model
            transition_matrix: pandas DataFrame with transition probabilities
            label_encoder: LabelEncoder to map category names to indices
            threshold: Minimum probability for valid transitions (default=0.01)
            apply_constraint: Whether to apply constraints (can be toggled at runtime)
        """
        super().__init__()
        self.model = model
        self.apply_constraint = apply_constraint
        self.constraint = TransitionMatrixConstraint(transition_matrix, label_encoder, threshold=threshold)

    def forward(self, *args, source_categories=None, label_encoder=None, **kwargs):
        """
        Forward pass with optional constraint application.

        Args:
            *args, **kwargs: Arguments passed to the underlying model
            source_categories: List of source category names for constraint application
            label_encoder: LabelEncoder (required if applying constraints)

        Returns:
            Model predictions (constrained if enabled)
        """
        # Get predictions from the underlying model
        logits = self.model(*args, **kwargs)

        # Apply constraints if enabled and source categories are provided
        if self.apply_constraint and source_categories is not None and label_encoder is not None:
            logits = self.constraint(logits, source_categories, label_encoder)

        return logits

    def toggle_constraint(self, enabled=True):
        """Enable or disable constraint application."""
        self.apply_constraint = enabled


class NextCategoryPredictor(nn.Module):
    """
    Model based on EuroBERT for next sentence/categorization prediction.
    """
    def __init__(self, num_categories: int, pretrained_model: str = "EuroBERT/EuroBERT-610m"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Multiple prediction heads for different aspects of dialogue
        self.category_head = nn.Linear(hidden_size, num_categories)

        # Add a speaker context encoder
        self.speaker_encoder = nn.Embedding(2, 64)  # 2 speakers
        self.combined_projection = nn.Linear(hidden_size + 64, hidden_size)

    def forward(self, input_ids, attention_mask, speaker_ids=None, category_history_ids=None, **kwargs):
        # category_history_ids and kwargs are accepted but ignored for compatibility
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Attention pooling
        attention_weights = self.attention(sequence_output)
        context_vector = torch.sum(attention_weights * sequence_output, dim=1)

        # Add speaker information if available
        if speaker_ids is not None:
            speaker_embeds = self.speaker_encoder(speaker_ids)
            context_vector = self.combined_projection(torch.cat([context_vector, speaker_embeds], dim=1))

        category_logits = self.category_head(context_vector)
        return category_logits

# class HierarchyAwarePredictor(nn.Module):
#     def __init__(self, num_categories, hierarchy, pretrained_model="EuroBERT/EuroBERT-610m"):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
#         hidden_size = self.bert.config.hidden_size
        
#         # Store hierarchy info
#         self.hierarchy = hierarchy
        
#         # Utterance encoding
#         self.utterance_attention = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1),
#             nn.Softmax(dim=1)
#         )
        
#         # Conversation-level encoding
#         self.conversation_gru = nn.GRU(
#             hidden_size, hidden_size, bidirectional=True, 
#             batch_first=True, num_layers=2, dropout=0.2
#         )
        
#         # Component embedding layers
#         embed_dim = 64
#         self.speaker_embedding = nn.Embedding(hierarchy["num_speakers"], embed_dim)
#         self.main_cat_embedding = nn.Embedding(hierarchy["num_main_categories"], embed_dim)
#         self.sub_cat_embedding = nn.Embedding(hierarchy["num_sub_categories"], embed_dim)
#         self.third_level_embedding = nn.Embedding(hierarchy["num_third_level"], embed_dim)
#         self.fourth_level_embedding = nn.Embedding(hierarchy["num_fourth_level"], embed_dim)
        
#         # Combined features
#         self.combined_projection = nn.Linear(
#             hidden_size*2 + embed_dim*5, hidden_size
#         )
        
#         # Prediction heads
#         self.category_head = nn.Linear(hidden_size, num_categories)
#         self.speaker_head = nn.Linear(hidden_size, hierarchy["num_speakers"])
#         self.main_cat_head = nn.Linear(hidden_size, hierarchy["num_main_categories"])
#         self.sub_cat_head = nn.Linear(hidden_size, hierarchy["num_sub_categories"])
#         self.third_level_head = nn.Linear(hidden_size, hierarchy["num_third_level"])
#         self.fourth_level_head = nn.Linear(hidden_size, hierarchy["num_fourth_level"])
        
#         # Hierarchical classification gate - predicts whether to use component-wise 
#         # or direct category prediction
#         self.hierarchy_gate = nn.Linear(hidden_size, 1)
        
#         # Optional category transitions
#         self.use_transitions = False
#         if self.use_transitions:
#             self.category_transitions = nn.Parameter(
#                 torch.zeros(num_categories, num_categories)
#             )
    
#     def forward(self, input_ids=None, attention_mask=None, hierarchical_input_ids=None, 
#                 hierarchical_attention_mask=None, speaker_ids=None, main_cat_ids=None, 
#                 sub_cat_ids=None, third_level_ids=None, fourth_level_ids=None):
        
#         # Detect input mode (flat vs hierarchical)
#         is_hierarchical = hierarchical_input_ids is not None
        
#         if not is_hierarchical:
#             # Process flat input
#             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#             sequence_output = outputs.last_hidden_state
            
#             # Attention pooling
#             attention_weights = self.utterance_attention(sequence_output)
#             context_vector = torch.sum(attention_weights * sequence_output, dim=1)
            
#             # Simple category prediction for flat input
#             category_logits = self.category_head(context_vector)
#             return category_logits
        
#         # Process hierarchical input
#         batch_size = hierarchical_input_ids.size(0)
#         num_utterances = hierarchical_input_ids.size(1)
        
#         # Process each utterance
#         utterance_vectors = []
        
#         for i in range(num_utterances):
#             # Get i-th utterance from all batches
#             utterance_input_ids = hierarchical_input_ids[:, i]
#             utterance_attention_mask = hierarchical_attention_mask[:, i]
            
#             # Skip empty utterances (all padding tokens)
#             if (utterance_input_ids == 0).all():
#                 # Create zero vector as placeholder
#                 utterance_vectors.append(torch.zeros(batch_size, self.bert.config.hidden_size, 
#                                                    device=hierarchical_input_ids.device))
#                 continue
            
#             # Process through BERT
#             outputs = self.bert(
#                 input_ids=utterance_input_ids,
#                 attention_mask=utterance_attention_mask
#             )
            
#             # Attention pooling
#             attention_weights = self.utterance_attention(outputs.last_hidden_state)
#             utterance_vector = torch.sum(attention_weights * outputs.last_hidden_state, dim=1)
#             utterance_vectors.append(utterance_vector)
        
#         # Stack utterance vectors [batch_size, num_utterances, hidden_size]
#         utterance_sequence = torch.stack(utterance_vectors, dim=1)
        
#         # Process conversation through GRU
#         conv_output, _ = self.conversation_gru(utterance_sequence)
        
#         # Get the last state
#         conv_state = conv_output[:, -1]
        
#         # Get component embeddings from the last utterance
#         speaker_embed = self.speaker_embedding(speaker_ids[:, -1])
#         main_cat_embed = self.main_cat_embedding(main_cat_ids[:, -1])
#         sub_cat_embed = self.sub_cat_embedding(sub_cat_ids[:, -1])
#         third_level_embed = self.third_level_embedding(third_level_ids[:, -1])
#         fourth_level_embed = self.fourth_level_embedding(fourth_level_ids[:, -1])
        
#         # Combine all features
#         combined_features = torch.cat([
#             conv_state,
#             speaker_embed,
#             main_cat_embed,
#             sub_cat_embed,
#             third_level_embed,
#             fourth_level_embed
#         ], dim=1)
        
#         context_vector = self.combined_projection(combined_features)
        
#         # Hierarchical gate - predict whether to use component-wise or direct prediction
#         gate_value = torch.sigmoid(self.hierarchy_gate(context_vector))
        
#         # Direct category prediction
#         direct_logits = self.category_head(context_vector)
        
#         # Component-wise predictions
#         speaker_logits = self.speaker_head(context_vector)
#         main_cat_logits = self.main_cat_head(context_vector)
#         sub_cat_logits = self.sub_cat_head(context_vector)
#         third_level_logits = self.third_level_head(context_vector)
#         fourth_level_logits = self.fourth_level_head(context_vector)
        
#         # Apply transition scores if enabled
#         if self.use_transitions:
#             # Use the category of the last utterance as the source
#             last_category_idx = self.hierarchy["speaker_types"].get(speaker_ids[:, -1].cpu().numpy()[0], 0)
#             transition_scores = self.category_transitions[last_category_idx]
#             direct_logits = direct_logits + transition_scores
        
#         # Return all outputs for multi-task learning
#         outputs = {
#             # Direct category prediction
#             "category_logits": direct_logits,
            
#             # Component predictions
#             "speaker_logits": speaker_logits,
#             "main_cat_logits": main_cat_logits, 
#             "sub_cat_logits": sub_cat_logits,
#             "third_level_logits": third_level_logits,
#             "fourth_level_logits": fourth_level_logits,
            
#             # Gate value
#             "gate": gate_value
#         }
        
#         # For compatibility with existing code, return direct category logits
#         return direct_logits

# RNN Baseline Models
class SimpleRNNPredictor(nn.Module):
    """
    Simple RNN baseline that processes conversation history as a sequence of tokens.
    """
    def __init__(self, num_categories: int, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Use LSTM instead of basic RNN for better performance
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention mechanism for better context aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_categories)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Embedding layer
        embeddings = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embeddings = self.dropout(embeddings)
        
        # RNN processing
        rnn_output, _ = self.rnn(embeddings)  # [batch, seq_len, hidden_dim*2]
        
        # Apply attention if we have attention mask
        if attention_mask is not None:
            # Mask padded positions
            attention_mask = attention_mask.unsqueeze(-1).float()
            rnn_output = rnn_output * attention_mask
        
        # Attention pooling
        attention_weights = self.attention(rnn_output)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context_vector = torch.sum(attention_weights * rnn_output, dim=1)  # [batch, hidden_dim*2]
        
        # Classification
        logits = self.classifier(context_vector)
        return logits


class HierarchicalRNNPredictor(nn.Module):
    """
    Hierarchical RNN that processes utterances individually, then models conversation flow.
    Similar to the BERT version but using simpler embeddings.
    """
    def __init__(self, num_categories: int, vocab_size: int, embedding_dim: int = 128,
                 utterance_hidden: int = 128, conversation_hidden: int = 256, 
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Utterance-level RNN
        self.utterance_rnn = nn.LSTM(
            embedding_dim, utterance_hidden, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Conversation-level RNN
        self.conversation_rnn = nn.LSTM(
            utterance_hidden * 2, conversation_hidden, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention for utterance encoding
        self.utterance_attention = nn.Sequential(
            nn.Linear(utterance_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Projection layer for flat mode (when no hierarchical structure)
        # Maps utterance encoding to conversation encoding size
        self.flat_projection = nn.Linear(utterance_hidden * 2, conversation_hidden * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(conversation_hidden * 2, conversation_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conversation_hidden, num_categories)
        )
    
    def encode_utterance(self, utterance_ids: torch.Tensor, utterance_mask: torch.Tensor):
        """Encode a single utterance."""
        embeddings = self.embedding(utterance_ids)
        embeddings = self.dropout(embeddings)
        
        # RNN processing
        output, _ = self.utterance_rnn(embeddings)
        
        # Apply mask
        if utterance_mask is not None:
            output = output * utterance_mask.unsqueeze(-1).float()
        
        # Attention pooling
        attention_weights = self.utterance_attention(output)
        attention_weights = F.softmax(attention_weights, dim=1)
        utterance_vector = torch.sum(attention_weights * output, dim=1)
        
        return utterance_vector
    
    def forward(self, input_ids=None, attention_mask=None, hierarchical_input_ids=None, 
                hierarchical_attention_mask=None, **kwargs):
        # Handle both flat and hierarchical inputs
        if hierarchical_input_ids is not None:
            batch_size, num_utterances, seq_len = hierarchical_input_ids.shape
            
            # Encode each utterance
            utterance_vectors = []
            for i in range(num_utterances):
                utterance_vector = self.encode_utterance(
                    hierarchical_input_ids[:, i], 
                    hierarchical_attention_mask[:, i]
                )
                utterance_vectors.append(utterance_vector)
            
            # Stack utterance vectors
            conversation_input = torch.stack(utterance_vectors, dim=1)
            
            # Process conversation
            conv_output, _ = self.conversation_rnn(conversation_input)
            
            # Use the last output
            final_state = conv_output[:, -1]
            
            # Classification
            logits = self.classifier(final_state)
            return logits
        else:
            # Fall back to flat processing (similar to SimpleRNNPredictor)
            embeddings = self.embedding(input_ids)
            embeddings = self.dropout(embeddings)
            
            # Process as single utterance
            output, _ = self.utterance_rnn(embeddings)
            
            if attention_mask is not None:
                output = output * attention_mask.unsqueeze(-1).float()
            
            # Attention pooling
            attention_weights = self.utterance_attention(output)
            attention_weights = F.softmax(attention_weights, dim=1)
            context_vector = torch.sum(attention_weights * output, dim=1)

            # Project to conversation encoding size for classification
            projected = self.flat_projection(context_vector)

            # Classification
            logits = self.classifier(projected)
            return logits


class HierarchicalRNNPredictorWithHistory(nn.Module):
    """
    Hierarchical RNN with Category History Modeling.

    This model extends HierarchicalRNNPredictor by:
    1. Modeling the sequence of previous categories with positional encoding
    2. Cross-attention between text and category histories
    3. Enhanced dialogue flow understanding through temporal patterns

    This provides a fair RNN comparison to HierarchyAwarePredictorWithCategoryHistory (BERT).
    """

    def __init__(self, num_categories: int, vocab_size: int, embedding_dim: int = 128,
                 utterance_hidden: int = 128, conversation_hidden: int = 256,
                 num_layers: int = 1, dropout: float = 0.3, max_history_length: int = 10):
        super().__init__()

        self.num_categories = num_categories
        self.max_history_length = max_history_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Utterance-level RNN
        self.utterance_rnn = nn.LSTM(
            embedding_dim, utterance_hidden, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )

        # Conversation-level RNN
        self.conversation_rnn = nn.LSTM(
            utterance_hidden * 2, conversation_hidden, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )

        # Attention for utterance encoding
        self.utterance_attention = nn.Sequential(
            nn.Linear(utterance_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Category history modeling
        self.category_embedding = nn.Embedding(num_categories + 1, conversation_hidden * 2)  # +1 for padding

        # Positional encoding for category history
        self.register_buffer('category_pos_encoding', self._create_positional_encoding(max_history_length, conversation_hidden * 2))

        # Category sequence attention (self-attention over history)
        self.category_sequence_attn = nn.MultiheadAttention(
            embed_dim=conversation_hidden * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.category_norm = nn.LayerNorm(conversation_hidden * 2)

        # Cross-attention: text queries category history
        self.text_category_cross_attn = nn.MultiheadAttention(
            embed_dim=conversation_hidden * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.text_category_norm = nn.LayerNorm(conversation_hidden * 2)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(conversation_hidden * 4, conversation_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Projection layer for flat mode (when no hierarchical structure)
        # Maps utterance encoding to conversation encoding size
        self.flat_projection = nn.Linear(utterance_hidden * 2, conversation_hidden * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(conversation_hidden * 2, conversation_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conversation_hidden, num_categories)
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def encode_utterance(self, utterance_ids: torch.Tensor, utterance_mask: torch.Tensor):
        """Encode a single utterance."""
        embeddings = self.embedding(utterance_ids)
        embeddings = self.dropout(embeddings)

        # RNN processing
        output, _ = self.utterance_rnn(embeddings)

        # Apply mask
        if utterance_mask is not None:
            output = output * utterance_mask.unsqueeze(-1).float()

        # Attention pooling
        attention_weights = self.utterance_attention(output)
        attention_weights = F.softmax(attention_weights, dim=1)
        utterance_vector = torch.sum(attention_weights * output, dim=1)

        return utterance_vector

    def encode_category_history(self, category_history: torch.Tensor):
        """
        Encode the category history sequence.

        Args:
            category_history: (batch_size, history_length) tensor of category indices
                             Use num_categories as padding index
        """
        batch_size, hist_len = category_history.shape

        # Embed categories
        cat_embeds = self.category_embedding(category_history)  # (B, H, D)

        # Add positional encoding
        cat_embeds = cat_embeds + self.category_pos_encoding[:, :hist_len, :]

        # Create padding mask (True = ignore)
        padding_mask = (category_history == self.num_categories)

        # Self-attention over category history
        cat_attended, _ = self.category_sequence_attn(
            cat_embeds, cat_embeds, cat_embeds,
            key_padding_mask=padding_mask
        )
        cat_attended = self.category_norm(cat_embeds + cat_attended)

        return cat_attended, padding_mask

    def forward(self, input_ids=None, attention_mask=None, hierarchical_input_ids=None,
                hierarchical_attention_mask=None, category_history=None, **kwargs):
        """
        Forward pass with optional category history.

        Args:
            hierarchical_input_ids: (batch, num_utterances, seq_len)
            hierarchical_attention_mask: (batch, num_utterances, seq_len)
            category_history: (batch, history_length) - previous category indices
        """
        # Handle hierarchical input
        if hierarchical_input_ids is not None:
            batch_size, num_utterances, seq_len = hierarchical_input_ids.shape

            # Encode each utterance
            utterance_vectors = []
            for i in range(num_utterances):
                utterance_vector = self.encode_utterance(
                    hierarchical_input_ids[:, i],
                    hierarchical_attention_mask[:, i]
                )
                utterance_vectors.append(utterance_vector)

            # Stack utterance vectors
            conversation_input = torch.stack(utterance_vectors, dim=1)

            # Process conversation
            conv_output, _ = self.conversation_rnn(conversation_input)

            # Use the last output as text representation
            text_repr = conv_output[:, -1]  # (B, D)
        else:
            # Flat processing fallback
            embeddings = self.embedding(input_ids)
            embeddings = self.dropout(embeddings)
            output, _ = self.utterance_rnn(embeddings)

            if attention_mask is not None:
                output = output * attention_mask.unsqueeze(-1).float()

            attention_weights = self.utterance_attention(output)
            attention_weights = F.softmax(attention_weights, dim=1)
            utt_repr = torch.sum(attention_weights * output, dim=1)

            # Project to conversation encoding size
            text_repr = self.flat_projection(utt_repr)

        # Process category history if provided
        if category_history is not None:
            # Encode category history
            cat_repr, cat_padding_mask = self.encode_category_history(category_history)

            # Cross-attention: text attends to category history
            text_repr_expanded = text_repr.unsqueeze(1)  # (B, 1, D)
            cross_attended, _ = self.text_category_cross_attn(
                text_repr_expanded, cat_repr, cat_repr,
                key_padding_mask=cat_padding_mask
            )
            cross_attended = self.text_category_norm(text_repr_expanded + cross_attended)
            cross_attended = cross_attended.squeeze(1)  # (B, D)

            # Fuse text and history representations
            combined = torch.cat([text_repr, cross_attended], dim=-1)
            final_repr = self.fusion(combined)
        else:
            final_repr = text_repr

        # Classification
        logits = self.classifier(final_repr)
        return logits


class TanakaPredictor(nn.Module):
    """
    Tanaka et al. (2019) three-encoder architecture for next-DA prediction.

    Reference: "Dialogue-Act Prediction of Future Responses based on Conversation History"
    ACL Student Research Workshop 2019.

    Components:
    - Utterance Encoder: GRU over word embeddings → utterance vector
    - Context Encoder: GRU over utterance vectors + speaker change tag
    - DA Encoder: GRU over previous DA sequence
    - Classifier: FC layer on concatenated context + DA representations

    This implementation allows integration with transition matrix regularization
    to demonstrate architecture-agnostic benefits of TM loss.
    """

    def __init__(self, num_categories: int, vocab_size: int,
                 word_embed_dim: int = 300, da_embed_dim: int = 100,
                 utterance_hidden: int = 512, context_hidden: int = 512,
                 da_hidden: int = 128, classifier_hidden: int = 100,
                 dropout: float = 0.1):
        super().__init__()

        self.num_categories = num_categories
        self.utterance_hidden = utterance_hidden
        self.context_hidden = context_hidden
        self.da_hidden = da_hidden

        # Word embedding for utterance encoder
        self.word_embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Utterance Encoder: word-level GRU
        self.utterance_encoder = nn.GRU(
            word_embed_dim, utterance_hidden,
            batch_first=True, bidirectional=False
        )

        # Context Encoder: utterance-level GRU
        # Input: utterance_hidden + 1 (for speaker change tag)
        self.context_encoder = nn.GRU(
            utterance_hidden + 1, context_hidden,
            batch_first=True, bidirectional=False
        )

        # DA Encoder: GRU over previous DA sequence
        # +1 for padding index
        self.da_embedding = nn.Embedding(num_categories + 1, da_embed_dim, padding_idx=num_categories)
        self.da_encoder = nn.GRU(
            da_embed_dim, da_hidden,
            batch_first=True, bidirectional=False
        )

        # Classifier: context_hidden + da_hidden → num_categories
        self.classifier = nn.Sequential(
            nn.Linear(context_hidden + da_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_categories)
        )

    def encode_utterance(self, utterance_ids: torch.Tensor, utterance_mask: torch.Tensor):
        """
        Encode a single utterance using word-level GRU.

        Args:
            utterance_ids: (batch, seq_len) word token ids
            utterance_mask: (batch, seq_len) attention mask

        Returns:
            utterance_vector: (batch, utterance_hidden) final hidden state
        """
        # Embed words
        embeddings = self.word_embedding(utterance_ids)  # (B, L, E)
        embeddings = self.dropout(embeddings)

        # Get sequence lengths for packing
        if utterance_mask is not None:
            lengths = utterance_mask.sum(dim=1).cpu()
            lengths = lengths.clamp(min=1)  # Ensure at least 1
        else:
            lengths = torch.full((utterance_ids.size(0),), utterance_ids.size(1))

        # Pack, process, unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.utterance_encoder(packed)

        # Return final hidden state
        return hidden.squeeze(0)  # (B, H)

    def forward(self, input_ids=None, attention_mask=None,
                hierarchical_input_ids=None, hierarchical_attention_mask=None,
                speaker_ids=None, main_cat_ids=None, **kwargs):
        """
        Forward pass implementing Tanaka et al. architecture.

        Args:
            hierarchical_input_ids: (batch, num_utterances, seq_len) word tokens
            hierarchical_attention_mask: (batch, num_utterances, seq_len) masks
            speaker_ids: (batch, num_utterances) speaker identifiers per utterance
            main_cat_ids: (batch, num_utterances) DA labels for history

        Returns:
            logits: (batch, num_categories) prediction logits
        """
        # Handle hierarchical input
        if hierarchical_input_ids is not None:
            batch_size, num_utterances, seq_len = hierarchical_input_ids.shape
            device = hierarchical_input_ids.device

            # === Utterance Encoder ===
            # Encode each utterance independently
            utterance_vectors = []
            for i in range(num_utterances):
                utt_vec = self.encode_utterance(
                    hierarchical_input_ids[:, i],
                    hierarchical_attention_mask[:, i] if hierarchical_attention_mask is not None else None
                )
                utterance_vectors.append(utt_vec)

            # Stack: (batch, num_utterances, utterance_hidden)
            utterance_sequence = torch.stack(utterance_vectors, dim=1)

            # === Compute Speaker Change Tags ===
            if speaker_ids is not None:
                # Binary flag: 1 if speaker changed from previous utterance
                speaker_change = torch.zeros(batch_size, num_utterances, 1, device=device)
                if num_utterances > 1:
                    changes = (speaker_ids[:, 1:] != speaker_ids[:, :-1]).float().unsqueeze(-1)
                    speaker_change[:, 1:, :] = changes
            else:
                # No speaker info - use zeros
                speaker_change = torch.zeros(batch_size, num_utterances, 1, device=device)

            # === Context Encoder ===
            # Concatenate utterance vectors with speaker change tags
            context_input = torch.cat([utterance_sequence, speaker_change], dim=-1)

            # Process through context GRU
            _, context_hidden = self.context_encoder(context_input)
            context_repr = context_hidden.squeeze(0)  # (B, context_hidden)

            # === DA Encoder ===
            if main_cat_ids is not None:
                # Use DA history (all but last, since last is what we predict)
                da_history = main_cat_ids[:, :-1] if num_utterances > 1 else main_cat_ids

                # Clamp to valid range (in case of invalid indices)
                da_history = da_history.clamp(0, self.num_categories)

                # Embed and encode DA sequence
                da_embeds = self.da_embedding(da_history)  # (B, H-1, D)
                da_embeds = self.dropout(da_embeds)

                _, da_hidden = self.da_encoder(da_embeds)
                da_repr = da_hidden.squeeze(0)  # (B, da_hidden)
            else:
                # No DA history - use zeros
                da_repr = torch.zeros(batch_size, self.da_hidden, device=device)

            # === Classifier ===
            combined = torch.cat([context_repr, da_repr], dim=-1)
            logits = self.classifier(combined)

            return logits
        else:
            # Fallback for flat input (single concatenated sequence)
            # Process as single utterance
            embeddings = self.word_embedding(input_ids)
            embeddings = self.dropout(embeddings)

            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
            else:
                lengths = torch.full((input_ids.size(0),), input_ids.size(1))

            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.utterance_encoder(packed)
            context_repr = hidden.squeeze(0)

            # No DA history in flat mode
            da_repr = torch.zeros(input_ids.size(0), self.da_hidden, device=input_ids.device)

            combined = torch.cat([context_repr, da_repr], dim=-1)
            logits = self.classifier(combined)

            return logits


class TFIDFRNNPredictor(nn.Module):
    """
    RNN baseline using TF-IDF features instead of learned embeddings.
    Good for testing if learned representations matter.
    """
    def __init__(self, num_categories: int, tfidf_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.input_projection = nn.Linear(tfidf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_categories)
        )
    
    def forward(self, tfidf_features: torch.Tensor, **kwargs):
        # Project TF-IDF features to hidden dimension
        projected = self.input_projection(tfidf_features)
        projected = self.dropout(projected)
        
        # Add sequence dimension if needed (for single utterance)
        if len(projected.shape) == 2:
            projected = projected.unsqueeze(1)
        
        # RNN processing
        output, _ = self.rnn(projected)
        
        # Use last output
        final_output = output[:, -1]
        
        # Classification
        logits = self.classifier(final_output)
        return logits


class MinimalRNNBaseline(nn.Module):
    """
    Absolutely minimal RNN baseline - just for comparison.
    """
    def __init__(self, num_categories: int, vocab_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_categories)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        embeddings = self.embedding(input_ids)
        _, (hidden, _) = self.rnn(embeddings)
        logits = self.classifier(hidden[-1])  # Use last layer's hidden state
        return logits


class TransitionMatrixBaseline(nn.Module):
    """
    Non-trainable baseline that uses the transition matrix for predictions.

    This is a pure probabilistic baseline for evaluation purposes only - it has NO trainable parameters.
    It simply looks up the historical transition probabilities from the training data and uses them
    as predictions. This provides a simple baseline to compare learned models against.

    Modes:
    - use_sampling=False (default): Returns transition probabilities as logits for evaluation
    - use_sampling=True: Stochastically samples from the transition distribution
      (so if a transition is 90% probable, it will be sampled 90% of the time)
    """
    def __init__(self, num_categories: int, transition_matrix: pd.DataFrame, label_encoder, epsilon: float = 1e-8, use_sampling: bool = False):
        super().__init__()
        self.num_categories = num_categories
        self.epsilon = epsilon
        self.use_sampling = use_sampling  # Whether to sample during inference

        # Ensure no parameters are trainable
        self.requires_grad_(False)

        # Helper to extract code-only identifier
        def _code(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            parts = str(s).split('|', 1)
            return parts[0].strip()

        # Build code-level transition distributions
        self.target_class_codes: List[str] = [_code(c) for c in label_encoder.classes_]

        # Aggregate transition matrix by code
        col_codes: Dict[str, List[str]] = {}
        for col in transition_matrix.columns:
            c = _code(col)
            if c:
                col_codes.setdefault(c, []).append(col)

        rows_by_code: Dict[str, List[str]] = {}
        for row in transition_matrix.index:
            rc = _code(row)
            if rc:
                rows_by_code.setdefault(rc, []).append(row)

        # Build source_code -> distribution over target class indices
        distrib_by_source: Dict[str, torch.Tensor] = {}
        for src_code, row_names in rows_by_code.items():
            # Average rows if multiple
            row_values = None
            for rn in row_names:
                series = transition_matrix.loc[rn]
                row_values = series.values if row_values is None else (row_values + series.values)
            if row_values is None:
                continue
            row_values = row_values / max(1, len(row_names))

            # Convert to dict col_name -> prob
            row_series = pd.Series(row_values, index=transition_matrix.columns)

            # Aggregate by code for columns
            code_prob: Dict[str, float] = {}
            for code, cols in col_codes.items():
                val = float(row_series[cols].sum()) if len(cols) > 1 else float(row_series[cols[0]])
                code_prob[code] = max(val, 0.0)

            # Map to target class indices
            vec = torch.zeros(len(self.target_class_codes), dtype=torch.float32)
            for j, tgt_code in enumerate(self.target_class_codes):
                p = code_prob.get(tgt_code, 0.0)
                vec[j] = p

            # Add epsilon and normalize
            vec = vec + epsilon
            if vec.sum() > 0:
                vec = vec / vec.sum()

            distrib_by_source[src_code] = vec

        # Store as dict of tensors (will be moved to device when needed)
        self.distrib_by_source = distrib_by_source

        # Uniform distribution as fallback
        uniform = torch.ones(num_categories, dtype=torch.float32) / num_categories
        self.register_buffer('uniform_dist', uniform)

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
                source_categories: Optional[List[str]] = None, **kwargs):
        """
        Returns logits based on transition matrix probabilities.

        During training (self.training=True): Returns log probabilities as logits.
        During inference with use_sampling=True: Samples from the distribution and returns one-hot encoded logits.
        During inference with use_sampling=False: Returns log probabilities as logits (same as training).

        Args:
            input_ids: Used only to determine batch size and device (for compatibility)
            attention_mask: Ignored (for compatibility with other models)
            source_categories: List of source/previous category strings for each sample in batch
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            Logits tensor of shape [batch_size, num_categories]
        """
        # Determine batch size and device
        if source_categories is not None:
            batch_size = len(source_categories)
        elif input_ids is not None:
            batch_size = input_ids.size(0)
        else:
            batch_size = 1

        device = input_ids.device if input_ids is not None else self.uniform_dist.device

        # First, build the probability distributions
        probs = torch.zeros(batch_size, self.num_categories, device=device)

        if source_categories is None:
            # No context - use uniform distribution
            for i in range(batch_size):
                probs[i] = self.uniform_dist.to(device)
        else:
            for i, prev_cat in enumerate(source_categories):
                if prev_cat is None or prev_cat == "":
                    # No previous category - use uniform
                    probs[i] = self.uniform_dist.to(device)
                else:
                    # Extract code from previous category
                    prev_code = str(prev_cat).split('|', 1)[0].strip()

                    # Get distribution for this source code
                    if prev_code in self.distrib_by_source:
                        dist = self.distrib_by_source[prev_code].to(device)
                        probs[i] = dist
                    else:
                        # Unknown previous category - use uniform
                        probs[i] = self.uniform_dist.to(device)

        # During training or when not sampling, return log probabilities as logits
        if self.training or not self.use_sampling:
            logits = torch.log(probs + self.epsilon)
            return logits

        # During inference with sampling enabled: sample from the distribution
        # This makes the model stochastic - it will sample according to the probabilities
        else:
            # Sample from categorical distribution
            sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]

            # Create one-hot encoded output (or you could return the probabilities)
            # For compatibility with evaluation, we'll return very high logits for sampled class
            sampled_logits = torch.full((batch_size, self.num_categories), -1000.0, device=device)
            sampled_logits[torch.arange(batch_size, device=device), sampled_indices] = 100.0

            return sampled_logits


class HierarchyAwarePredictor(nn.Module):
    def __init__(self, num_categories, hierarchy, pretrained_model="EuroBERT/EuroBERT-610m"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size
        
        # Store hierarchy info
        self.hierarchy = hierarchy
        
        # Multi-head attention for utterance encoding (replaces simple attention)
        self.utterance_multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-utterance attention (replaces/augments GRU)
        self.cross_utterance_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for attention blocks
        self.utterance_norm = nn.LayerNorm(hidden_size)
        self.cross_utterance_norm = nn.LayerNorm(hidden_size)
        
        # Positional encoding for utterance sequences
        self.max_seq_length = 512
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=self.max_seq_length)
        
        # Keep GRU as an option (can be used in combination with attention)
        self.use_gru = False
        if self.use_gru:
            self.conversation_gru = nn.GRU(
                hidden_size, hidden_size, bidirectional=True, 
                batch_first=True, num_layers=2, dropout=0.2
            )
            # Project bidirectional GRU output to hidden_size for attention
            self.gru_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Component embedding layers
        embed_dim = 64
        self.speaker_embedding = nn.Embedding(hierarchy["num_speakers"], embed_dim)
        self.main_cat_embedding = nn.Embedding(hierarchy["num_main_categories"], embed_dim)
        self.sub_cat_embedding = nn.Embedding(hierarchy["num_sub_categories"], embed_dim)
        self.third_level_embedding = nn.Embedding(hierarchy["num_third_level"], embed_dim)
        self.fourth_level_embedding = nn.Embedding(hierarchy["num_fourth_level"], embed_dim)
        
        # Multi-head attention for component interactions
        self.component_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.component_norm = nn.LayerNorm(embed_dim)
        
        # Context-component cross attention
        self.context_component_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Projection layers
        component_output_dim = embed_dim * 5  # 5 components
        if self.use_gru:
            conv_dim = hidden_size * 2  # bidirectional GRU
        else:
            conv_dim = hidden_size
            
        self.combined_projection = nn.Linear(
            conv_dim + component_output_dim, hidden_size
        )
        
        # Component projection for cross-attention
        self.component_projection = nn.Linear(embed_dim, hidden_size)
        
        # Prediction heads
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.speaker_head = nn.Linear(hidden_size, hierarchy["num_speakers"])
        self.main_cat_head = nn.Linear(hidden_size, hierarchy["num_main_categories"])
        self.sub_cat_head = nn.Linear(hidden_size, hierarchy["num_sub_categories"])
        self.third_level_head = nn.Linear(hidden_size, hierarchy["num_third_level"])
        self.fourth_level_head = nn.Linear(hidden_size, hierarchy["num_fourth_level"])
        
        # Hierarchical classification gate
        self.hierarchy_gate = nn.Linear(hidden_size, 1)
        
        # Optional category transitions
        self.use_transitions = False
        if self.use_transitions:
            self.category_transitions = nn.Parameter(
                torch.zeros(num_categories, num_categories)
            )
    
    def encode_utterance_with_attention(self, sequence_output, attention_mask):
        """Use multi-head self-attention for utterance encoding instead of simple attention pooling"""
        # sequence_output: [batch_size, seq_len, hidden_size]
        
        # Self-attention over tokens within utterance
        attn_output, attn_weights = self.utterance_multihead_attn(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()  # Invert mask for PyTorch convention
        )
        
        # Residual connection + layer norm
        attn_output = self.utterance_norm(sequence_output + attn_output)
        
        # Global average pooling (can also use CLS token or learned pooling)
        # Mask out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(attn_output)
        masked_output = attn_output * mask_expanded
        utterance_vector = masked_output.sum(dim=1) / mask_expanded.sum(dim=1)
        
        return utterance_vector
    
    def process_component_interactions(self, components):
        """Use multi-head attention to model component interactions"""
        # components: [batch_size, num_components, embed_dim]
        
        # Self-attention among components
        attn_output, _ = self.component_attn(components, components, components)
        
        # Residual connection + layer norm
        enhanced_components = self.component_norm(components + attn_output)
        
        return enhanced_components
    
    def cross_attention_context_components(self, context_vector, component_embeddings):
        """Use cross-attention between conversation context and hierarchical components"""
        # context_vector: [batch_size, hidden_size]
        # component_embeddings: [batch_size, num_components, embed_dim]
        
        # Project components to match context dimension
        projected_components = self.component_projection(component_embeddings)
        
        # Expand context for attention
        context_expanded = context_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Cross-attention: context attends to components
        attended_context, attn_weights = self.context_component_attn(
            context_expanded, projected_components, projected_components
        )
        
        return attended_context.squeeze(1), attn_weights
    
    def forward(self, input_ids=None, attention_mask=None, hierarchical_input_ids=None, 
                hierarchical_attention_mask=None, speaker_ids=None, main_cat_ids=None, 
                sub_cat_ids=None, third_level_ids=None, fourth_level_ids=None):
        
        # Detect input mode (flat vs hierarchical)
        is_hierarchical = hierarchical_input_ids is not None
        
        if not is_hierarchical:
            # Process flat input with multi-head attention
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            
            # Use multi-head attention for utterance encoding
            context_vector = self.encode_utterance_with_attention(sequence_output, attention_mask)
            
            # Simple category prediction for flat input
            category_logits = self.category_head(context_vector)
            return category_logits
        
        # Process hierarchical input
        batch_size = hierarchical_input_ids.size(0)
        num_utterances = hierarchical_input_ids.size(1)
        
        # Process each utterance with attention-based encoding
        utterance_vectors = []
        
        for i in range(num_utterances):
            utterance_input_ids = hierarchical_input_ids[:, i]
            utterance_attention_mask = hierarchical_attention_mask[:, i]
            
            # Skip empty utterances
            if (utterance_input_ids == 0).all():
                utterance_vectors.append(torch.zeros(batch_size, self.bert.config.hidden_size, 
                                                   device=hierarchical_input_ids.device))
                continue
            
            # Process through BERT
            outputs = self.bert(
                input_ids=utterance_input_ids,
                attention_mask=utterance_attention_mask
            )
            
            # Use multi-head attention for utterance encoding
            utterance_vector = self.encode_utterance_with_attention(
                outputs.last_hidden_state, utterance_attention_mask
            )
            utterance_vectors.append(utterance_vector)
        
        # Stack utterance vectors
        utterance_sequence = torch.stack(utterance_vectors, dim=1)
        
        # Add positional encoding to utterance sequence
        utterance_sequence = self.pos_encoding(utterance_sequence)
        
        # Cross-utterance attention (lets utterances attend to each other)
        attn_output, cross_attn_weights = self.cross_utterance_attn(
            utterance_sequence, utterance_sequence, utterance_sequence
        )
        
        # Residual connection + layer norm
        enhanced_utterances = self.cross_utterance_norm(utterance_sequence + attn_output)
        
        # Optional: Also use GRU for temporal modeling
        if self.use_gru:
            conv_output, _ = self.conversation_gru(enhanced_utterances)
            conv_state = conv_output[:, -1]  # Last state [batch_size, hidden_size*2]
            # Project to hidden_size for attention compatibility
            conv_state_projected = self.gru_projection(conv_state)
        else:
            # Use the last utterance representation
            conv_state = enhanced_utterances[:, -1]
            conv_state_projected = conv_state
        
        # Get component embeddings from the last utterance
        speaker_embed = self.speaker_embedding(speaker_ids[:, -1])
        main_cat_embed = self.main_cat_embedding(main_cat_ids[:, -1])
        sub_cat_embed = self.sub_cat_embedding(sub_cat_ids[:, -1])
        third_level_embed = self.third_level_embedding(third_level_ids[:, -1])
        fourth_level_embed = self.fourth_level_embedding(fourth_level_ids[:, -1])
        
        # Stack components for attention
        component_embeddings = torch.stack([
            speaker_embed, main_cat_embed, sub_cat_embed, 
            third_level_embed, fourth_level_embed
        ], dim=1)  # [batch_size, 5, embed_dim]
        
        # Process component interactions with attention
        enhanced_components = self.process_component_interactions(component_embeddings)
        
        # Cross-attention between context and components
        attended_context, component_attn_weights = self.cross_attention_context_components(
            conv_state_projected, enhanced_components
        )
        
        # Flatten enhanced components for combination
        flattened_components = enhanced_components.view(batch_size, -1)
        
        # Combine context and components (use original conv_state for full features)
        if self.use_gru:
            combined_features = torch.cat([conv_state, flattened_components], dim=1)
        else:
            combined_features = torch.cat([attended_context, flattened_components], dim=1)
        context_vector = self.combined_projection(combined_features)
        
        # Hierarchical gate
        gate_value = torch.sigmoid(self.hierarchy_gate(context_vector))
        
        # Predictions
        direct_logits = self.category_head(context_vector)
        speaker_logits = self.speaker_head(context_vector)
        main_cat_logits = self.main_cat_head(context_vector)
        sub_cat_logits = self.sub_cat_head(context_vector)
        third_level_logits = self.third_level_head(context_vector)
        fourth_level_logits = self.fourth_level_head(context_vector)
        
        # Apply transition scores if enabled
        if self.use_transitions:
            last_category_idx = self.hierarchy["speaker_types"].get(speaker_ids[:, -1].cpu().numpy()[0], 0)
            transition_scores = self.category_transitions[last_category_idx]
            direct_logits = direct_logits + transition_scores
        
        outputs = {
            "category_logits": direct_logits,
            "speaker_logits": speaker_logits,
            "main_cat_logits": main_cat_logits, 
            "sub_cat_logits": sub_cat_logits,
            "third_level_logits": third_level_logits,
            "fourth_level_logits": fourth_level_logits,
            "gate": gate_value,
            # Attention weights for interpretability
            "cross_utterance_attention": cross_attn_weights,
            "component_attention": component_attn_weights
        }
        
        return direct_logits


class PositionalEncoding(nn.Module):
    """Positional encoding for utterance sequences"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x

class HierarchyAwarePredictorWithCategoryHistory(nn.Module):
    """
    Advanced Hierarchy-Aware Predictor with Category History Modeling.
    
    This model enhances the base HierarchyAwarePredictor by:
    1. Modeling the sequence of previous categories with positional encoding
    2. Cross-attention between text and category histories
    3. Component-level category history tracking for hierarchical consistency
    4. Enhanced dialogue flow understanding through temporal patterns
    """
    
    def __init__(self, num_categories, hierarchy, pretrained_model="EuroBERT/EuroBERT-610m", 
                 max_history_length=10, use_component_history=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size
        
        # Store hierarchy info and configuration
        self.hierarchy = hierarchy
        self.max_history_length = max_history_length
        self.use_component_history = use_component_history
        
        # Multi-head attention for utterance encoding
        self.utterance_multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-utterance attention for conversation flow
        self.cross_utterance_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for attention blocks
        self.utterance_norm = nn.LayerNorm(hidden_size)
        self.cross_utterance_norm = nn.LayerNorm(hidden_size)
        
        # Positional encoding for utterance sequences
        self.max_seq_length = 512
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=self.max_seq_length)
        
        # NEW: Category history modeling
        self.category_embedding = nn.Embedding(num_categories + 1, hidden_size)  # +1 for padding
        self.category_positional_encoding = PositionalEncoding(hidden_size, max_len=max_history_length)
        
        # NEW: Multi-head attention for category sequence modeling
        self.category_sequence_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.category_norm = nn.LayerNorm(hidden_size)
        
        # NEW: Cross-attention between text and category histories
        self.text_category_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.text_category_norm = nn.LayerNorm(hidden_size)
        
        # Component embedding layers
        embed_dim = 64
        self.speaker_embedding = nn.Embedding(hierarchy["num_speakers"], embed_dim)
        self.main_cat_embedding = nn.Embedding(hierarchy["num_main_categories"], embed_dim)
        self.sub_cat_embedding = nn.Embedding(hierarchy["num_sub_categories"], embed_dim)
        self.third_level_embedding = nn.Embedding(hierarchy["num_third_level"], embed_dim)
        self.fourth_level_embedding = nn.Embedding(hierarchy["num_fourth_level"], embed_dim)
        
        # NEW: Component-level category history embeddings
        if self.use_component_history:
            self.speaker_history_embedding = nn.Embedding(hierarchy["num_speakers"] + 1, embed_dim)
            self.main_cat_history_embedding = nn.Embedding(hierarchy["num_main_categories"] + 1, embed_dim)
            self.sub_cat_history_embedding = nn.Embedding(hierarchy["num_sub_categories"] + 1, embed_dim)
            self.third_level_history_embedding = nn.Embedding(hierarchy["num_third_level"] + 1, embed_dim)
            self.fourth_level_history_embedding = nn.Embedding(hierarchy["num_fourth_level"] + 1, embed_dim)
            
            # Attention for component history sequences
            self.component_history_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.component_history_norm = nn.LayerNorm(embed_dim)
        
        # Multi-head attention for component interactions
        self.component_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.component_norm = nn.LayerNorm(embed_dim)
        
        # Context-component cross attention
        self.context_component_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # NEW: Category transition modeling
        self.use_transitions = True
        if self.use_transitions:
            self.category_transition_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.transition_norm = nn.LayerNorm(hidden_size)
        
        # Projection layers
        component_output_dim = embed_dim * 5  # 5 components
        conv_dim = hidden_size
        category_history_dim = hidden_size  # From category history context
        
        self.combined_projection = nn.Linear(
            conv_dim + component_output_dim + category_history_dim, hidden_size
        )
        
        # Component projection for cross-attention
        self.component_projection = nn.Linear(embed_dim, hidden_size)
        
        # Prediction heads
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.speaker_head = nn.Linear(hidden_size, hierarchy["num_speakers"])
        self.main_cat_head = nn.Linear(hidden_size, hierarchy["num_main_categories"])
        self.sub_cat_head = nn.Linear(hidden_size, hierarchy["num_sub_categories"])
        self.third_level_head = nn.Linear(hidden_size, hierarchy["num_third_level"])
        self.fourth_level_head = nn.Linear(hidden_size, hierarchy["num_fourth_level"])
        
        # Hierarchical classification gate
        self.hierarchy_gate = nn.Linear(hidden_size, 1)
        
        # NEW: Category confidence predictor (predicts how confident the model is about next category)
        self.confidence_head = nn.Linear(hidden_size, 1)
    
    def encode_utterance_with_attention(self, sequence_output, attention_mask):
        """Use multi-head self-attention for utterance encoding"""
        # Self-attention over tokens within utterance
        attn_output, attn_weights = self.utterance_multihead_attn(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Residual connection + layer norm
        attn_output = self.utterance_norm(sequence_output + attn_output)
        
        # Global average pooling with masking
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(attn_output)
        masked_output = attn_output * mask_expanded
        utterance_vector = masked_output.sum(dim=1) / mask_expanded.sum(dim=1)
        
        return utterance_vector
    
    def encode_category_history(self, category_history_ids, component_history_ids=None):
        """
        Encode the sequence of previous categories with positional information
        
        Args:
            category_history_ids: [batch_size, history_len] - sequence of category IDs
            component_history_ids: dict with keys like 'speaker', 'main_cat', etc.
        """
        batch_size, history_len = category_history_ids.shape
        
        # Embed category sequence
        category_embeds = self.category_embedding(category_history_ids)
        
        # Add positional encoding
        category_embeds = self.category_positional_encoding(category_embeds)
        
        # Self-attention over category sequence
        attn_output, category_attn_weights = self.category_sequence_attn(
            category_embeds, category_embeds, category_embeds
        )
        
        # Residual connection + layer norm
        enhanced_categories = self.category_norm(category_embeds + attn_output)
        
        # Optional: Process component-level histories
        component_context = None
        if self.use_component_history and component_history_ids:
            component_histories = []
            
            for component_name, component_ids in component_history_ids.items():
                if component_name == 'speaker':
                    comp_embeds = self.speaker_history_embedding(component_ids)
                elif component_name == 'main_cat':
                    comp_embeds = self.main_cat_history_embedding(component_ids)
                elif component_name == 'sub_cat':
                    comp_embeds = self.sub_cat_history_embedding(component_ids)
                elif component_name == 'third_level':
                    comp_embeds = self.third_level_history_embedding(component_ids)
                elif component_name == 'fourth_level':
                    comp_embeds = self.fourth_level_history_embedding(component_ids)
                else:
                    continue
                
                component_histories.append(comp_embeds)
            
            if component_histories:
                # Stack and process component histories
                stacked_components = torch.stack(component_histories, dim=2)  # [batch, history, components, embed]
                batch_size, hist_len, num_components, embed_dim = stacked_components.shape
                
                # Reshape for attention processing
                stacked_components = stacked_components.view(batch_size * hist_len, num_components, embed_dim)
                
                # Apply attention across components for each time step
                comp_attn_out, _ = self.component_history_attn(
                    stacked_components, stacked_components, stacked_components
                )
                comp_enhanced = self.component_history_norm(stacked_components + comp_attn_out)
                
                # Reshape back and pool
                comp_enhanced = comp_enhanced.view(batch_size, hist_len, num_components, embed_dim)
                component_context = comp_enhanced.mean(dim=2)  # Pool across components
        
        return enhanced_categories, category_attn_weights, component_context
    
    def process_component_interactions(self, components):
        """Use multi-head attention to model component interactions"""
        # Self-attention among components
        attn_output, _ = self.component_attn(components, components, components)
        
        # Residual connection + layer norm
        enhanced_components = self.component_norm(components + attn_output)
        
        return enhanced_components
    
    def cross_attention_context_components(self, context_vector, component_embeddings):
        """Use cross-attention between conversation context and hierarchical components"""
        # Project components to match context dimension
        projected_components = self.component_projection(component_embeddings)
        
        # Expand context for attention
        context_expanded = context_vector.unsqueeze(1)
        
        # Cross-attention: context attends to components
        attended_context, attn_weights = self.context_component_attn(
            context_expanded, projected_components, projected_components
        )
        
        return attended_context.squeeze(1), attn_weights
    
    def forward(self, input_ids=None, attention_mask=None, hierarchical_input_ids=None, 
                hierarchical_attention_mask=None, speaker_ids=None, main_cat_ids=None, 
                sub_cat_ids=None, third_level_ids=None, fourth_level_ids=None,
                # NEW PARAMETERS:
                category_history_ids=None, component_history_ids=None):
        
        # Detect input mode (flat vs hierarchical)
        is_hierarchical = hierarchical_input_ids is not None
        
        if not is_hierarchical:
            # Process flat input with multi-head attention
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            
            # Use multi-head attention for utterance encoding
            context_vector = self.encode_utterance_with_attention(sequence_output, attention_mask)
            
            # If we have category history for flat input, use it
            category_context = None
            if category_history_ids is not None:
                enhanced_category_history, _, _ = self.encode_category_history(
                    category_history_ids, component_history_ids
                )
                category_context = enhanced_category_history[:, -1]  # Use last category representation
                
                # Combine with text context
                combined_features = torch.cat([context_vector, category_context], dim=1)
                context_vector = self.combined_projection(combined_features)
            
            # Simple category prediction for flat input
            category_logits = self.category_head(context_vector)
            return category_logits
        
        # Process hierarchical input
        batch_size = hierarchical_input_ids.size(0)
        num_utterances = hierarchical_input_ids.size(1)
        
        # Process each utterance with attention-based encoding
        utterance_vectors = []
        
        for i in range(num_utterances):
            utterance_input_ids = hierarchical_input_ids[:, i]
            utterance_attention_mask = hierarchical_attention_mask[:, i]
            
            # Skip empty utterances
            if (utterance_input_ids == 0).all():
                utterance_vectors.append(torch.zeros(batch_size, self.bert.config.hidden_size, 
                                                   device=hierarchical_input_ids.device))
                continue
            
            # Process through BERT
            outputs = self.bert(
                input_ids=utterance_input_ids,
                attention_mask=utterance_attention_mask
            )
            
            # Use multi-head attention for utterance encoding
            utterance_vector = self.encode_utterance_with_attention(
                outputs.last_hidden_state, utterance_attention_mask
            )
            utterance_vectors.append(utterance_vector)
        
        # Stack utterance vectors
        utterance_sequence = torch.stack(utterance_vectors, dim=1)
        
        # Add positional encoding to utterance sequence
        utterance_sequence = self.pos_encoding(utterance_sequence)
        
        # Cross-utterance attention (lets utterances attend to each other)
        attn_output, cross_attn_weights = self.cross_utterance_attn(
            utterance_sequence, utterance_sequence, utterance_sequence
        )
        
        # Residual connection + layer norm
        enhanced_utterances = self.cross_utterance_norm(utterance_sequence + attn_output)
        
        # Use the last utterance representation
        conv_state = enhanced_utterances[:, -1]
        
        # NEW: Process category history if available
        category_context = None
        text_category_attn_weights = None
        if category_history_ids is not None:
            enhanced_category_history, category_attn_weights, component_context = self.encode_category_history(
                category_history_ids, component_history_ids
            )
            
            # Cross-attention between text utterances and category history
            text_attended, text_category_attn_weights = self.text_category_cross_attn(
                enhanced_utterances, enhanced_category_history, enhanced_category_history
            )
            
            # Residual connection for text-category integration
            enhanced_utterances = self.text_category_norm(enhanced_utterances + text_attended)
            conv_state = enhanced_utterances[:, -1]  # Update conversation state
            
            # Use category history context
            category_context = enhanced_category_history[:, -1]  # Last category representation
            
            # Optional: Model category transitions
            if self.use_transitions and enhanced_category_history.size(1) > 1:
                # Attention over category transitions
                transition_attn, _ = self.category_transition_attn(
                    enhanced_category_history[:, -1:],  # Current as query
                    enhanced_category_history[:, :-1],  # Previous as key/value
                    enhanced_category_history[:, :-1]
                )
                category_context = self.transition_norm(category_context + transition_attn.squeeze(1))
        
        # Get component embeddings from the last utterance
        speaker_embed = self.speaker_embedding(speaker_ids[:, -1])
        main_cat_embed = self.main_cat_embedding(main_cat_ids[:, -1])
        sub_cat_embed = self.sub_cat_embedding(sub_cat_ids[:, -1])
        third_level_embed = self.third_level_embedding(third_level_ids[:, -1])
        fourth_level_embed = self.fourth_level_embedding(fourth_level_ids[:, -1])
        
        # Stack components for attention
        component_embeddings = torch.stack([
            speaker_embed, main_cat_embed, sub_cat_embed, 
            third_level_embed, fourth_level_embed
        ], dim=1)
        
        # Process component interactions with attention
        enhanced_components = self.process_component_interactions(component_embeddings)
        
        # Cross-attention between context and components
        attended_context, component_attn_weights = self.cross_attention_context_components(
            conv_state, enhanced_components
        )
        
        # Flatten enhanced components for combination
        flattened_components = enhanced_components.view(batch_size, -1)
        
        # Combine context, components, and category history
        if category_context is not None:
            combined_features = torch.cat([attended_context, flattened_components, category_context], dim=1)
        else:
            combined_features = torch.cat([attended_context, flattened_components], dim=1)
            # Pad with zeros for category context
            zero_category_context = torch.zeros(batch_size, self.bert.config.hidden_size, device=attended_context.device)
            combined_features = torch.cat([combined_features, zero_category_context], dim=1)
        
        context_vector = self.combined_projection(combined_features)
        
        # Hierarchical gate
        gate_value = torch.sigmoid(self.hierarchy_gate(context_vector))
        
        # Predictions
        direct_logits = self.category_head(context_vector)
        speaker_logits = self.speaker_head(context_vector)
        main_cat_logits = self.main_cat_head(context_vector)
        sub_cat_logits = self.sub_cat_head(context_vector)
        third_level_logits = self.third_level_head(context_vector)
        fourth_level_logits = self.fourth_level_head(context_vector)
        
        # NEW: Confidence prediction
        confidence_score = torch.sigmoid(self.confidence_head(context_vector))
        
        outputs = {
            "category_logits": direct_logits,
            "speaker_logits": speaker_logits,
            "main_cat_logits": main_cat_logits, 
            "sub_cat_logits": sub_cat_logits,
            "third_level_logits": third_level_logits,
            "fourth_level_logits": fourth_level_logits,
            "gate": gate_value,
            "confidence": confidence_score,
            # Attention weights for interpretability
            "cross_utterance_attention": cross_attn_weights,
            "component_attention": component_attn_weights,
            "category_attention": category_attn_weights if category_history_ids is not None else None,
            "text_category_attention": text_category_attn_weights
        }

        return direct_logits


class CategoryHistoryPredictor(nn.Module):
    """
    BERT-based Predictor with Category History Modeling (without hierarchy embeddings).

    This model uses:
    1. BERT encoder with multi-head attention for text encoding
    2. Category history modeling with positional encoding and attention
    3. Cross-attention between text and category histories

    Unlike HierarchyAwarePredictorWithCategoryHistory, this model does NOT use
    hierarchy component embeddings (speaker, main_cat, sub_cat, etc.), making it
    more generalizable to other label systems that don't have hierarchical structure.
    """

    def __init__(self, num_categories, pretrained_model="EuroBERT/EuroBERT-610m",
                 max_history_length=10):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size

        # Store configuration
        self.max_history_length = max_history_length
        self.num_categories = num_categories

        # Multi-head attention for utterance encoding
        self.utterance_multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Cross-utterance attention for conversation flow
        self.cross_utterance_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Layer normalization for attention blocks
        self.utterance_norm = nn.LayerNorm(hidden_size)
        self.cross_utterance_norm = nn.LayerNorm(hidden_size)

        # Positional encoding for utterance sequences
        self.max_seq_length = 512
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=self.max_seq_length)

        # Category history modeling
        self.category_embedding = nn.Embedding(num_categories + 1, hidden_size)  # +1 for padding
        self.category_positional_encoding = PositionalEncoding(hidden_size, max_len=max_history_length)

        # Multi-head attention for category sequence modeling
        self.category_sequence_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.category_norm = nn.LayerNorm(hidden_size)

        # Cross-attention between text and category histories
        self.text_category_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.text_category_norm = nn.LayerNorm(hidden_size)

        # Category transition modeling
        self.category_transition_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.transition_norm = nn.LayerNorm(hidden_size)

        # Projection layers
        # Combined features: conv_state (hidden_size) + category_context (hidden_size)
        self.combined_projection = nn.Linear(hidden_size * 2, hidden_size)

        # Prediction head
        self.category_head = nn.Linear(hidden_size, num_categories)

        # Confidence predictor
        self.confidence_head = nn.Linear(hidden_size, 1)

    def encode_utterance_with_attention(self, sequence_output, attention_mask):
        """Use multi-head self-attention for utterance encoding"""
        # Self-attention over tokens within utterance
        attn_output, attn_weights = self.utterance_multihead_attn(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )

        # Residual connection + layer norm
        attn_output = self.utterance_norm(sequence_output + attn_output)

        # Global average pooling with masking
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(attn_output)
        masked_output = attn_output * mask_expanded
        utterance_vector = masked_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        return utterance_vector

    def encode_category_history(self, category_history_ids):
        """
        Encode the sequence of previous categories with positional information

        Args:
            category_history_ids: [batch_size, history_len] - sequence of category IDs
        """
        batch_size, history_len = category_history_ids.shape

        # Embed category sequence
        category_embeds = self.category_embedding(category_history_ids)

        # Add positional encoding
        category_embeds = self.category_positional_encoding(category_embeds)

        # Self-attention over category sequence
        attn_output, category_attn_weights = self.category_sequence_attn(
            category_embeds, category_embeds, category_embeds
        )

        # Residual connection + layer norm
        enhanced_categories = self.category_norm(category_embeds + attn_output)

        return enhanced_categories, category_attn_weights

    def forward(self, input_ids=None, attention_mask=None, hierarchical_input_ids=None,
                hierarchical_attention_mask=None, category_history_ids=None,
                # Accept but ignore hierarchy-related inputs for compatibility
                speaker_ids=None, main_cat_ids=None, sub_cat_ids=None,
                third_level_ids=None, fourth_level_ids=None, component_history_ids=None):

        # Detect input mode (flat vs hierarchical)
        is_hierarchical = hierarchical_input_ids is not None

        if not is_hierarchical:
            # Process flat input with multi-head attention
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state

            # Use multi-head attention for utterance encoding
            context_vector = self.encode_utterance_with_attention(sequence_output, attention_mask)

            # If we have category history for flat input, use it
            if category_history_ids is not None:
                enhanced_category_history, _ = self.encode_category_history(category_history_ids)
                category_context = enhanced_category_history[:, -1]  # Use last category representation

                # Combine with text context
                combined_features = torch.cat([context_vector, category_context], dim=1)
                context_vector = self.combined_projection(combined_features)

            # Category prediction
            category_logits = self.category_head(context_vector)
            return category_logits

        # Process hierarchical input
        batch_size = hierarchical_input_ids.size(0)
        num_utterances = hierarchical_input_ids.size(1)

        # Process each utterance with attention-based encoding
        utterance_vectors = []

        for i in range(num_utterances):
            utterance_input_ids = hierarchical_input_ids[:, i]
            utterance_attention_mask = hierarchical_attention_mask[:, i]

            # Skip empty utterances
            if (utterance_input_ids == 0).all():
                utterance_vectors.append(torch.zeros(batch_size, self.bert.config.hidden_size,
                                                   device=hierarchical_input_ids.device))
                continue

            # Process through BERT
            outputs = self.bert(
                input_ids=utterance_input_ids,
                attention_mask=utterance_attention_mask
            )

            # Use multi-head attention for utterance encoding
            utterance_vector = self.encode_utterance_with_attention(
                outputs.last_hidden_state, utterance_attention_mask
            )
            utterance_vectors.append(utterance_vector)

        # Stack utterance vectors
        utterance_sequence = torch.stack(utterance_vectors, dim=1)

        # Add positional encoding to utterance sequence
        utterance_sequence = self.pos_encoding(utterance_sequence)

        # Cross-utterance attention (lets utterances attend to each other)
        attn_output, cross_attn_weights = self.cross_utterance_attn(
            utterance_sequence, utterance_sequence, utterance_sequence
        )

        # Residual connection + layer norm
        enhanced_utterances = self.cross_utterance_norm(utterance_sequence + attn_output)

        # Use the last utterance representation
        conv_state = enhanced_utterances[:, -1]

        # Process category history if available
        category_context = None
        if category_history_ids is not None:
            enhanced_category_history, category_attn_weights = self.encode_category_history(
                category_history_ids
            )

            # Cross-attention between text utterances and category history
            text_attended, text_category_attn_weights = self.text_category_cross_attn(
                enhanced_utterances, enhanced_category_history, enhanced_category_history
            )

            # Residual connection for text-category integration
            enhanced_utterances = self.text_category_norm(enhanced_utterances + text_attended)
            conv_state = enhanced_utterances[:, -1]  # Update conversation state

            # Use category history context
            category_context = enhanced_category_history[:, -1]  # Last category representation

            # Model category transitions if we have history
            if enhanced_category_history.size(1) > 1:
                # Attention over category transitions
                transition_attn, _ = self.category_transition_attn(
                    enhanced_category_history[:, -1:],  # Current as query
                    enhanced_category_history[:, :-1],  # Previous as key/value
                    enhanced_category_history[:, :-1]
                )
                category_context = self.transition_norm(category_context + transition_attn.squeeze(1))

        # Combine context and category history
        if category_context is not None:
            combined_features = torch.cat([conv_state, category_context], dim=1)
        else:
            # Pad with zeros for category context if not available
            zero_category_context = torch.zeros(batch_size, self.bert.config.hidden_size, device=conv_state.device)
            combined_features = torch.cat([conv_state, zero_category_context], dim=1)

        context_vector = self.combined_projection(combined_features)

        # Predictions
        category_logits = self.category_head(context_vector)

        return category_logits


class CategoryHistoryPredictorEasy(nn.Module):
    """
    Simplified BERT + Category History model.

    This model provides a lightweight alternative to CategoryHistoryPredictor by:
    1. Using simple learned attention pooling (like EuroBERTPredictor) instead of multi-head attention
    2. Using concatenation-based fusion instead of cross-attention
    3. Simple weighted pooling over category history instead of self-attention

    Architecture:
    - BERT encoder for text
    - Simple attention pooling for utterance encoding
    - Category embeddings with positional encoding
    - Learned weighted pooling over history sequence
    - Concatenation + projection for fusion
    """

    def __init__(self, num_categories, pretrained_model="EuroBERT/EuroBERT-610m",
                 max_history_length=10, history_embed_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size

        self.max_history_length = max_history_length
        self.num_categories = num_categories
        self.history_embed_dim = history_embed_dim

        # Simple attention pooling for BERT output (like EuroBERTPredictor)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Category history modeling
        # Use smaller embedding dim for history to keep it lightweight
        self.category_embedding = nn.Embedding(num_categories + 1, history_embed_dim, padding_idx=num_categories)

        # Positional encoding for category history
        self.register_buffer('history_pos_encoding', self._create_positional_encoding(max_history_length, history_embed_dim))

        # Simple attention pooling over history (learned weighted sum)
        self.history_attention = nn.Sequential(
            nn.Linear(history_embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Fusion: concatenate text and history, then project
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + history_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Classification head
        self.category_head = nn.Linear(hidden_size, num_categories)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def encode_text(self, input_ids, attention_mask):
        """Encode text using BERT + simple attention pooling."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, H)

        # Simple attention pooling
        attention_weights = self.attention(sequence_output)  # (B, L, 1)

        # Mask padding positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            attention_weights = attention_weights + (1 - mask) * -1e9

        attention_weights = F.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * sequence_output, dim=1)  # (B, H)

        return context_vector

    def encode_category_history(self, category_history_ids):
        """
        Encode category history with positional encoding and weighted pooling.

        Args:
            category_history_ids: (batch_size, history_len) - sequence of category IDs
                                  Use num_categories as padding index
        """
        batch_size, hist_len = category_history_ids.shape

        # Embed categories
        cat_embeds = self.category_embedding(category_history_ids)  # (B, H, D)

        # Add positional encoding
        cat_embeds = cat_embeds + self.history_pos_encoding[:, :hist_len, :]

        # Create padding mask
        padding_mask = (category_history_ids == self.num_categories)  # (B, H)

        # Simple attention pooling over history
        attention_weights = self.history_attention(cat_embeds)  # (B, H, 1)

        # Mask padding positions
        attention_weights = attention_weights + padding_mask.unsqueeze(-1).float() * -1e9
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum
        history_vector = torch.sum(attention_weights * cat_embeds, dim=1)  # (B, D)

        return history_vector

    def forward(self, input_ids=None, attention_mask=None, category_history_ids=None,
                # Accept but ignore these for compatibility with other models
                hierarchical_input_ids=None, hierarchical_attention_mask=None,
                speaker_ids=None, main_cat_ids=None, sub_cat_ids=None,
                third_level_ids=None, fourth_level_ids=None, component_history_ids=None,
                **kwargs):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) - tokenized text
            attention_mask: (batch_size, seq_len) - attention mask for text
            category_history_ids: (batch_size, history_len) - previous category indices
        """
        batch_size = input_ids.size(0)

        # Encode text
        text_repr = self.encode_text(input_ids, attention_mask)  # (B, hidden_size)

        # Encode category history if provided
        if category_history_ids is not None:
            history_repr = self.encode_category_history(category_history_ids)  # (B, history_embed_dim)
        else:
            # Zero vector if no history
            history_repr = torch.zeros(batch_size, self.history_embed_dim, device=text_repr.device)

        # Fusion: concatenate and project
        combined = torch.cat([text_repr, history_repr], dim=-1)  # (B, hidden_size + history_embed_dim)
        fused = self.fusion(combined)  # (B, hidden_size)

        # Classification
        logits = self.category_head(fused)

        return logits
