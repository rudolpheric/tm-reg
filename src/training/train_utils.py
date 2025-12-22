import os
import gc
import random
import time
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ParameterGrid

from core.models import LabelSmoothingCrossEntropy, FocalLoss, CombinedLoss
from core.metrics import (
    top_k_accuracy, evaluate_transition_threshold, calculate_entropy_and_js,
    calculate_entropy_and_js_with_transitions, filter_transition_matrix,
    save_confusion_matrix_best_f1, save_transition_alignment_matrix,
    evaluate_cumulative_mass_metrics
)
from core.model_tracker import ModelTracker

def train_model(model: torch.nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                device: torch.device,
                num_epochs: int = 3,
                class_weights: torch.Tensor = None,
                tokenizer=None,
                label_encoder=None,
                use_label_smoothing: bool = False,
                label_smoothing_value: float = 0.1,
                use_focal_loss: bool = False,
                focal_alpha: float = 1.0,
                focal_gamma: float = 2.0,
                transition_matrix: pd.DataFrame = None,
                tm_weight: float = 0.1,
                use_transition_loss: bool = False,
                use_transition_loss_with_prior: bool = False,
                prior_weight: float = 0.1,
                prior_path: str = None,
                use_constraint: bool = False,
                constraint_threshold: float = 0.01,
                ce_temperature: float = 1.0,
                tm_temperature: float = 1.0,
                kl_direction: str = 'forward',
                accumulation_steps: int = 0,
                log_file_path: str = "detailed_training_log.txt",
                use_wandb: bool = False,
                wandb_project: str = "nextcat-predictor",
                wandb_run_name: str = None,
                wandb_config: dict = None,
                model_save_path: str = "eurobert_next_prediction_best.pt",
                experiment_name: str = None,
                model_key: str = None,
                # Early stopping parameters
                use_early_stopping: bool = False,
                early_stopping_patience: int = 5,
                early_stopping_min_delta: float = 0.001,
                early_stopping_monitor: str = "val_macro_f1",
                # Gradient clipping parameter
                max_grad_norm: float = 1.0) -> dict:
    """
    Train the model and validate on the validation set. Produces detailed logs.
    Now integrates the transition matrix metrics after each epoch if provided.

    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        device: Device to use for training
        num_epochs: Number of training epochs
        class_weights: Optional tensor of class weights for loss function
        tokenizer: Tokenizer for decoding inputs
        label_encoder: Label encoder for category names
        use_label_smoothing: Whether to use label smoothing in loss function
        transition_matrix: Optional dataframe with transition probabilities
        accumulation_steps: Number of steps to accumulate gradients
        log_file_path: Path to save detailed training logs
        use_wandb: Whether to log metrics to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        wandb_config: Additional config parameters to log in W&B
        use_early_stopping: Whether to use early stopping
        early_stopping_patience: Number of epochs with no improvement after which training stops
        early_stopping_min_delta: Minimum change in monitored metric to qualify as an improvement
        early_stopping_monitor: Metric to monitor ('val_loss', 'val_macro_f1', 'val_accuracy')
        max_grad_norm: Maximum gradient norm for gradient clipping
    """
    # Initialize wandb if enabled and not already initialized
    if use_wandb:
        # Check if wandb is already initialized (run is active)
        if wandb.run is None:
            if wandb_run_name is None:
                # Generate a default run name if not provided
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                label_smoothing_tag = "smooth" if use_label_smoothing else "no-smooth"
                class_weights_tag = "weighted" if class_weights is not None else "no-weights"
                wandb_run_name = f"nextcat-{timestamp}-{label_smoothing_tag}-{class_weights_tag}"
            
            # Initialize the W&B run
            config = {
                "model_type": model.__class__.__name__,
                "num_epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                "accumulation_steps": accumulation_steps,
                "use_label_smoothing": use_label_smoothing,
                "class_weights_used": class_weights is not None,
                "num_classes": len(label_encoder.classes_) if label_encoder else "unknown",
                "learning_rate": 2e-5,  # Default value, should be updated when optimizer is created
                "weight_decay": 0.01,  # Default value, should be updated when optimizer is created
            }
            
            # Update with any additional config parameters
            if wandb_config:
                config.update(wandb_config)
            
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            print(f"Initialized wandb run in train_model: {wandb.run.name}")
        else:
            print(f"Using existing wandb run: {wandb.run.name}")
        
        # Log the model architecture
        # wandb.watch(model, log="all", log_freq=10)  # Disabled to prevent model upload

    if use_transition_loss and transition_matrix is not None:
        # Use CombinedLoss when transition matrix is provided
        criterion = CombinedLoss(
            transition_matrix=transition_matrix,
            label_encoder=label_encoder,
            ce_weight=1.0,
            tm_weight=tm_weight,
            use_label_smoothing=use_label_smoothing,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            smoothing=label_smoothing_value if use_label_smoothing else 0.0,
            ce_temperature=ce_temperature,
            tm_temperature=tm_temperature,
            use_constraint=use_constraint,
            constraint_threshold=constraint_threshold,
            kl_direction=kl_direction,
            use_transition_loss_with_prior=use_transition_loss_with_prior,
            prior_weight=prior_weight,
            prior_path=prior_path
        )
    else:
        # Use original loss setup
        if use_focal_loss:
            # Use focal loss with specified parameters
            if class_weights is not None:
                # Use class weights as alpha parameter for focal loss
                criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, temperature=ce_temperature)
            else:
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, temperature=ce_temperature)
        elif use_label_smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing_value, temperature=ce_temperature)
        else:
            # For standard CrossEntropyLoss, temperature scaling needs to be applied manually in forward pass
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else torch.nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(train_dataloader), 
        eta_min=1e-6
    )
    
    # Update wandb config with actual optimizer settings
    if use_wandb:
        # Only update if these values weren't already set in wandb config
        update_dict = {}
        if "learning_rate" not in wandb.config:
            update_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
        if "weight_decay" not in wandb.config:
            update_dict["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
        if "scheduler" not in wandb.config:
            update_dict["scheduler"] = scheduler.__class__.__name__
        
        if update_dict:
            wandb.config.update(update_dict)
    
    best_f1 = 0.0  # F1 scores range from 0 to 1, so start with 0
    best_composite_score = 0.0  # Track best composite metric
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_cat_acc': [],
        'val_cat_acc': [],
        'epoch_times': []
    }

    # Initialize model tracker if model_key is provided
    model_tracker = ModelTracker() if model_key else None
    if model_tracker and model_key:
        print(f"\n=== Model Tracker Initialized for '{model_key}' ===")
        existing_best = model_tracker.get_best_model_info(model_key)
        if existing_best:
            print(f"Current best composite score: {existing_best['composite_score']:.4f}")
            print(f"  From experiment: {existing_best['experiment_name']}")
        else:
            print(f"No previous best model found for '{model_key}'")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # Early stopping setup
    early_stop_counter = 0
    best_early_stop_metric = None
    early_stopped = False
    if use_early_stopping:
        print(f"\n=== Early Stopping Enabled ===")
        print(f"  Monitor: {early_stopping_monitor}")
        print(f"  Patience: {early_stopping_patience}")
        print(f"  Min delta: {early_stopping_min_delta}")

    print("\n=== Starting Training ===")
    start_time = time.time()

    # Create the directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, train_cat_correct, train_examples = 0.0, 0, 0
        epoch_start = time.time()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)", leave=False)
        log_file.write(f"\n\n===== EPOCH {epoch+1}/{num_epochs} =====\n\n")
        scaler = GradScaler()

        batch_losses = []  # For tracking batch losses within an epoch
        # Track loss components for the epoch
        epoch_primary_losses = []
        epoch_transition_losses = []
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            category_labels = batch["category_label"].to(device)
            
            optimizer.zero_grad()
            with autocast(device_type="cuda" if device.type == "cuda" else "mps"):
                category_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                if use_transition_loss and transition_matrix is not None:
                    # Get source categories from batch
                    source_categories = batch.get("source_category", None)
                    loss_result = criterion(category_logits, category_labels, source_categories, label_encoder)
                    if isinstance(loss_result, tuple):
                        loss, loss_components = loss_result
                    else:
                        loss = loss_result
                        loss_components = {}
                else:
                    # Apply temperature scaling for standard CrossEntropyLoss if needed
                    if isinstance(criterion, torch.nn.CrossEntropyLoss) and ce_temperature != 1.0:
                        scaled_logits = category_logits / ce_temperature
                        loss = criterion(scaled_logits, category_labels)
                    else:
                        loss = criterion(category_logits, category_labels)
                    loss_components = {}

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            batch_loss = loss.item()
            total_train_loss += batch_loss
            batch_losses.append(batch_loss)
            
            with torch.no_grad():
                cat_preds = torch.argmax(category_logits, dim=1)
                train_cat_correct += (cat_preds == category_labels).sum().item()
                train_examples += input_ids.size(0)
            
            progress_bar.set_postfix(loss=batch_loss)
            
            # Log batch metrics to wandb
            if use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "batch": batch_idx + epoch * len(train_dataloader),
                    "batch_loss": batch_loss,
                    "batch_accuracy": (cat_preds == category_labels).sum().item() / input_ids.size(0),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            if batch_idx % 10 == 0 or batch_idx < 5:
                log_file.write(f"Batch {batch_idx}/{len(train_dataloader)}\nLoss: {batch_loss:.4f}\n\n")
                for i in range(min(3, input_ids.size(0))):
                    decoded_input = tokenizer.decode(input_ids[i].cpu().numpy(), skip_special_tokens=False)
                    raw_history = batch.get("raw_history", ["N/A"])[i]
                    pred_idx = cat_preds[i].item()
                    true_idx = category_labels[i].item()
                    if label_encoder is not None:
                        pred_name = label_encoder.inverse_transform([pred_idx])[0]
                        true_name = label_encoder.inverse_transform([true_idx])[0]
                    else:
                        pred_name = f"Category {pred_idx}"
                        true_name = f"Category {true_idx}"
                    confidence = torch.softmax(category_logits[i], dim=0)[pred_idx].item()
                    log_file.write(f"Example {i+1}:\nTruncated input: {decoded_input}...\n")
                    log_file.write(f"Raw history: {raw_history}...\n")
                    log_file.write(f"Predicted: {pred_name} ({pred_idx}), True: {true_name} ({true_idx})\n")
                    log_file.write(f"Confidence: {confidence:.4f}\n\n")
                log_file.write("--------------------------------------------------\n\n")
                log_file.flush()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = train_cat_correct / train_examples
        
        # Validation phase
        model.eval()
        total_val_loss, val_cat_correct, val_examples = 0.0, 0, 0
        all_cat_preds, all_cat_true = [], []
        all_logits_list = []
        all_source_categories = []  # To collect source categories for transition-based metrics
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training completed. Avg Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
        log_file.write(f"\nTraining Summary for Epoch {epoch+1}:\nAvg Loss: {avg_train_loss:.4f}\nAccuracy: {train_accuracy:.4f}\n\n=== VALIDATION ===\n\n")
        
        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)", leave=False)
        none_source_total = 0
        source_entries_total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                category_labels = batch["category_label"].to(device)
                
                category_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                if use_transition_loss and transition_matrix is not None:
                    # Get source categories from batch
                    source_categories = batch.get("source_category", None)
                    loss_result = criterion(category_logits, category_labels, source_categories, label_encoder)
                    if isinstance(loss_result, tuple):
                        loss, loss_components = loss_result
                    else:
                        loss = loss_result
                        loss_components = {}
                else:
                    # Apply temperature scaling for standard CrossEntropyLoss if needed
                    if isinstance(criterion, torch.nn.CrossEntropyLoss) and ce_temperature != 1.0:
                        scaled_logits = category_logits / ce_temperature
                        loss = criterion(scaled_logits, category_labels)
                    else:
                        loss = criterion(category_logits, category_labels)
                    loss_components = {}
                total_val_loss += loss.item()
                
                cat_preds = torch.argmax(category_logits, dim=1)
                val_cat_correct += (cat_preds == category_labels).sum().item()
                val_examples += input_ids.size(0)
                all_cat_preds.extend(cat_preds.cpu().numpy())
                all_cat_true.extend(category_labels.cpu().numpy())
                all_logits_list.append(category_logits.cpu())
                
                # Collect source categories if provided in the batch
                if transition_matrix is not None and "source_category" in batch:
                    # Track total and None counts; extend the full list to preserve alignment
                    batch_source = batch["source_category"]
                    source_entries_total += len(batch_source)
                    none_source_total += sum(1 for cat in batch_source if cat is None)
                    all_source_categories.extend(batch_source)
                elif transition_matrix is not None:
                    # Keep a single helpful message if the key is entirely missing
                    if batch_idx == 0:
                        print(f"DEBUG: transition_matrix provided but 'source_category' not in batch. Batch keys: {list(batch.keys())}")
                
                progress_bar.set_postfix(loss=loss.item(), acc=val_cat_correct / val_examples)
                
                if batch_idx < 3:
                    log_file.write(f"Validation Batch {batch_idx}:\nLoss: {loss.item():.4f}\n\n")
                    for i in range(min(3, input_ids.size(0))):
                        pred_idx = cat_preds[i].item()
                        true_idx = category_labels[i].item()
                        if label_encoder is not None:
                            pred_name = label_encoder.inverse_transform([pred_idx])[0]
                            true_name = label_encoder.inverse_transform([true_idx])[0]
                        else:
                            pred_name = f"Category {pred_idx}"
                            true_name = f"Category {true_idx}"
                        confidence = torch.softmax(category_logits[i], dim=0)[pred_idx].item()
                        log_file.write(f"Example {i+1}: Predicted: {pred_name} ({pred_idx}), True: {true_name} ({true_idx})\n")
                        log_file.write(f"Confidence: {confidence:.4f}\n\n")
        
        # Summarize None source categories once per validation phase to avoid log spam
        if transition_matrix is not None and none_source_total > 0 and source_entries_total > 0:
            print(f"DEBUG: Validation found {none_source_total} None source categories across {source_entries_total} items")

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = val_cat_correct / val_examples
        cat_f1 = f1_score(all_cat_true, all_cat_preds, average='weighted')
        epoch_duration = time.time() - epoch_start
        
        # Concatenate all logits for aggregated evaluation
        all_logits_tensor = torch.cat(all_logits_list, dim=0)
        
        # Compute aggregated top-k accuracies over the entire validation set.
        topk_acc_k2 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=2)
        topk_acc_k3 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=3)
        topk_acc_k4 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=4)
        topk_acc_k5 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=5)
        topk_acc_k6 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=6)
        
        print(f"Aggregated TOP-K Accuracy (k=2): {topk_acc_k2:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=3): {topk_acc_k3:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=4): {topk_acc_k4:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=5): {topk_acc_k5:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=6): {topk_acc_k6:.4f}")
        
        log_file.write(f"Aggregated TOP-K Accuracy (k=2): {topk_acc_k2:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=3): {topk_acc_k3:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=4): {topk_acc_k4:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=5): {topk_acc_k5:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=6): {topk_acc_k6:.4f}\n")
        # Add transition threshold evaluation if transition matrix is available
        transition_threshold_results = None
        print(f"DEBUG: transition_matrix is not None: {transition_matrix is not None}")
        print(f"DEBUG: len(all_source_categories): {len(all_source_categories)}")
        print(f"DEBUG: all_logits_tensor.size(0): {all_logits_tensor.size(0)}")
        print(f"DEBUG: First few source categories: {all_source_categories[:5] if all_source_categories else 'None'}")
        if transition_matrix is not None and len(all_source_categories) == all_logits_tensor.size(0):
            # Filter transition matrix to match available categories
            filtered_transition_matrix = filter_transition_matrix(transition_matrix, label_encoder)
            
            # Run transition threshold evaluation with different thresholds
            thresholds = [0.05, 0.1, 0.2, 0.5]
            transition_threshold_results = {}
            
            for threshold in thresholds:
                results = evaluate_transition_threshold(
                    logits=all_logits_tensor,
                    targets=torch.tensor(all_cat_true),
                    source_categories=all_source_categories,
                    transition_matrix=filtered_transition_matrix,
                    label_encoder=label_encoder,
                    threshold=threshold
                )
                
                # Store results for this threshold
                transition_threshold_results[threshold] = results
                
                # Log key metrics
                log_file.write(f"\nTransition Threshold Metrics (threshold={threshold}):\n")
                log_file.write(f"  Valid source categories: {results.get('valid_source_count', 0)}/{len(all_cat_true)}\n")
                log_file.write(f"  Predictions above threshold: {results.get('predictions_above_threshold_percentage', 0):.4f}\n")
                log_file.write(f"  Correct & above threshold: {results.get('correct_and_above_threshold', 0):.4f}\n")
                log_file.write(f"  Top-1 in valid transitions: {results.get('top1_in_valid_transitions', 0):.4f}\n")
                log_file.write(f"  Top-3 in valid transitions: {results.get('top3_in_valid_transitions', 0):.4f}\n")
                
                if 'transition_aware_accuracy' in results:
                    log_file.write(f"  Transition-aware accuracy: {results['transition_aware_accuracy']:.4f}\n")
                if 'constrained_accuracy' in results:
                    log_file.write(f"  Constrained (masked) accuracy: {results['constrained_accuracy']:.4f}\n")
                
                print(f"Transition Threshold Metrics (t={threshold}): "
                    f"Correct & above threshold: {results.get('correct_and_above_threshold', 0):.4f}, "
                    f"Top-3 in valid: {results.get('top3_in_valid_transitions', 0):.4f}")

            # Run cumulative mass evaluation (adaptive approach)
            cumulative_results = evaluate_cumulative_mass_metrics(
                logits=all_logits_tensor,
                targets=torch.tensor(all_cat_true),
                source_categories=all_source_categories,
                transition_matrix=filtered_transition_matrix,
                label_encoder=label_encoder,
                cumulative_thresholds=[0.7, 0.8, 0.9]
            )

            # Log cumulative mass metrics
            log_file.write(f"\nCumulative Mass Metrics (Adaptive):\n")
            log_file.write(f"  Valid source categories: {cumulative_results.get('valid_source_count_cumulative', 0)}/{len(all_cat_true)}\n")
            for thresh_pct in [70, 80, 90]:
                pred_acc = cumulative_results.get(f'cumulative{thresh_pct}_pred_accuracy', 0)
                true_cov = cumulative_results.get(f'cumulative{thresh_pct}_true_coverage', 0)
                pred_acc_plus = cumulative_results.get(f'cumulative{thresh_pct}_pred_accuracy_plus_true', 0)

                log_file.write(f"  Cumulative {thresh_pct}%:\n")
                log_file.write(f"    Pred in valid: {pred_acc:.4f}\n")
                log_file.write(f"    True coverage: {true_cov:.4f}\n")
                log_file.write(f"    Pred in valid+true: {pred_acc_plus:.4f}\n")

            print(f"Cumulative Mass Metrics: "
                  f"80% pred_accuracy={cumulative_results.get('cumulative80_pred_accuracy', 0):.4f}, "
                  f"true_coverage={cumulative_results.get('cumulative80_true_coverage', 0):.4f}")

            # Log to W&B if enabled
            if use_wandb and wandb.run is not None:
                wandb_cumulative = {}
                for key, value in cumulative_results.items():
                    if key != 'valid_source_count_cumulative':
                        wandb_cumulative[f"val_{key}"] = value
                wandb.log(wandb_cumulative)

        # Compute entropy and JS divergence using transition matrix metrics if possible.
        print(f"DEBUG (entropy): Checking transition matrix conditions again")
        if transition_matrix is not None and len(all_source_categories) == all_logits_tensor.size(0):
            filtered_transition_matrix = filter_transition_matrix(transition_matrix, label_encoder) # remove labels from transition matrix since I removed labels in training bc. of low label count
            mean_entropy, mean_js_div = calculate_entropy_and_js_with_transitions(
                pred_logits=all_logits_tensor,
                source_categories=all_source_categories,
                transition_matrix=filtered_transition_matrix,
                label_encoder=label_encoder
            )
            log_file.write(f"Transition Matrix Metrics:\nMean Entropy: {mean_entropy:.4f}\nMean JS Divergence: {mean_js_div:.4f}\n\n")
            print(f"Transition Matrix Metrics: Mean Entropy: {mean_entropy:.4f}, Mean JS Divergence: {mean_js_div:.4f}")
        else:
            mean_entropy, mean_js_div = calculate_entropy_and_js(all_logits_tensor)
            log_file.write(f"Baseline Metrics:\nMean Entropy: {mean_entropy:.4f}\nMean JS Divergence: {mean_js_div:.4f}\n\n")
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['train_cat_acc'].append(train_accuracy)
        training_history['val_cat_acc'].append(val_accuracy)
        training_history['epoch_times'].append(epoch_duration)
        
        # Log metrics to W&B
        if use_wandb:
            metrics_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "val_f1": cat_f1,
                "epoch_time": epoch_duration,
                "top_k_acc_2": topk_acc_k2,
                "top_k_acc_3": topk_acc_k3,
                "top_k_acc_4": topk_acc_k4,
                "top_k_acc_5": topk_acc_k5,
                "top_k_acc_6": topk_acc_k6,
                "mean_entropy": mean_entropy,
                "mean_js_divergence": mean_js_div,
                "learning_rate": scheduler.get_last_lr()[0]
            }
            wandb.log(metrics_dict)
            print(f"DEBUG: transition_threshold_results is not None: {transition_threshold_results is not None}")
            print(f"DEBUG: transition_threshold_results content: {transition_threshold_results}")
            if transition_threshold_results:
                for threshold, results in transition_threshold_results.items():
                    threshold_metrics = {
                        f"transition_t{threshold}_correct_above": results.get('correct_and_above_threshold', 0),
                        f"transition_t{threshold}_top1_in_valid": results.get('top1_in_valid_transitions', 0),
                        f"transition_t{threshold}_top3_in_valid": results.get('top3_in_valid_transitions', 0),
                    }
                    if 'transition_aware_accuracy' in results:
                        threshold_metrics[f"transition_t{threshold}_aware_accuracy"] = results['transition_aware_accuracy']
                    if 'constrained_accuracy' in results:
                        threshold_metrics[f"transition_t{threshold}_constrained_acc"] = results['constrained_accuracy']

                    wandb.log(threshold_metrics)
            # Create a histogram of batch losses
            if len(batch_losses) > 0:
                wandb.log({"batch_loss_distribution": wandb.Histogram(batch_losses)})
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Validation completed. Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | F1: {cat_f1:.4f} | Epoch time: {epoch_duration:.2f}s")
        log_file.write(f"\nValidation Summary for Epoch {epoch+1}:\nAvg Loss: {avg_val_loss:.4f}\nAccuracy: {val_accuracy:.4f}\nF1 Score: {cat_f1:.4f}\n")
        log_file.write(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n\n")
        
        if label_encoder:
            try:
                unique_true = np.unique(all_cat_true)
                unique_classes = np.unique(np.concatenate([unique_true, np.unique(all_cat_preds)]))
                category_names = label_encoder.inverse_transform(unique_classes)
                cm = confusion_matrix(all_cat_true, all_cat_preds, labels=unique_classes)
                conf_matrix = pd.DataFrame(cm, index=category_names, columns=category_names)
                log_file.write("Confusion Matrix:\n" + conf_matrix.to_string() + "\n\n")
                log_file.write(f"Note: {len(unique_classes)} out of {len(label_encoder.classes_)} categories in validation run\n\n")
                
                # Log confusion matrix to W&B
                if use_wandb:
                    # Create mapping from category IDs to indices for wandb confusion matrix
                    id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_classes)}
                    
                    # Map the true labels and predictions to indices
                    y_true_indexed = [id_to_idx[cat_id] for cat_id in all_cat_true]
                    preds_indexed = [id_to_idx[cat_id] for cat_id in all_cat_preds]
                    
                    confusion_matrix_wandb = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true_indexed,
                        preds=preds_indexed,
                        class_names=category_names.tolist()
                    )
                    wandb.log({"confusion_matrix": confusion_matrix_wandb})
            except Exception as e:
                log_file.write(f"Error generating confusion matrix: {str(e)}\n\n")
            
            try:
                unique_classes = np.unique(all_cat_true)
                category_names_present = label_encoder.inverse_transform(unique_classes)
                report = classification_report(
                    all_cat_true, all_cat_preds,
                    labels=unique_classes, target_names=category_names_present,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                log_file.write("Classification Report:\n" + report_df.to_string() + "\n\n")
                
                # Log per-class metrics to W&B
                if use_wandb:
                    for cls_name, metrics in report.items():
                        if isinstance(metrics, dict):
                            wandb.log({
                                f"class_{cls_name}_precision": metrics["precision"],
                                f"class_{cls_name}_recall": metrics["recall"],
                                f"class_{cls_name}_f1": metrics["f1-score"],
                                f"class_{cls_name}_support": metrics["support"],
                            })
            except Exception as e:
                log_file.write(f"Error generating classification report: {str(e)}\n\n")

        # Calculate composite metric: F1 + top-k accuracy + transition validity
        # Weights: 0.5 for F1, 0.25 for top-k acc, 0.25 for transition validity
        transition_t005_top1 = 0.0
        if transition_threshold_results and 0.05 in transition_threshold_results:
            transition_t005_top1 = transition_threshold_results[0.05].get('top1_in_valid_transitions', 0)

        composite_score = 0.5 * cat_f1 + 0.25 * topk_acc_k3 + 0.25 * transition_t005_top1

        log_file.write(f"\nComposite Metric (epoch {epoch+1}):\n")
        log_file.write(f"  F1 Score: {cat_f1:.4f} (weight: 0.5)\n")
        log_file.write(f"  Top-3 Accuracy: {topk_acc_k3:.4f} (weight: 0.25)\n")
        log_file.write(f"  Transition T0.05 Top1: {transition_t005_top1:.4f} (weight: 0.25)\n")
        log_file.write(f"  Composite Score: {composite_score:.4f}\n\n")

        if use_wandb:
            wandb.log({
                "composite_score": composite_score,
                "composite_f1_component": cat_f1,
                "composite_topk3_component": topk_acc_k3,
                "composite_transition_component": transition_t005_top1
            })

        # Save model based on composite score (with model tracker check)
        if composite_score > best_composite_score:
            best_composite_score = composite_score

            # Check with model tracker if we should save
            should_save = True
            if model_tracker and model_key:
                should_save, prev_best = model_tracker.should_save_model(
                    model_key, composite_score, experiment_name
                )
                if not should_save:
                    print(f"  New best in this run! {composite_score:.4f} but NOT better than tracked best: {prev_best:.4f}")
                    log_file.write(f"*** New best in this run: {composite_score:.4f}, but tracked best is {prev_best:.4f}. Not saving. ***\n\n")
                else:
                    print(f"  New GLOBAL best Composite Score! {composite_score:.4f} (F1: {cat_f1:.4f}, Top-3: {topk_acc_k3:.4f}, Trans: {transition_t005_top1:.4f})")
                    if prev_best:
                        print(f"    Previous best: {prev_best:.4f}")
                    log_file.write(f"*** New GLOBAL best Composite Score! {composite_score:.4f} Saving model... ***\n\n")
            else:
                print(f"  New best Composite Score! {composite_score:.4f} (F1: {cat_f1:.4f}, Top-3: {topk_acc_k3:.4f}, Trans: {transition_t005_top1:.4f}) Saving model...")
                log_file.write(f"*** New best Composite Score! {composite_score:.4f} Saving model... ***\n\n")

            if should_save:
                save_model(model, model_save_path, device)

                # Update tracker
                if model_tracker and model_key:
                    model_tracker.update_best_model(
                        model_key=model_key,
                        composite_score=composite_score,
                        f1_score=cat_f1,
                        top_k_acc=topk_acc_k3,
                        transition_score=transition_t005_top1,
                        experiment_name=experiment_name or "unknown",
                        epoch=epoch + 1,
                        model_path=model_save_path
                    )

            # Save confusion matrix for best composite score model (only if model was saved)
            if should_save and label_encoder is not None:
                try:
                    # Use experiment_name or fallback to wandb_run_name for organized saving
                    exp_name = experiment_name or wandb_run_name or "default_experiment"
                    save_confusion_matrix_best_f1(
                        all_cat_true, all_cat_preds, label_encoder, composite_score,
                        f"confusion_matrix_best_composite_epoch_{epoch+1}.png",
                        experiment_name=exp_name
                    )
                except Exception as e:
                    print(f"Error saving confusion matrix: {e}")
                    log_file.write(f"Error saving confusion matrix: {e}\n\n")

                # Save transition alignment matrix if transition matrix is available
                if transition_matrix is not None and len(all_source_categories) > 0:
                    try:
                        print(f"Generating transition alignment matrix...")
                        save_transition_alignment_matrix(
                            y_true=all_cat_true,
                            y_pred=all_cat_preds,
                            source_categories=all_source_categories,
                            transition_matrix=transition_matrix,
                            label_encoder=label_encoder,
                            f1_score_val=composite_score,
                            threshold=0.05,  # Can be made configurable
                            save_path=f"transition_alignment_best_composite_epoch_{epoch+1}.png",
                            experiment_name=exp_name
                        )
                    except Exception as e:
                        print(f"Error saving transition alignment matrix: {e}")
                        log_file.write(f"Error saving transition alignment matrix: {e}\n\n")
            
            # Log best model to W&B
            if use_wandb:
                wandb.log({
                    "best_model_epoch": epoch + 1,
                    "best_model_f1": cat_f1,
                    "best_model_accuracy": val_accuracy
                })
                # Save the best model to W&B
                # wandb.save(model_save_path)  # Disabled to prevent model upload
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Early stopping check
        if use_early_stopping:
            # Get the metric to monitor
            if early_stopping_monitor == "val_loss":
                current_metric = avg_val_loss
                is_improvement = best_early_stop_metric is None or current_metric < (best_early_stop_metric - early_stopping_min_delta)
            elif early_stopping_monitor == "val_macro_f1":
                current_metric = cat_f1
                is_improvement = best_early_stop_metric is None or current_metric > (best_early_stop_metric + early_stopping_min_delta)
            elif early_stopping_monitor == "val_accuracy":
                current_metric = val_accuracy
                is_improvement = best_early_stop_metric is None or current_metric > (best_early_stop_metric + early_stopping_min_delta)
            else:
                # Default to val_macro_f1
                current_metric = cat_f1
                is_improvement = best_early_stop_metric is None or current_metric > (best_early_stop_metric + early_stopping_min_delta)

            if is_improvement:
                best_early_stop_metric = current_metric
                early_stop_counter = 0
                print(f"  Early stopping: {early_stopping_monitor} improved to {current_metric:.4f}")
            else:
                early_stop_counter += 1
                print(f"  Early stopping: {early_stopping_monitor} did not improve. Counter: {early_stop_counter}/{early_stopping_patience}")

            if early_stop_counter >= early_stopping_patience:
                print(f"\n=== Early Stopping Triggered at Epoch {epoch+1} ===")
                print(f"  No improvement in {early_stopping_monitor} for {early_stopping_patience} epochs")
                print(f"  Best {early_stopping_monitor}: {best_early_stop_metric:.4f}")
                log_file.write(f"\n*** Early stopping triggered at epoch {epoch+1}. Best {early_stopping_monitor}: {best_early_stop_metric:.4f} ***\n")
                early_stopped = True
                break

    total_training_time = time.time() - start_time
    if early_stopped:
        print(f"\nTraining stopped early after {epoch+1} epochs. Total time: {total_training_time:.2f} seconds")
    else:
        print(f"\nTotal training time: {total_training_time:.2f} seconds")
    log_file.write(f"\nTotal training time: {total_training_time:.2f} seconds")
    log_file.close()
    
    # Finish W&B run
    if use_wandb:
        # Log final metrics and hyperparameters
        wandb.log({
            "total_training_time": total_training_time,
            "best_f1": best_f1
        })
        
        # Save training history to W&B
        history_df = pd.DataFrame(training_history)
        wandb.log({"training_history": wandb.Table(dataframe=history_df)})
        
        # Create a line plot of training and validation losses
        if len(training_history['train_loss']) > 0:
            epochs = list(range(1, num_epochs + 1))
            loss_table = wandb.Table(data=[[e, tl, vl] for e, tl, vl in 
                                          zip(epochs, 
                                              training_history['train_loss'], 
                                              training_history['val_loss'])],
                                    columns=["epoch", "train_loss", "val_loss"])
            wandb.log({"loss_plot": wandb.plot.line(
                loss_table, "epoch", ["train_loss", "val_loss"], 
                title="Training and Validation Loss")})
        
        # Don't finish wandb here - let the main script handle wandb lifecycle
        # wandb.finish()
    
    return training_history


def evaluate_model_on_test_set(model: torch.nn.Module,
                               test_dataloader: DataLoader,
                               device: torch.device,
                               dataset,
                               transition_matrix: pd.DataFrame = None,
                               use_wandb: bool = False) -> dict:
    """
    Evaluate the model on the test set and return evaluation metrics.
    
    Args:
        model: The trained model
        test_dataloader: DataLoader for test data
        device: Device to use for evaluation
        dataset: Dataset object containing the label encoder
        transition_matrix: Optional dataframe with transition probabilities
        use_wandb: Whether to log metrics to Weights & Biases
    """
    model.eval()
    all_cat_preds, all_cat_true = [], []
    all_cat_names_pred, all_cat_names_true = [], []
    all_logits = []
    all_source_categories = []  # To store the source (current) category for each sample
    
    print("Starting test set evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Set Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            category_labels = batch["category_label"].to(device)
            
            # Check if this is a refactored model that needs legacy format
            is_refactored = hasattr(model, 'use_backward_compatibility')
            
            if is_refactored:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_legacy_format=True
                )
            else:
                # Check if model needs source_categories (e.g., TransitionMatrixBaseline)
                if hasattr(model, '__class__') and model.__class__.__name__ == 'TransitionMatrixBaseline':
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        source_categories=batch.get("source_category", None)
                    )
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                category_logits = outputs["category_logits"]
            elif isinstance(outputs, tuple):
                category_logits = outputs[0]
            else:
                category_logits = outputs
                
            all_logits.append(category_logits.cpu())
            
            cat_preds = torch.argmax(category_logits, dim=1).cpu().numpy()
            all_cat_preds.extend(cat_preds)
            all_cat_true.extend(category_labels.cpu().numpy())
            
            if hasattr(dataset, 'label_encoder'):
                cat_names_pred = dataset.label_encoder.inverse_transform(cat_preds)
                cat_names_true = dataset.label_encoder.inverse_transform(category_labels.cpu().numpy())
                all_cat_names_pred.extend(cat_names_pred)
                all_cat_names_true.extend(cat_names_true)
            
            # Collect source categories if provided in the batch
            if transition_matrix is not None and "source_category" in batch:
                all_source_categories.extend(batch["source_category"])
    
    # Check if we have any data to evaluate
    if len(all_logits) == 0:
        print("Warning: No batches were processed during evaluation. Test loader might be empty.")
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "mean_entropy": 0.0,
            "mean_js_divergence": 0.0,
            "num_samples": 0,
            "error": "No data to evaluate - empty test loader"
        }
    
    all_logits_tensor = torch.cat(all_logits, dim=0)
    
    if transition_matrix is not None and len(all_source_categories) == all_logits_tensor.size(0):
        filtered_transition_matrix = filter_transition_matrix(transition_matrix, dataset.label_encoder) 
        mean_entropy, mean_js_div = calculate_entropy_and_js_with_transitions(
            pred_logits=all_logits_tensor,
            source_categories=all_source_categories,
            transition_matrix=filtered_transition_matrix,
            label_encoder=dataset.label_encoder
        )
    else:
        mean_entropy, mean_js_div = calculate_entropy_and_js(all_logits_tensor)
    
    # Calculate accuracy and F1 score
    test_accuracy = accuracy_score(all_cat_true, all_cat_preds)
    test_f1 = f1_score(all_cat_true, all_cat_preds, average='weighted')
    
    # Calculate top-k metrics
    top_k_metrics = {}
    for k in [2, 3, 4, 5, 6]:
        top_k_metrics[f'top_k_acc_{k}'] = top_k_accuracy(
            all_logits_tensor, 
            torch.tensor(all_cat_true), 
            k=k
        )
    
    # Log test metrics to W&B if enabled
    if use_wandb:
        # Ensure W&B is initialized (in case this function is called separately)
        if not wandb.run:
            wandb.init(project="nextcat-predictor", name="test-evaluation")
        
        metrics_dict = {
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_mean_entropy": mean_entropy,
            "test_mean_js_divergence": mean_js_div,
        }
        metrics_dict.update({f"test_{k}": v for k, v in top_k_metrics.items()})
        wandb.log(metrics_dict)
        
        # Log confusion matrix if label encoder is available
        if hasattr(dataset, 'label_encoder'):
            try:
                unique_true = np.unique(all_cat_true)
                unique_classes = np.unique(np.concatenate([unique_true, np.unique(all_cat_preds)]))
                category_names = dataset.label_encoder.inverse_transform(unique_classes)
                
                # Create mapping from category IDs to indices for wandb confusion matrix
                id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_classes)}
                
                # Map the true labels and predictions to indices
                y_true_indexed = [id_to_idx[cat_id] for cat_id in all_cat_true]
                preds_indexed = [id_to_idx[cat_id] for cat_id in all_cat_preds]
                
                test_cm = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_indexed,
                    preds=preds_indexed,
                    class_names=category_names.tolist()
                )
                wandb.log({"test_confusion_matrix": test_cm})
                
                # Generate and log classification report
                report = classification_report(
                    all_cat_true, all_cat_preds,
                    labels=unique_classes,
                    target_names=category_names,
                    output_dict=True
                )
                for cls_name, metrics in report.items():
                    if isinstance(metrics, dict):
                        wandb.log({
                            f"test_class_{cls_name}_precision": metrics["precision"],
                            f"test_class_{cls_name}_recall": metrics["recall"],
                            f"test_class_{cls_name}_f1": metrics["f1-score"],
                            f"test_class_{cls_name}_support": metrics["support"],
                        })
            except Exception as e:
                print(f"Error logging test metrics to W&B: {str(e)}")
    
    results = {
        'cat_accuracy': test_accuracy,
        'cat_f1': test_f1,
        'cat_true': all_cat_true,
        'cat_pred': all_cat_preds,
        'cat_names_true': all_cat_names_true,
        'cat_names_pred': all_cat_names_pred,
        'mean_entropy': mean_entropy,
        'mean_js_divergence': mean_js_div
    }
    results.update(top_k_metrics)
    
    # Evaluate transition threshold metrics if transition matrix is provided
    transition_threshold_results = None
    if transition_matrix is not None and len(all_source_categories) == all_logits_tensor.size(0):
        filtered_transition_matrix = filter_transition_matrix(transition_matrix, dataset.label_encoder)
        
        # Use multiple thresholds for a more complete analysis
        thresholds = [0.05, 0.1, 0.2, 0.5]
        transition_threshold_results = {}
        
        for threshold in thresholds:
            threshold_results = evaluate_transition_threshold(
                logits=all_logits_tensor,
                targets=torch.tensor(all_cat_true),
                source_categories=all_source_categories,
                transition_matrix=filtered_transition_matrix,
                label_encoder=dataset.label_encoder,
                threshold=threshold
            )
            
            transition_threshold_results[threshold] = threshold_results
            
            # Log to W&B if enabled
            if use_wandb and wandb.run is not None:
                wandb.log({
                    f"test_transition_t{threshold}_correct_above": threshold_results.get('correct_and_above_threshold', 0),
                    f"test_transition_t{threshold}_top1_in_valid": threshold_results.get('top1_in_valid_transitions', 0),
                    f"test_transition_t{threshold}_top3_in_valid": threshold_results.get('top3_in_valid_transitions', 0),
                    f"test_transition_t{threshold}_aware_accuracy": threshold_results.get('transition_aware_accuracy', 0),
                    f"test_transition_t{threshold}_constrained_acc": threshold_results.get('constrained_accuracy', 0),
                })

        # Run cumulative mass evaluation (adaptive approach)
        cumulative_results = evaluate_cumulative_mass_metrics(
            logits=all_logits_tensor,
            targets=torch.tensor(all_cat_true),
            source_categories=all_source_categories,
            transition_matrix=filtered_transition_matrix,
            label_encoder=dataset.label_encoder,
            cumulative_thresholds=[0.7, 0.8, 0.9]
        )

        # Log to W&B if enabled
        if use_wandb and wandb.run is not None:
            wandb_cumulative = {}
            for key, value in cumulative_results.items():
                if key != 'valid_source_count_cumulative':
                    wandb_cumulative[f"test_{key}"] = value
            wandb.log(wandb_cumulative)

        # Add to results
        results['cumulative_mass_metrics'] = cumulative_results


    return results


def save_model(model: torch.nn.Module, path: str, device: torch.device):
    """
    Save the model state to disk. Moves the state to CPU if necessary.
    """
    if device.type == "cuda":
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, path)
    else:
        torch.save(model.state_dict(), path)


def run_hyperparameter_search(base_model_class, dataset, split_info, device, use_wandb=False, wandb_project="nextcat-hyperparam"):
    """
    Run a basic hyperparameter search with optional W&B integration for tracking.
    
    Args:
        base_model_class: The model class to instantiate
        dataset: Dataset object
        split_info: Dictionary containing train and validation indices
        device: Device to use for training/evaluation
        use_wandb: Whether to log metrics to Weights & Biases
        wandb_project: W&B project name for the hyperparameter search
    """
    param_grid = {
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'batch_size': [4, 8],
        'weight_decay': [0.01, 0.1],
        'label_smoothing': [0.0, 0.1],
        'use_class_weight': [True, False]
    }
    
    results = []
    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]
    
    # Create smaller subsets for faster hyperparam search
    subset_size = min(1000, len(train_indices))
    train_subset = random.sample(train_indices, subset_size)
    val_subset = random.sample(val_indices, min(200, len(val_indices)))
    
    # Initialize W&B sweep if enabled
    if use_wandb:
        sweep_config = {
            'method': 'grid',  # grid, random, bayes
            'parameters': {
                'learning_rate': {
                    'values': [1e-5, 2e-5, 5e-5]
                },
                'batch_size': {
                    'values': [16, 32, 64]
                },
                'weight_decay': {
                    'values': [0.01, 0.1]
                },
                'label_smoothing': {
                    'values': [0.0, 0.1]
                },
                'use_class_weight': {
                    'values': [True, False]
                }
            },
            'metric': {
                'name': 'val_f1',
                'goal': 'maximize'
            }
        }
        sweep_id = wandb.sweep(sweep_config, project=wandb_project)
        
        # Define the sweep agent function
        def sweep_agent():
            # Initialize a new wandb run
            wandb.init()
            
            # Get hyperparameters from wandb
            config = wandb.config
            
            # Create model with current params
            model = base_model_class(num_categories=len(dataset.label_encoder.classes_)).to(device)
            
            # Configure training with current params
            train_loader = DataLoader(
                Subset(dataset, train_subset), 
                batch_size=config.batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                Subset(dataset, val_subset), 
                batch_size=config.batch_size,
                shuffle=False
            )
            
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config.learning_rate, 
                weight_decay=config.weight_decay
            )
            
            # Use class weights if specified
            if config.use_class_weight:
                class_counts = np.bincount([dataset[i]["category_label"].item() for i in train_subset])
                class_weights = torch.FloatTensor(compute_class_weight(
                    'balanced', 
                    classes=np.unique([dataset[i]["category_label"].item() for i in train_subset]), 
                    y=[dataset[i]["category_label"].item() for i in train_subset]
                )).to(device)
            else:
                class_weights = None
            
            # Use label smoothing if specified
            if config.label_smoothing > 0:
                criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else torch.nn.CrossEntropyLoss()
            
            # Log the model architecture
            # wandb.watch(model, log="all", log_freq=10)  # Disabled to prevent model upload
            
            # Train for a few epochs
            for epoch in range(5):  # Just 2 epochs for hyperparameter search
                model.train()
                
                # Training loop
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    category_labels = batch["category_label"].to(device)
                    
                    optimizer.zero_grad()
                    category_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(category_logits, category_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    preds = torch.argmax(category_logits, dim=1)
                    train_correct += (preds == category_labels).sum().item()
                    train_total += input_ids.size(0)
                
                # Calculate training metrics
                train_avg_loss = train_loss / len(train_loader)
                train_accuracy = train_correct / train_total
                
                # Log training metrics
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_avg_loss,
                    "train_accuracy": train_accuracy
                })
            
                # Validation loop
                model.eval()
                val_loss = 0.0
                all_preds, all_true = [], []
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        category_labels = batch["category_label"].to(device)
                        
                        category_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(category_logits, category_labels)
                        val_loss += loss.item()
                        
                        preds = torch.argmax(category_logits, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_true.extend(category_labels.cpu().numpy())
                
                # Calculate validation metrics
                val_avg_loss = val_loss / len(val_loader)
                val_accuracy = accuracy_score(all_true, all_preds)
                val_f1 = f1_score(all_true, all_preds, average='weighted')
                
                # Log validation metrics
                wandb.log({
                    "epoch": epoch + 1,
                    "val_loss": val_avg_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1": val_f1
                })
            
            # Log final metrics for sweep comparison
            wandb.log({
                "final_val_accuracy": val_accuracy,
                "final_val_f1": val_f1
            })
            
            # Don't close wandb here - let the main script handle wandb lifecycle
            # wandb.finish()
            
            return {
                'params': dict(config),
                'val_f1': val_f1,
                'val_acc': val_accuracy
            }
        
        # Run the sweep
        wandb.agent(sweep_id, function=sweep_agent, count=len(ParameterGrid(param_grid)))
        
        # Get the best parameters from the sweep
        api = wandb.Api()
        sweep = api.sweep(f"{wandb.run.entity}/{wandb_project}/{sweep_id}")
        best_run = sorted(sweep.runs, key=lambda run: run.summary.get('val_f1', 0), reverse=True)[0]
        best_params = {k: v for k, v in best_run.config.items() if k in param_grid}
        
        print(f"\nBest parameters from W&B sweep: {best_params}")
        print(f"Best validation F1: {best_run.summary.get('val_f1', 0):.4f}")
        
        return best_params
    
    # Standard hyperparameter search without W&B
    else:
        for params in tqdm(ParameterGrid(param_grid), desc="Hyperparameter Search"):
            print(f"\nTrying parameters: {params}")
            
            # Create model with current params
            model = base_model_class(num_categories=len(dataset.label_encoder.classes_)).to(device)
            
            # Configure training with current params
            train_loader = DataLoader(
                Subset(dataset, train_subset), 
                batch_size=params['batch_size'],
                shuffle=True
            )
            
            val_loader = DataLoader(
                Subset(dataset, val_subset), 
                batch_size=params['batch_size'],
                shuffle=False
            )
            
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=params['learning_rate'], 
                weight_decay=params['weight_decay']
            )
            
            # Use label smoothing if specified
            if params['label_smoothing'] > 0:
                criterion = LabelSmoothingCrossEntropy(smoothing=params['label_smoothing'])
            else:
                criterion = torch.nn.CrossEntropyLoss()
            
            # Train for a few epochs
            for epoch in range(2):  # Just 2 epochs for hyperparameter search
                model.train()
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    category_labels = batch["category_label"].to(device)
                    
                    optimizer.zero_grad()
                    category_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(category_logits, category_labels)
                    loss.backward()
                    optimizer.step()
            
            # Quick validation
            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    category_labels = batch["category_label"].to(device)
                    
                    category_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(category_logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_true.extend(category_labels.cpu().numpy())
            
            val_f1 = f1_score(all_true, all_preds, average='weighted')
            val_acc = accuracy_score(all_true, all_preds)
            
            results.append({
                'params': params,
                'val_f1': val_f1,
                'val_acc': val_acc
            })
            
            print(f"Validation F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        
        # Find best params
        best_result = max(results, key=lambda x: x['val_f1'])
        print(f"\nBest parameters: {best_result['params']}")
        print(f"Best validation F1: {best_result['val_f1']:.4f}, Acc: {best_result['val_acc']:.4f}")
        
        return best_result['params']
    
def train_hierarchy_aware_model(model: torch.nn.Module,
                               train_dataloader: DataLoader,
                               val_dataloader: DataLoader,
                               device: torch.device,
                               num_epochs: int = 3,
                               class_weights: torch.Tensor = None,
                               tokenizer=None,
                               label_encoder=None,
                               use_label_smoothing: bool = False,
                               label_smoothing_value: float = 0.1,
                               use_focal_loss: bool = False,
                               focal_alpha: float = 1.0,
                               focal_gamma: float = 2.0,
                               transition_matrix: pd.DataFrame = None,
                               tm_weight: float = 0.1,
                               use_transition_loss: bool = False,
                               use_transition_loss_with_prior: bool = False,
                               prior_weight: float = 0.1,
                               prior_path: str = None,
                               use_constraint: bool = False,
                               constraint_threshold: float = 0.01,
                               ce_temperature: float = 1.0,
                               tm_temperature: float = 1.0,
                               kl_direction: str = 'forward',
                               hierarchy_info: dict = None,
                               accumulation_steps: int = 0,
                               log_file_path: str = "detailed_training_log.txt",
                               use_wandb: bool = False,
                               wandb_project: str = "nextcat-predictor",
                               wandb_run_name: str = None,
                               wandb_config: dict = None,
                               model_save_path: str = "hierarchical_next_prediction_best.pt",
                               experiment_name: str = None,
                               model_key: str = None,
                               # Early stopping parameters
                               use_early_stopping: bool = False,
                               early_stopping_patience: int = 5,
                               early_stopping_min_delta: float = 0.001,
                               early_stopping_monitor: str = "val_macro_f1",
                               # Gradient clipping parameter
                               max_grad_norm: float = 1.0) -> dict:
    """
    Train a hierarchical model and validate on the validation set. 
    Supports structured label prediction with multi-task learning.
    
    Args:
        model: The hierarchical model to train
        train_dataloader: DataLoader for training data (hierarchical format)
        val_dataloader: DataLoader for validation data (hierarchical format)
        device: Device to use for training
        num_epochs: Number of training epochs
        class_weights: Optional tensor of class weights for loss function
        tokenizer: Tokenizer for decoding inputs
        label_encoder: Label encoder for category names
        use_label_smoothing: Whether to use label smoothing in loss function
        transition_matrix: Optional dataframe with transition probabilities
        hierarchy_info: Dictionary with hierarchical category information
        accumulation_steps: Number of steps to accumulate gradients
        log_file_path: Path to save detailed training logs
        use_wandb: Whether to log metrics to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        wandb_config: Additional config parameters to log in W&B
    """
    # Initialize wandb if enabled and not already initialized
    if use_wandb:
        # Check if wandb is already initialized (run is active)
        if wandb.run is None:
            if wandb_run_name is None:
                # Generate a default run name if not provided
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                label_smoothing_tag = "smooth" if use_label_smoothing else "no-smooth"
                class_weights_tag = "weighted" if class_weights is not None else "no-weights"
                hierarchy_tag = "hierarchical" if hierarchy_info is not None else "flat"
                wandb_run_name = f"nextcat-{timestamp}-{hierarchy_tag}-{label_smoothing_tag}-{class_weights_tag}"
            
            # Initialize the W&B run
            config = {
                "model_type": model.__class__.__name__,
                "num_epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                "accumulation_steps": accumulation_steps,
                "use_label_smoothing": use_label_smoothing,
                "class_weights_used": class_weights is not None,
                "num_classes": len(label_encoder.classes_) if label_encoder else "unknown",
                "learning_rate": 2e-5,  # Default value, updated when optimizer is created
                "weight_decay": 0.01,   # Default value, updated when optimizer is created
                "hierarchical_model": hierarchy_info is not None,
            }
            
            # Add hierarchy information if available
            if hierarchy_info:
                config.update({
                    "num_speakers": hierarchy_info.get("num_speakers", 0),
                    "num_main_categories": hierarchy_info.get("num_main_categories", 0),
                    "num_sub_categories": hierarchy_info.get("num_sub_categories", 0),
                    "num_third_level": hierarchy_info.get("num_third_level", 0),
                    "num_fourth_level": hierarchy_info.get("num_fourth_level", 0),
                })
            
            # Update with any additional config parameters
            if wandb_config:
                config.update(wandb_config)
            
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            print(f"Initialized wandb run in train_hierarchy_aware_model: {wandb.run.name}")
        else:
            print(f"Using existing wandb run: {wandb.run.name}")
        
        # Log the model architecture
        # wandb.watch(model, log="all", log_freq=10)  # Disabled to prevent model upload
    
    # Setup loss function
    if use_transition_loss and transition_matrix is not None:
        # Use CombinedLoss when transition matrix is provided
        criterion = CombinedLoss(
            transition_matrix=transition_matrix,
            label_encoder=label_encoder,
            ce_weight=1.0,
            tm_weight=tm_weight,
            use_label_smoothing=use_label_smoothing,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            smoothing=label_smoothing_value if use_label_smoothing else 0.0,
            ce_temperature=ce_temperature,
            tm_temperature=tm_temperature,
            use_constraint=use_constraint,
            constraint_threshold=constraint_threshold,
            kl_direction=kl_direction,
            use_transition_loss_with_prior=use_transition_loss_with_prior,
            prior_weight=prior_weight,
            prior_path=prior_path
        )
    else:
        # Use original loss setup
        if use_focal_loss:
            # Use focal loss with specified parameters
            if class_weights is not None:
                # Use class weights as alpha parameter for focal loss
                criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, temperature=ce_temperature)
            else:
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, temperature=ce_temperature)
        elif use_label_smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing_value, temperature=ce_temperature)
        else:
            # For standard CrossEntropyLoss, temperature scaling needs to be applied manually in forward pass
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else torch.nn.CrossEntropyLoss()
    
    # Setup component-wise loss functions for multi-task learning
    component_criterion = torch.nn.CrossEntropyLoss()
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(train_dataloader), 
        eta_min=1e-6
    )
    
    # Update wandb config with actual optimizer settings
    if use_wandb:
        # Only update if these values weren't already set in wandb config
        update_dict = {}
        if "learning_rate" not in wandb.config:
            update_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
        if "weight_decay" not in wandb.config:
            update_dict["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
        if "scheduler" not in wandb.config:
            update_dict["scheduler"] = scheduler.__class__.__name__
        
        if update_dict:
            wandb.config.update(update_dict)
    
    # Tracking variables
    best_f1 = 0.0  # F1 scores range from 0 to 1, so start with 0
    best_composite_score = 0.0  # Track best composite metric
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_cat_acc': [],
        'val_cat_acc': [],
        'val_cat_f1': [],
        'epoch_times': [],
        # Component-specific metrics for multi-task learning
        'train_speaker_acc': [],
        'val_speaker_acc': [],
        'val_speaker_f1': [],
        'train_main_cat_acc': [],
        'val_main_cat_acc': [],
        'val_main_cat_f1': [],
        'train_sub_cat_acc': [],
        'val_sub_cat_acc': [],
        'val_sub_cat_f1': [],
        'val_third_level_acc': [],
        'val_third_level_f1': [],
        'val_fourth_level_acc': [],
        'val_fourth_level_f1': []
    }

    # Initialize model tracker if model_key is provided
    model_tracker = ModelTracker() if model_key else None
    if model_tracker and model_key:
        print(f"\n=== Model Tracker Initialized for '{model_key}' ===")
        existing_best = model_tracker.get_best_model_info(model_key)
        if existing_best:
            print(f"Current best composite score: {existing_best['composite_score']:.4f}")
            print(f"  From experiment: {existing_best['experiment_name']}")
        else:
            print(f"No previous best model found for '{model_key}'")

    # Memory cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # Early stopping setup
    early_stop_counter = 0
    best_early_stop_metric = None
    early_stopped = False
    if use_early_stopping:
        print(f"\n=== Early Stopping Enabled ===")
        print(f"  Monitor: {early_stopping_monitor}")
        print(f"  Patience: {early_stopping_patience}")
        print(f"  Min delta: {early_stopping_min_delta}")

    print("\n=== Starting Hierarchical Model Training ===")
    start_time = time.time()
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    # Function to calculate multi-task loss
    def calculate_multi_task_loss(outputs, targets):
        """Calculate combined loss for multi-task learning"""
        # If outputs is not a dictionary, just use it directly for category prediction
        if not isinstance(outputs, dict):
            # Check if criterion is CombinedLoss (requires additional arguments)
            if isinstance(criterion, CombinedLoss):
                source_categories = targets.get("source_category", None)
                loss_result = criterion(outputs, targets["category_label"], source_categories, label_encoder)
                if isinstance(loss_result, tuple):
                    loss, loss_dict = loss_result
                    return loss, loss_dict
                else:
                    return loss_result, {"category_loss": loss_result.item()}
            else:
                return criterion(outputs, targets["category_label"]), {"category_loss": criterion(outputs, targets["category_label"]).item()}

        # Main category loss
        # Check if criterion is CombinedLoss (requires additional arguments)
        if isinstance(criterion, CombinedLoss):
            source_categories = targets.get("source_category", None)
            loss_result = criterion(outputs["category_logits"], targets["category_label"], source_categories, label_encoder)
            if isinstance(loss_result, tuple):
                category_loss, loss_dict = loss_result
                loss_components = loss_dict.copy()
            else:
                category_loss = loss_result
                loss_components = {"category_loss": category_loss.item()}
        else:
            category_loss = criterion(outputs["category_logits"], targets["category_label"])
            loss_components = {"category_loss": category_loss.item()}

        # Component-specific losses if available
        if "speaker_logits" in outputs and "target_speaker" in targets:
            speaker_loss = component_criterion(outputs["speaker_logits"], targets["target_speaker"])
            loss_components["speaker_loss"] = speaker_loss.item()
        else:
            speaker_loss = 0
            
        if "main_cat_logits" in outputs and "target_main_cat" in targets:
            main_cat_loss = component_criterion(outputs["main_cat_logits"], targets["target_main_cat"]) 
            loss_components["main_cat_loss"] = main_cat_loss.item()
        else:
            main_cat_loss = 0
            
        if "sub_cat_logits" in outputs and "target_sub_cat" in targets:
            sub_cat_loss = component_criterion(outputs["sub_cat_logits"], targets["target_sub_cat"])
            loss_components["sub_cat_loss"] = sub_cat_loss.item()
        else:
            sub_cat_loss = 0
            
        if "third_level_logits" in outputs and "target_third_level" in targets:
            third_level_loss = component_criterion(outputs["third_level_logits"], targets["target_third_level"])
            loss_components["third_level_loss"] = third_level_loss.item()
        else:
            third_level_loss = 0
            
        if "fourth_level_logits" in outputs and "target_fourth_level" in targets:
            fourth_level_loss = component_criterion(outputs["fourth_level_logits"], targets["target_fourth_level"])
            loss_components["fourth_level_loss"] = fourth_level_loss.item()
        else:
            fourth_level_loss = 0
        
        # Calculate composite loss (main task gets higher weight)
        total_loss = category_loss
        component_count = 1
        
        if speaker_loss != 0:
            total_loss += 0.2 * speaker_loss
            component_count += 0.2
            
        if main_cat_loss != 0:
            total_loss += 0.2 * main_cat_loss
            component_count += 0.2
            
        if sub_cat_loss != 0:
            total_loss += 0.2 * sub_cat_loss
            component_count += 0.2
            
        if third_level_loss != 0:
            total_loss += 0.2 * third_level_loss
            component_count += 0.2
            
        if fourth_level_loss != 0:
            total_loss += 0.2 * fourth_level_loss
            component_count += 0.2
        
        # Normalize by component count to keep gradient scale comparable
        normalized_loss = total_loss / component_count
        
        return normalized_loss, loss_components
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_cat_correct = 0
        train_speaker_correct = 0
        train_main_cat_correct = 0
        train_sub_cat_correct = 0
        train_examples = 0
        epoch_start = time.time()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)", leave=False)
        log_file.write(f"\n\n===== EPOCH {epoch+1}/{num_epochs} =====\n\n")
        scaler = GradScaler()
        
        batch_losses = []  # For tracking batch losses within an epoch
        component_losses = {
            "category_loss": [],
            "speaker_loss": [],
            "main_cat_loss": [],
            "sub_cat_loss": [],
            "third_level_loss": [],
            "fourth_level_loss": [],
            "primary_loss": [],  # For CombinedLoss
            "transition_loss": [],  # For CombinedLoss
            "total_loss": []  # For CombinedLoss
        }
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move tensors to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Check if batch has hierarchical data structure
            has_hierarchical = "hierarchical_input_ids" in batch
            
            optimizer.zero_grad()
            with autocast(device_type="cuda" if device.type == "cuda" else "mps"):
                # Check if this is a refactored model that needs legacy format
                is_refactored = hasattr(model, 'use_backward_compatibility')
                
                # Forward pass - different handling based on input type
                if has_hierarchical:
                    # Use hierarchical inputs
                    if is_refactored:
                        outputs = model(
                            hierarchical_input_ids=batch["hierarchical_input_ids"],
                            hierarchical_attention_mask=batch["hierarchical_attention_mask"],
                            speaker_ids=batch.get("speaker_ids", None),
                            main_cat_ids=batch.get("main_cat_ids", None),
                            sub_cat_ids=batch.get("sub_cat_ids", None),
                            third_level_ids=batch.get("third_level_ids", None),
                            fourth_level_ids=batch.get("fourth_level_ids", None),
                            return_legacy_format=True
                        )
                    else:
                        outputs = model(
                            hierarchical_input_ids=batch["hierarchical_input_ids"],
                            hierarchical_attention_mask=batch["hierarchical_attention_mask"],
                            speaker_ids=batch.get("speaker_ids", None),
                            main_cat_ids=batch.get("main_cat_ids", None),
                            sub_cat_ids=batch.get("sub_cat_ids", None),
                            third_level_ids=batch.get("third_level_ids", None),
                            fourth_level_ids=batch.get("fourth_level_ids", None)
                        )
                else:
                    # Fallback to standard inputs
                    if is_refactored:
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            return_legacy_format=True
                        )
                    else:
                        # Check if model needs source_categories (e.g., TransitionMatrixBaseline)
                        if hasattr(model, '__class__') and model.__class__.__name__ == 'TransitionMatrixBaseline':
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                source_categories=batch.get("source_category", None)
                            )
                        else:
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]
                            )
                
                # Calculate loss
                loss, loss_dict = calculate_multi_task_loss(outputs, batch)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Track losses
            batch_loss = loss.item()
            total_train_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Track component losses
            for loss_name, loss_value in loss_dict.items():
                if loss_name in component_losses:
                    component_losses[loss_name].append(loss_value)
            
            # Track accuracy metrics
            with torch.no_grad():
                # Get category predictions
                if isinstance(outputs, dict):
                    cat_logits = outputs["category_logits"]
                else:
                    cat_logits = outputs
                
                cat_preds = torch.argmax(cat_logits, dim=1)
                train_cat_correct += (cat_preds == batch["category_label"]).sum().item()
                
                # Track component predictions if available
                if isinstance(outputs, dict) and "speaker_logits" in outputs and "target_speaker" in batch:
                    speaker_preds = torch.argmax(outputs["speaker_logits"], dim=1)
                    train_speaker_correct += (speaker_preds == batch["target_speaker"]).sum().item()
                
                if isinstance(outputs, dict) and "main_cat_logits" in outputs and "target_main_cat" in batch:
                    main_cat_preds = torch.argmax(outputs["main_cat_logits"], dim=1)
                    train_main_cat_correct += (main_cat_preds == batch["target_main_cat"]).sum().item()
                
                if isinstance(outputs, dict) and "sub_cat_logits" in outputs and "target_sub_cat" in batch:
                    sub_cat_preds = torch.argmax(outputs["sub_cat_logits"], dim=1)
                    train_sub_cat_correct += (sub_cat_preds == batch["target_sub_cat"]).sum().item()
                
                train_examples += batch["category_label"].size(0)
            
            # Update progress bar
            progress_bar.set_postfix(loss=batch_loss)
            
            # Log batch metrics to wandb
            if use_wandb and batch_idx % 10 == 0:
                wandb_batch_metrics = {
                    "batch": batch_idx + epoch * len(train_dataloader),
                    "batch_loss": batch_loss,
                    "batch_category_accuracy": (cat_preds == batch["category_label"]).sum().item() / batch["category_label"].size(0),
                    "learning_rate": scheduler.get_last_lr()[0]
                }
                
                # Add component-specific metrics if available
                for loss_name, loss_value in loss_dict.items():
                    wandb_batch_metrics[f"batch_{loss_name}"] = loss_value
                
                wandb.log(wandb_batch_metrics)
            
            # Log sample predictions to file
            if batch_idx % 10 == 0 or batch_idx < 5:
                log_file.write(f"Batch {batch_idx}/{len(train_dataloader)}\nLoss: {batch_loss:.4f}\n\n")
                
                # Decode and log a few examples
                for i in range(min(3, batch["category_label"].size(0))):
                    # Log input
                    if has_hierarchical:
                        # For hierarchical input, decode the last utterance
                        last_utterance_ids = batch["hierarchical_input_ids"][i, -1].cpu().numpy()
                        decoded_input = tokenizer.decode(last_utterance_ids, skip_special_tokens=False)
                    else:
                        # For flat input, decode the full sequence
                        decoded_input = tokenizer.decode(batch["input_ids"][i].cpu().numpy(), skip_special_tokens=False)
                    
                    # Get raw history
                    raw_history = batch.get("raw_history", ["N/A"])[i]
                    
                    # Log predictions and ground truth
                    pred_idx = cat_preds[i].item()
                    true_idx = batch["category_label"][i].item()
                    
                    if label_encoder is not None:
                        pred_name = label_encoder.inverse_transform([pred_idx])[0]
                        true_name = label_encoder.inverse_transform([true_idx])[0]
                    else:
                        pred_name = f"Category {pred_idx}"
                        true_name = f"Category {true_idx}"
                    
                    confidence = torch.softmax(cat_logits[i], dim=0)[pred_idx].item()
                    
                    log_file.write(f"Example {i+1}:\nTruncated input: {decoded_input}...\n")
                    log_file.write(f"Raw history: {raw_history}...\n")
                    log_file.write(f"Predicted: {pred_name} ({pred_idx}), True: {true_name} ({true_idx})\n")
                    log_file.write(f"Confidence: {confidence:.4f}\n\n")
                    
                    # Log component predictions if available
                    if isinstance(outputs, dict):
                        log_file.write("Component predictions:\n")
                        
                        if "speaker_logits" in outputs and "target_speaker" in batch:
                            speaker_pred = torch.argmax(outputs["speaker_logits"][i]).item()
                            speaker_true = batch["target_speaker"][i].item()
                            log_file.write(f"  Speaker: Predicted {speaker_pred}, True {speaker_true}\n")
                        
                        if "main_cat_logits" in outputs and "target_main_cat" in batch:
                            main_cat_pred = torch.argmax(outputs["main_cat_logits"][i]).item()
                            main_cat_true = batch["target_main_cat"][i].item()
                            log_file.write(f"  Main Category: Predicted {main_cat_pred}, True {main_cat_true}\n")
                        
                        if "sub_cat_logits" in outputs and "target_sub_cat" in batch:
                            sub_cat_pred = torch.argmax(outputs["sub_cat_logits"][i]).item()
                            sub_cat_true = batch["target_sub_cat"][i].item()
                            log_file.write(f"  Sub Category: Predicted {sub_cat_pred}, True {sub_cat_true}\n")
                        
                        if "third_level_logits" in outputs and "target_third_level" in batch:
                            third_level_pred = torch.argmax(outputs["third_level_logits"][i]).item()
                            third_level_true = batch["target_third_level"][i].item()
                            log_file.write(f"  Third Level: Predicted {third_level_pred}, True {third_level_true}\n")
                        
                        if "fourth_level_logits" in outputs and "target_fourth_level" in batch:
                            fourth_level_pred = torch.argmax(outputs["fourth_level_logits"][i]).item()
                            fourth_level_true = batch["target_fourth_level"][i].item()
                            log_file.write(f"  Fourth Level: Predicted {fourth_level_pred}, True {fourth_level_true}\n")
                        
                        log_file.write("\n")
                
                log_file.write("--------------------------------------------------\n\n")
                log_file.flush()
        
        # Calculate training metrics
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_category_accuracy = train_cat_correct / train_examples
        train_speaker_accuracy = train_speaker_correct / train_examples if train_speaker_correct > 0 else 0
        train_main_cat_accuracy = train_main_cat_correct / train_examples if train_main_cat_correct > 0 else 0
        train_sub_cat_accuracy = train_sub_cat_correct / train_examples if train_sub_cat_correct > 0 else 0
        
        # Calculate average component losses
        avg_component_losses = {name: np.mean(values) if values else 0 for name, values in component_losses.items()}
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_cat_correct = 0
        val_speaker_correct = 0
        val_main_cat_correct = 0
        val_sub_cat_correct = 0
        val_third_level_correct = 0
        val_fourth_level_correct = 0
        val_examples = 0
        all_cat_preds, all_cat_true = [], []
        all_logits_list = []
        all_source_categories = []  # For transition matrix metrics
        
        # Component-specific tracking for F1 scores
        all_speaker_preds, all_speaker_true = [], []
        all_main_cat_preds, all_main_cat_true = [], []
        all_sub_cat_preds, all_sub_cat_true = [], []
        all_third_level_preds, all_third_level_true = [], []
        all_fourth_level_preds, all_fourth_level_true = [], []
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training completed. Avg Loss: {avg_train_loss:.4f} | Category Accuracy: {train_category_accuracy:.4f}")
        log_file.write(f"\nTraining Summary for Epoch {epoch+1}:\n")
        log_file.write(f"Avg Loss: {avg_train_loss:.4f}\n")
        log_file.write(f"Category Accuracy: {train_category_accuracy:.4f}\n")
        
        # Log component-specific metrics
        if train_speaker_accuracy > 0:
            log_file.write(f"Speaker Accuracy: {train_speaker_accuracy:.4f}\n")
        if train_main_cat_accuracy > 0:
            log_file.write(f"Main Category Accuracy: {train_main_cat_accuracy:.4f}\n")
        if train_sub_cat_accuracy > 0:
            log_file.write(f"Sub Category Accuracy: {train_sub_cat_accuracy:.4f}\n")
        
        log_file.write("\n=== VALIDATION ===\n\n")
        
        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move tensors to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Check if batch has hierarchical data structure
                has_hierarchical = "hierarchical_input_ids" in batch
                
                # Check if this is a refactored model that needs legacy format
                is_refactored = hasattr(model, 'use_backward_compatibility')
                
                # Forward pass
                if has_hierarchical:
                    if is_refactored:
                        outputs = model(
                            hierarchical_input_ids=batch["hierarchical_input_ids"],
                            hierarchical_attention_mask=batch["hierarchical_attention_mask"],
                            speaker_ids=batch.get("speaker_ids", None),
                            main_cat_ids=batch.get("main_cat_ids", None),
                            sub_cat_ids=batch.get("sub_cat_ids", None),
                            third_level_ids=batch.get("third_level_ids", None),
                            fourth_level_ids=batch.get("fourth_level_ids", None),
                            return_legacy_format=True
                        )
                    else:
                        outputs = model(
                            hierarchical_input_ids=batch["hierarchical_input_ids"],
                            hierarchical_attention_mask=batch["hierarchical_attention_mask"],
                            speaker_ids=batch.get("speaker_ids", None),
                            main_cat_ids=batch.get("main_cat_ids", None),
                            sub_cat_ids=batch.get("sub_cat_ids", None),
                            third_level_ids=batch.get("third_level_ids", None),
                            fourth_level_ids=batch.get("fourth_level_ids", None)
                        )
                else:
                    if is_refactored:
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            return_legacy_format=True
                        )
                    else:
                        # Check if model needs source_categories (e.g., TransitionMatrixBaseline)
                        if hasattr(model, '__class__') and model.__class__.__name__ == 'TransitionMatrixBaseline':
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                source_categories=batch.get("source_category", None)
                            )
                        else:
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]
                            )

                # Calculate loss
                loss, _ = calculate_multi_task_loss(outputs, batch)
                total_val_loss += loss.item()
                
                # Get category predictions
                if isinstance(outputs, dict):
                    cat_logits = outputs["category_logits"]
                else:
                    cat_logits = outputs
                
                cat_preds = torch.argmax(cat_logits, dim=1)
                
                # Track metrics
                val_cat_correct += (cat_preds == batch["category_label"]).sum().item()
                val_examples += batch["category_label"].size(0)
                
                # Track component predictions if available
                if isinstance(outputs, dict) and "speaker_logits" in outputs and "target_speaker" in batch:
                    speaker_preds = torch.argmax(outputs["speaker_logits"], dim=1)
                    val_speaker_correct += (speaker_preds == batch["target_speaker"]).sum().item()
                    # Collect for F1 score calculation
                    all_speaker_preds.extend(speaker_preds.cpu().numpy())
                    all_speaker_true.extend(batch["target_speaker"].cpu().numpy())
                
                if isinstance(outputs, dict) and "main_cat_logits" in outputs and "target_main_cat" in batch:
                    main_cat_preds = torch.argmax(outputs["main_cat_logits"], dim=1)
                    val_main_cat_correct += (main_cat_preds == batch["target_main_cat"]).sum().item()
                    # Collect for F1 score calculation
                    all_main_cat_preds.extend(main_cat_preds.cpu().numpy())
                    all_main_cat_true.extend(batch["target_main_cat"].cpu().numpy())
                
                if isinstance(outputs, dict) and "sub_cat_logits" in outputs and "target_sub_cat" in batch:
                    sub_cat_preds = torch.argmax(outputs["sub_cat_logits"], dim=1)
                    val_sub_cat_correct += (sub_cat_preds == batch["target_sub_cat"]).sum().item()
                    # Collect for F1 score calculation
                    all_sub_cat_preds.extend(sub_cat_preds.cpu().numpy())
                    all_sub_cat_true.extend(batch["target_sub_cat"].cpu().numpy())
                
                if isinstance(outputs, dict) and "third_level_logits" in outputs and "target_third_level" in batch:
                    third_level_preds = torch.argmax(outputs["third_level_logits"], dim=1)
                    val_third_level_correct += (third_level_preds == batch["target_third_level"]).sum().item()
                    # Collect for F1 score calculation
                    all_third_level_preds.extend(third_level_preds.cpu().numpy())
                    all_third_level_true.extend(batch["target_third_level"].cpu().numpy())
                
                if isinstance(outputs, dict) and "fourth_level_logits" in outputs and "target_fourth_level" in batch:
                    fourth_level_preds = torch.argmax(outputs["fourth_level_logits"], dim=1)
                    val_fourth_level_correct += (fourth_level_preds == batch["target_fourth_level"]).sum().item()
                    # Collect for F1 score calculation
                    all_fourth_level_preds.extend(fourth_level_preds.cpu().numpy())
                    all_fourth_level_true.extend(batch["target_fourth_level"].cpu().numpy())
                
                # Collect predictions and targets for metrics
                all_cat_preds.extend(cat_preds.cpu().numpy())
                all_cat_true.extend(batch["category_label"].cpu().numpy())
                all_logits_list.append(cat_logits.cpu())
                
                # Collect source categories for transition matrix metrics
                if transition_matrix is not None and "source_category" in batch:
                    all_source_categories.extend(batch["source_category"])
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), acc=val_cat_correct / val_examples)
                
                # Log a few examples
                if batch_idx < 3:
                    log_file.write(f"Validation Batch {batch_idx}:\nLoss: {loss.item():.4f}\n\n")
                    
                    for i in range(min(3, batch["category_label"].size(0))):
                        pred_idx = cat_preds[i].item()
                        true_idx = batch["category_label"][i].item()
                        
                        if label_encoder is not None:
                            pred_name = label_encoder.inverse_transform([pred_idx])[0]
                            true_name = label_encoder.inverse_transform([true_idx])[0]
                        else:
                            pred_name = f"Category {pred_idx}"
                            true_name = f"Category {true_idx}"
                        
                        confidence = torch.softmax(cat_logits[i], dim=0)[pred_idx].item()
                        log_file.write(f"Example {i+1}: Predicted: {pred_name} ({pred_idx}), True: {true_name} ({true_idx})\n")
                        log_file.write(f"Confidence: {confidence:.4f}\n\n")
                        
                        # Log component predictions if available
                        if isinstance(outputs, dict):
                            log_file.write("Component predictions:\n")
                            
                            if "speaker_logits" in outputs and "target_speaker" in batch:
                                speaker_pred = torch.argmax(outputs["speaker_logits"][i]).item()
                                speaker_true = batch["target_speaker"][i].item()
                                log_file.write(f"  Speaker: Predicted {speaker_pred}, True {speaker_true}\n")
                            
                            if "main_cat_logits" in outputs and "target_main_cat" in batch:
                                main_cat_pred = torch.argmax(outputs["main_cat_logits"][i]).item()
                                main_cat_true = batch["target_main_cat"][i].item()
                                log_file.write(f"  Main Category: Predicted {main_cat_pred}, True {main_cat_true}\n")
                            
                            if "sub_cat_logits" in outputs and "target_sub_cat" in batch:
                                sub_cat_pred = torch.argmax(outputs["sub_cat_logits"][i]).item()
                                sub_cat_true = batch["target_sub_cat"][i].item()
                                log_file.write(f"  Sub Category: Predicted {sub_cat_pred}, True {sub_cat_true}\n")
                            
                            if "third_level_logits" in outputs and "target_third_level" in batch:
                                third_level_pred = torch.argmax(outputs["third_level_logits"][i]).item()
                                third_level_true = batch["target_third_level"][i].item()
                                log_file.write(f"  Third Level: Predicted {third_level_pred}, True {third_level_true}\n")
                            
                            if "fourth_level_logits" in outputs and "target_fourth_level" in batch:
                                fourth_level_pred = torch.argmax(outputs["fourth_level_logits"][i]).item()
                                fourth_level_true = batch["target_fourth_level"][i].item()
                                log_file.write(f"  Fourth Level: Predicted {fourth_level_pred}, True {fourth_level_true}\n")
                            
                            log_file.write("\n")
        
        # Calculate validation metrics
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_category_accuracy = val_cat_correct / val_examples
        val_speaker_accuracy = val_speaker_correct / val_examples if val_speaker_correct > 0 else 0
        val_main_cat_accuracy = val_main_cat_correct / val_examples if val_main_cat_correct > 0 else 0
        val_sub_cat_accuracy = val_sub_cat_correct / val_examples if val_sub_cat_correct > 0 else 0
        val_third_level_accuracy = val_third_level_correct / val_examples if val_third_level_correct > 0 else 0
        val_fourth_level_accuracy = val_fourth_level_correct / val_examples if val_fourth_level_correct > 0 else 0
        
        # Calculate F1 score for main category
        cat_f1 = f1_score(all_cat_true, all_cat_preds, average='weighted')
        
        # Calculate F1 scores for hierarchical components
        val_speaker_f1 = f1_score(all_speaker_true, all_speaker_preds, average='weighted') if len(all_speaker_true) > 0 else 0
        val_main_cat_f1 = f1_score(all_main_cat_true, all_main_cat_preds, average='weighted') if len(all_main_cat_true) > 0 else 0
        val_sub_cat_f1 = f1_score(all_sub_cat_true, all_sub_cat_preds, average='weighted') if len(all_sub_cat_true) > 0 else 0
        val_third_level_f1 = f1_score(all_third_level_true, all_third_level_preds, average='weighted') if len(all_third_level_true) > 0 else 0
        val_fourth_level_f1 = f1_score(all_fourth_level_true, all_fourth_level_preds, average='weighted') if len(all_fourth_level_true) > 0 else 0
        
        # Track epoch duration
        epoch_duration = time.time() - epoch_start
        
        # Compute aggregated metrics using all logits
        all_logits_tensor = torch.cat(all_logits_list, dim=0)
        
        # Top-k accuracy
        topk_acc_k2 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=2)
        topk_acc_k3 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=3)
        topk_acc_k4 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=4)
        topk_acc_k5 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=5)
        topk_acc_k6 = top_k_accuracy(all_logits_tensor, torch.tensor(all_cat_true), k=6)
        
        # Log top-k metrics
        print(f"Aggregated TOP-K Accuracy (k=2): {topk_acc_k2:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=3): {topk_acc_k3:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=4): {topk_acc_k4:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=5): {topk_acc_k5:.4f}")
        print(f"Aggregated TOP-K Accuracy (k=6): {topk_acc_k6:.4f}")
        
        log_file.write(f"Aggregated TOP-K Accuracy (k=2): {topk_acc_k2:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=3): {topk_acc_k3:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=4): {topk_acc_k4:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=5): {topk_acc_k5:.4f}\n")
        log_file.write(f"Aggregated TOP-K Accuracy (k=6): {topk_acc_k6:.4f}\n")
        
        # Transition matrix metrics (if available)
        transition_threshold_results = None
        if transition_matrix is not None and len(all_source_categories) == all_logits_tensor.size(0):
            # Filter transition matrix to match available categories
            filtered_transition_matrix = filter_transition_matrix(transition_matrix, label_encoder)
            
            # Calculate threshold metrics
            thresholds = [0.05, 0.1, 0.2]
            transition_threshold_results = {}
            
            for threshold in thresholds:
                results = evaluate_transition_threshold(
                    logits=all_logits_tensor,
                    targets=torch.tensor(all_cat_true),
                    source_categories=all_source_categories,
                    transition_matrix=filtered_transition_matrix,
                    label_encoder=label_encoder,
                    threshold=threshold
                )
                
                # Store results for this threshold
                transition_threshold_results[threshold] = results
                
                # Log key metrics
                log_file.write(f"\nTransition Threshold Metrics (threshold={threshold}):\n")
                log_file.write(f"  Valid source categories: {results.get('valid_source_count', 0)}/{len(all_cat_true)}\n")
                log_file.write(f"  Predictions above threshold: {results.get('predictions_above_threshold_percentage', 0):.4f}\n")
                log_file.write(f"  Correct & above threshold: {results.get('correct_and_above_threshold', 0):.4f}\n")
                log_file.write(f"  Top-1 in valid transitions: {results.get('top1_in_valid_transitions', 0):.4f}\n")
                log_file.write(f"  Top-3 in valid transitions: {results.get('top3_in_valid_transitions', 0):.4f}\n")
                
                if 'transition_aware_accuracy' in results:
                    log_file.write(f"  Transition-aware accuracy: {results['transition_aware_accuracy']:.4f}\n")
                
                print(f"Transition Threshold Metrics (t={threshold}): "
                     f"Correct & above threshold: {results.get('correct_and_above_threshold', 0):.4f}, "
                     f"Top-3 in valid: {results.get('top3_in_valid_transitions', 0):.4f}")

            # Run cumulative mass evaluation (adaptive approach)
            cumulative_results = evaluate_cumulative_mass_metrics(
                logits=all_logits_tensor,
                targets=torch.tensor(all_cat_true),
                source_categories=all_source_categories,
                transition_matrix=filtered_transition_matrix,
                label_encoder=label_encoder,
                cumulative_thresholds=[0.7, 0.8, 0.9]
            )

            # Log cumulative mass metrics
            log_file.write(f"\nCumulative Mass Metrics (Adaptive):\n")
            log_file.write(f"  Valid source categories: {cumulative_results.get('valid_source_count_cumulative', 0)}/{len(all_cat_true)}\n")
            for thresh_pct in [70, 80, 90]:
                pred_acc = cumulative_results.get(f'cumulative{thresh_pct}_pred_accuracy', 0)
                true_cov = cumulative_results.get(f'cumulative{thresh_pct}_true_coverage', 0)
                pred_acc_plus = cumulative_results.get(f'cumulative{thresh_pct}_pred_accuracy_plus_true', 0)

                log_file.write(f"  Cumulative {thresh_pct}%:\n")
                log_file.write(f"    Pred in valid: {pred_acc:.4f}\n")
                log_file.write(f"    True coverage: {true_cov:.4f}\n")
                log_file.write(f"    Pred in valid+true: {pred_acc_plus:.4f}\n")

            print(f"Cumulative Mass Metrics: "
                  f"80% pred_accuracy={cumulative_results.get('cumulative80_pred_accuracy', 0):.4f}, "
                  f"true_coverage={cumulative_results.get('cumulative80_true_coverage', 0):.4f}")

            # Log to W&B if enabled
            if use_wandb and wandb.run is not None:
                wandb_cumulative = {}
                for key, value in cumulative_results.items():
                    if key != 'valid_source_count_cumulative':
                        wandb_cumulative[f"val_{key}"] = value
                wandb.log(wandb_cumulative)

            # Calculate entropy and JS divergence
            mean_entropy, mean_js_div = calculate_entropy_and_js_with_transitions(
                pred_logits=all_logits_tensor,
                source_categories=all_source_categories,
                transition_matrix=filtered_transition_matrix,
                label_encoder=label_encoder
            )
            log_file.write(f"Transition Matrix Metrics:\nMean Entropy: {mean_entropy:.4f}\nMean JS Divergence: {mean_js_div:.4f}\n\n")
            print(f"Transition Matrix Metrics: Mean Entropy: {mean_entropy:.4f}, Mean JS Divergence: {mean_js_div:.4f}")
        else:
            # Calculate basic entropy and JS divergence
            mean_entropy, mean_js_div = calculate_entropy_and_js(all_logits_tensor)
            log_file.write(f"Baseline Metrics:\nMean Entropy: {mean_entropy:.4f}\nMean JS Divergence: {mean_js_div:.4f}\n\n")
        
        # Update training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['train_cat_acc'].append(train_category_accuracy)
        training_history['val_cat_acc'].append(val_category_accuracy)
        training_history['val_cat_f1'].append(cat_f1)
        training_history['epoch_times'].append(epoch_duration)
        
        # Always append all metrics to maintain consistent array lengths
        # Use actual values if available, otherwise use 0.0 as placeholder
        training_history['train_speaker_acc'].append(train_speaker_accuracy if train_speaker_accuracy > 0 else 0.0)
        training_history['val_speaker_acc'].append(val_speaker_accuracy if train_speaker_accuracy > 0 else 0.0)
        training_history['val_speaker_f1'].append(val_speaker_f1 if train_speaker_accuracy > 0 else 0.0)
        
        training_history['train_main_cat_acc'].append(train_main_cat_accuracy if train_main_cat_accuracy > 0 else 0.0)
        training_history['val_main_cat_acc'].append(val_main_cat_accuracy if train_main_cat_accuracy > 0 else 0.0)
        training_history['val_main_cat_f1'].append(val_main_cat_f1 if train_main_cat_accuracy > 0 else 0.0)
        
        training_history['train_sub_cat_acc'].append(train_sub_cat_accuracy if train_sub_cat_accuracy > 0 else 0.0)
        training_history['val_sub_cat_acc'].append(val_sub_cat_accuracy if train_sub_cat_accuracy > 0 else 0.0)
        training_history['val_sub_cat_f1'].append(val_sub_cat_f1 if train_sub_cat_accuracy > 0 else 0.0)
        
        training_history['val_third_level_acc'].append(val_third_level_accuracy if val_third_level_accuracy > 0 else 0.0)
        training_history['val_third_level_f1'].append(val_third_level_f1 if val_third_level_accuracy > 0 else 0.0)
        
        training_history['val_fourth_level_acc'].append(val_fourth_level_accuracy if val_fourth_level_accuracy > 0 else 0.0)
        training_history['val_fourth_level_f1'].append(val_fourth_level_f1 if val_fourth_level_accuracy > 0 else 0.0)
        
        # Log metrics to W&B
        if use_wandb:
            metrics_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_category_accuracy,
                "val_accuracy": val_category_accuracy,
                "val_f1": cat_f1,
                "epoch_time": epoch_duration,
                "top_k_acc_2": topk_acc_k2,
                "top_k_acc_3": topk_acc_k3,
                "top_k_acc_4": topk_acc_k4,
                "top_k_acc_5": topk_acc_k5,
                "top_k_acc_6": topk_acc_k6,
                "mean_entropy": mean_entropy,
                "mean_js_divergence": mean_js_div,
                "learning_rate": scheduler.get_last_lr()[0]
            }
            
            # Add component-specific metrics
            if train_speaker_accuracy > 0:
                metrics_dict.update({
                    "train_speaker_accuracy": train_speaker_accuracy,
                    "val_speaker_accuracy": val_speaker_accuracy,
                    "val_speaker_f1": val_speaker_f1
                })
            
            if train_main_cat_accuracy > 0:
                metrics_dict.update({
                    "train_main_cat_accuracy": train_main_cat_accuracy,
                    "val_main_cat_accuracy": val_main_cat_accuracy,
                    "val_main_cat_f1": val_main_cat_f1
                })
            
            if train_sub_cat_accuracy > 0:
                metrics_dict.update({
                    "train_sub_cat_accuracy": train_sub_cat_accuracy,
                    "val_sub_cat_accuracy": val_sub_cat_accuracy,
                    "val_sub_cat_f1": val_sub_cat_f1
                })
            
            if val_third_level_accuracy > 0:
                metrics_dict.update({
                    "val_third_level_accuracy": val_third_level_accuracy,
                    "val_third_level_f1": val_third_level_f1
                })
            
            if val_fourth_level_accuracy > 0:
                metrics_dict.update({
                    "val_fourth_level_accuracy": val_fourth_level_accuracy,
                    "val_fourth_level_f1": val_fourth_level_f1
                })
            
            # Add component loss metrics
            for name, value in avg_component_losses.items():
                if value > 0:
                    metrics_dict[f"train_{name}"] = value
            
            # Log metrics
            wandb.log(metrics_dict)
            
            # Log transition metrics if available
            if transition_threshold_results:
                for threshold, results in transition_threshold_results.items():
                    threshold_metrics = {
                        f"transition_t{threshold}_correct_above": results.get('correct_and_above_threshold', 0),
                        f"transition_t{threshold}_top1_in_valid": results.get('top1_in_valid_transitions', 0),
                        f"transition_t{threshold}_top3_in_valid": results.get('top3_in_valid_transitions', 0),
                    }
                    if 'transition_aware_accuracy' in results:
                        threshold_metrics[f"transition_t{threshold}_aware_accuracy"] = results['transition_aware_accuracy']
                    
                    wandb.log(threshold_metrics)
            
            # Create a histogram of batch losses
            if len(batch_losses) > 0:
                wandb.log({"batch_loss_distribution": wandb.Histogram(batch_losses)})
        
        # Print validation summary
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Validation completed.")
        print(f"Loss: {avg_val_loss:.4f} | Acc: {val_category_accuracy:.4f} | F1: {cat_f1:.4f} | Epoch time: {epoch_duration:.2f}s")
        
        if val_speaker_accuracy > 0:
            print(f"Speaker Acc: {val_speaker_accuracy:.4f} | Speaker F1: {val_speaker_f1:.4f}")
        if val_main_cat_accuracy > 0:
            print(f"Main Cat Acc: {val_main_cat_accuracy:.4f} | Main Cat F1: {val_main_cat_f1:.4f}")
        if val_sub_cat_accuracy > 0:
            print(f"Sub Cat Acc: {val_sub_cat_accuracy:.4f} | Sub Cat F1: {val_sub_cat_f1:.4f}")
        if val_third_level_accuracy > 0:
            print(f"Third Level Acc: {val_third_level_accuracy:.4f} | Third Level F1: {val_third_level_f1:.4f}")
        if val_fourth_level_accuracy > 0:
            print(f"Fourth Level Acc: {val_fourth_level_accuracy:.4f} | Fourth Level F1: {val_fourth_level_f1:.4f}")
        
        # Log validation summary
        log_file.write(f"\nValidation Summary for Epoch {epoch+1}:\n")
        log_file.write(f"Avg Loss: {avg_val_loss:.4f}\n")
        log_file.write(f"Category Accuracy: {val_category_accuracy:.4f}\n")
        log_file.write(f"F1 Score: {cat_f1:.4f}\n")
        
        if val_speaker_accuracy > 0:
            log_file.write(f"Speaker Accuracy: {val_speaker_accuracy:.4f}\n")
            log_file.write(f"Speaker F1 Score: {val_speaker_f1:.4f}\n")
        if val_main_cat_accuracy > 0:
            log_file.write(f"Main Category Accuracy: {val_main_cat_accuracy:.4f}\n")
            log_file.write(f"Main Category F1 Score: {val_main_cat_f1:.4f}\n")
        if val_sub_cat_accuracy > 0:
            log_file.write(f"Sub Category Accuracy: {val_sub_cat_accuracy:.4f}\n")
            log_file.write(f"Sub Category F1 Score: {val_sub_cat_f1:.4f}\n")
        if val_third_level_accuracy > 0:
            log_file.write(f"Third Level Accuracy: {val_third_level_accuracy:.4f}\n")
            log_file.write(f"Third Level F1 Score: {val_third_level_f1:.4f}\n")
        if val_fourth_level_accuracy > 0:
            log_file.write(f"Fourth Level Accuracy: {val_fourth_level_accuracy:.4f}\n")
            log_file.write(f"Fourth Level F1 Score: {val_fourth_level_f1:.4f}\n")
        
        log_file.write(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n\n")
        
        # Generate confusion matrix for main category
        if label_encoder:
            try:
                unique_true = np.unique(all_cat_true)
                unique_classes = np.unique(np.concatenate([unique_true, np.unique(all_cat_preds)]))
                category_names = label_encoder.inverse_transform(unique_classes)
                cm = confusion_matrix(all_cat_true, all_cat_preds, labels=unique_classes)
                conf_matrix = pd.DataFrame(cm, index=category_names, columns=category_names)
                log_file.write("Main Category Confusion Matrix:\n" + conf_matrix.to_string() + "\n\n")
                log_file.write(f"Note: {len(unique_classes)} out of {len(label_encoder.classes_)} categories in validation run\n\n")
                
                # Log confusion matrix to W&B
                if use_wandb:
                    # Create mapping from category IDs to indices for wandb confusion matrix
                    id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_classes)}
                    
                    # Map the true labels and predictions to indices
                    y_true_indexed = [id_to_idx[cat_id] for cat_id in all_cat_true]
                    preds_indexed = [id_to_idx[cat_id] for cat_id in all_cat_preds]
                    
                    confusion_matrix_wandb = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true_indexed,
                        preds=preds_indexed,
                        class_names=category_names.tolist()
                    )
                    wandb.log({"confusion_matrix_main_category": confusion_matrix_wandb})
            except Exception as e:
                log_file.write(f"Error generating main category confusion matrix: {str(e)}\n\n")

        # Generate confusion matrices for hierarchical components
        if hierarchy_info and use_wandb:
            # Helper function to generate confusion matrix for hierarchical components
            def generate_component_confusion_matrix(component_name, all_true, all_preds, id_to_name_mapping):
                try:
                    if len(all_true) > 0 and len(all_preds) > 0:
                        unique_true = np.unique(all_true)
                        unique_classes = np.unique(np.concatenate([unique_true, np.unique(all_preds)]))
                        
                        # Get component names using the hierarchy mapping
                        component_names = []
                        for class_id in unique_classes:
                            if class_id in id_to_name_mapping:
                                component_names.append(id_to_name_mapping[class_id])
                            else:
                                component_names.append(f"Unknown_{class_id}")
                        
                        # Generate confusion matrix
                        cm = confusion_matrix(all_true, all_preds, labels=unique_classes)
                        conf_matrix = pd.DataFrame(cm, index=component_names, columns=component_names)
                        
                        # Log to file
                        log_file.write(f"{component_name} Confusion Matrix:\n" + conf_matrix.to_string() + "\n\n")
                        log_file.write(f"Note: {len(unique_classes)} {component_name.lower()} classes in validation run\n\n")
                        
                        # Log to W&B
                        # Create mapping from category IDs to indices for wandb confusion matrix
                        id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_classes)}
                        
                        # Map the true labels and predictions to indices
                        y_true_indexed = [id_to_idx[cat_id] for cat_id in all_true]
                        preds_indexed = [id_to_idx[cat_id] for cat_id in all_preds]
                        
                        confusion_matrix_wandb = wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=y_true_indexed,
                            preds=preds_indexed,
                            class_names=component_names
                        )
                        wandb.log({f"confusion_matrix_{component_name.lower().replace(' ', '_')}": confusion_matrix_wandb})
                        
                except Exception as e:
                    log_file.write(f"Error generating {component_name} confusion matrix: {str(e)}\n\n")

            # Generate confusion matrices for each hierarchical component
            if len(all_speaker_true) > 0:
                generate_component_confusion_matrix(
                    "Speaker", all_speaker_true, all_speaker_preds, 
                    hierarchy_info.get("id_to_speaker", {})
                )
            
            if len(all_main_cat_true) > 0:
                generate_component_confusion_matrix(
                    "Main Category", all_main_cat_true, all_main_cat_preds, 
                    hierarchy_info.get("id_to_main", {})
                )
            
            if len(all_sub_cat_true) > 0:
                generate_component_confusion_matrix(
                    "Sub Category", all_sub_cat_true, all_sub_cat_preds, 
                    hierarchy_info.get("id_to_sub", {})
                )
            
            if len(all_third_level_true) > 0:
                generate_component_confusion_matrix(
                    "Third Level", all_third_level_true, all_third_level_preds, 
                    hierarchy_info.get("id_to_third", {})
                )
            
            if len(all_fourth_level_true) > 0:
                generate_component_confusion_matrix(
                    "Fourth Level", all_fourth_level_true, all_fourth_level_preds, 
                    hierarchy_info.get("id_to_fourth", {})
                )
            
            # Generate classification report
            try:
                unique_classes = np.unique(all_cat_true)
                category_names_present = label_encoder.inverse_transform(unique_classes)
                report = classification_report(
                    all_cat_true, all_cat_preds,
                    labels=unique_classes, target_names=category_names_present,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                log_file.write("Classification Report:\n" + report_df.to_string() + "\n\n")
                
                # Log per-class metrics to W&B
                if use_wandb:
                    for cls_name, metrics in report.items():
                        if isinstance(metrics, dict):
                            wandb.log({
                                f"class_{cls_name}_precision": metrics["precision"],
                                f"class_{cls_name}_recall": metrics["recall"],
                                f"class_{cls_name}_f1": metrics["f1-score"],
                                f"class_{cls_name}_support": metrics["support"],
                            })
            except Exception as e:
                log_file.write(f"Error generating classification report: {str(e)}\n\n")

        # Calculate composite metric: F1 + top-k accuracy + transition validity
        # Weights: 0.5 for F1, 0.25 for top-k acc, 0.25 for transition validity
        transition_t005_top1 = 0.0
        if transition_threshold_results and 0.05 in transition_threshold_results:
            transition_t005_top1 = transition_threshold_results[0.05].get('top1_in_valid_transitions', 0)

        composite_score = 0.5 * cat_f1 + 0.25 * topk_acc_k3 + 0.25 * transition_t005_top1

        log_file.write(f"\nComposite Metric (epoch {epoch+1}):\n")
        log_file.write(f"  F1 Score: {cat_f1:.4f} (weight: 0.5)\n")
        log_file.write(f"  Top-3 Accuracy: {topk_acc_k3:.4f} (weight: 0.25)\n")
        log_file.write(f"  Transition T0.05 Top1: {transition_t005_top1:.4f} (weight: 0.25)\n")
        log_file.write(f"  Composite Score: {composite_score:.4f}\n\n")

        if use_wandb:
            wandb.log({
                "composite_score": composite_score,
                "composite_f1_component": cat_f1,
                "composite_topk3_component": topk_acc_k3,
                "composite_transition_component": transition_t005_top1
            })

        # Save model based on composite score (with model tracker check)
        if composite_score > best_composite_score:
            best_composite_score = composite_score

            # Check with model tracker if we should save
            should_save = True
            if model_tracker and model_key:
                should_save, prev_best = model_tracker.should_save_model(
                    model_key, composite_score, experiment_name
                )
                if not should_save:
                    print(f"  New best in this run! {composite_score:.4f} but NOT better than tracked best: {prev_best:.4f}")
                    log_file.write(f"*** New best in this run: {composite_score:.4f}, but tracked best is {prev_best:.4f}. Not saving. ***\n\n")
                else:
                    print(f"  New GLOBAL best Composite Score! {composite_score:.4f} (F1: {cat_f1:.4f}, Top-3: {topk_acc_k3:.4f}, Trans: {transition_t005_top1:.4f})")
                    if prev_best:
                        print(f"    Previous best: {prev_best:.4f}")
                    log_file.write(f"*** New GLOBAL best Composite Score! {composite_score:.4f} Saving model... ***\n\n")
            else:
                print(f"  New best Composite Score! {composite_score:.4f} (F1: {cat_f1:.4f}, Top-3: {topk_acc_k3:.4f}, Trans: {transition_t005_top1:.4f}) Saving model...")
                log_file.write(f"*** New best Composite Score! {composite_score:.4f} Saving model... ***\n\n")

            if should_save:
                save_model(model, model_save_path, device)

                # Update tracker
                if model_tracker and model_key:
                    model_tracker.update_best_model(
                        model_key=model_key,
                        composite_score=composite_score,
                        f1_score=cat_f1,
                        top_k_acc=topk_acc_k3,
                        transition_score=transition_t005_top1,
                        experiment_name=experiment_name or "unknown",
                        epoch=epoch + 1,
                        model_path=model_save_path
                    )

            # Save confusion matrix for best composite score model (only if model was saved)
            if should_save and label_encoder is not None:
                try:
                    # Use experiment_name or fallback to wandb_run_name for organized saving
                    exp_name = experiment_name or wandb_run_name or "default_hierarchical_experiment"
                    save_confusion_matrix_best_f1(
                        all_cat_true, all_cat_preds, label_encoder, composite_score,
                        f"confusion_matrix_best_composite_hierarchical_epoch_{epoch+1}.png",
                        experiment_name=exp_name
                    )
                except Exception as e:
                    print(f"Error saving confusion matrix: {e}")
                    log_file.write(f"Error saving confusion matrix: {e}\n\n")

                # Save transition alignment matrix if transition matrix is available
                if transition_matrix is not None and len(all_source_categories) > 0:
                    try:
                        print(f"Generating transition alignment matrix...")
                        save_transition_alignment_matrix(
                            y_true=all_cat_true,
                            y_pred=all_cat_preds,
                            source_categories=all_source_categories,
                            transition_matrix=transition_matrix,
                            label_encoder=label_encoder,
                            f1_score_val=composite_score,
                            threshold=0.05,  # Can be made configurable
                            save_path=f"transition_alignment_best_composite_hierarchical_epoch_{epoch+1}.png",
                            experiment_name=exp_name
                        )
                    except Exception as e:
                        print(f"Error saving transition alignment matrix: {e}")
                        log_file.write(f"Error saving transition alignment matrix: {e}\n\n")
            
            # Log best model to W&B
            if use_wandb:
                wandb.log({
                    "best_model_epoch": epoch + 1,
                    "best_model_f1": cat_f1,
                    "best_model_accuracy": val_category_accuracy
                })
                # Save the best model to W&B
                # wandb.save(model_save_path)  # Disabled to prevent model upload
        
        # Memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Early stopping check
        if use_early_stopping:
            # Get the metric to monitor
            if early_stopping_monitor == "val_loss":
                current_metric = avg_val_loss
                is_improvement = best_early_stop_metric is None or current_metric < (best_early_stop_metric - early_stopping_min_delta)
            elif early_stopping_monitor == "val_macro_f1":
                current_metric = cat_f1
                is_improvement = best_early_stop_metric is None or current_metric > (best_early_stop_metric + early_stopping_min_delta)
            elif early_stopping_monitor == "val_accuracy":
                current_metric = val_category_accuracy
                is_improvement = best_early_stop_metric is None or current_metric > (best_early_stop_metric + early_stopping_min_delta)
            else:
                # Default to val_macro_f1
                current_metric = cat_f1
                is_improvement = best_early_stop_metric is None or current_metric > (best_early_stop_metric + early_stopping_min_delta)

            if is_improvement:
                best_early_stop_metric = current_metric
                early_stop_counter = 0
                print(f"  Early stopping: {early_stopping_monitor} improved to {current_metric:.4f}")
            else:
                early_stop_counter += 1
                print(f"  Early stopping: {early_stopping_monitor} did not improve. Counter: {early_stop_counter}/{early_stopping_patience}")

            if early_stop_counter >= early_stopping_patience:
                print(f"\n=== Early Stopping Triggered at Epoch {epoch+1} ===")
                print(f"  No improvement in {early_stopping_monitor} for {early_stopping_patience} epochs")
                print(f"  Best {early_stopping_monitor}: {best_early_stop_metric:.4f}")
                log_file.write(f"\n*** Early stopping triggered at epoch {epoch+1}. Best {early_stopping_monitor}: {best_early_stop_metric:.4f} ***\n")
                early_stopped = True
                break

    # Training complete
    total_training_time = time.time() - start_time
    if early_stopped:
        print(f"\nTraining stopped early after {epoch+1} epochs. Total time: {total_training_time:.2f} seconds")
    else:
        print(f"\nTotal training time: {total_training_time:.2f} seconds")
    log_file.write(f"\nTotal training time: {total_training_time:.2f} seconds")
    log_file.close()
    
    # Finish W&B run
    if use_wandb:
        # Log final metrics and hyperparameters
        wandb.log({
            "total_training_time": total_training_time,
            "best_f1": best_f1
        })
        
        # Save training history to W&B
        history_df = pd.DataFrame(training_history)
        wandb.log({"training_history": wandb.Table(dataframe=history_df)})
        
        # Create a line plot of training and validation losses
        if len(training_history['train_loss']) > 0:
            epochs = list(range(1, num_epochs + 1))
            loss_table = wandb.Table(data=[[e, tl, vl] for e, tl, vl in 
                                          zip(epochs, 
                                              training_history['train_loss'], 
                                              training_history['val_loss'])],
                                    columns=["epoch", "train_loss", "val_loss"])
            wandb.log({"loss_plot": wandb.plot.line(
                loss_table, "epoch", ["train_loss", "val_loss"], 
                title="Training and Validation Loss")})
            
            # Create line plots for component-specific metrics if available
            if 'train_speaker_acc' in training_history and len(training_history['train_speaker_acc']) > 0:
                speaker_table = wandb.Table(data=[[e, ta, va, vf1] for e, ta, va, vf1 in 
                                              zip(epochs, 
                                                  training_history['train_speaker_acc'], 
                                                  training_history['val_speaker_acc'],
                                                  training_history['val_speaker_f1'])],
                                        columns=["epoch", "train_speaker_acc", "val_speaker_acc", "val_speaker_f1"])
                wandb.log({"speaker_acc_plot": wandb.plot.line(
                    speaker_table, "epoch", ["train_speaker_acc", "val_speaker_acc"], 
                    title="Speaker Prediction Accuracy")})
                wandb.log({"speaker_f1_plot": wandb.plot.line(
                    speaker_table, "epoch", ["val_speaker_f1"], 
                    title="Speaker Prediction F1 Score")})
            
            if 'train_main_cat_acc' in training_history and len(training_history['train_main_cat_acc']) > 0:
                main_cat_table = wandb.Table(data=[[e, ta, va, vf1] for e, ta, va, vf1 in 
                                                 zip(epochs, 
                                                     training_history['train_main_cat_acc'], 
                                                     training_history['val_main_cat_acc'],
                                                     training_history['val_main_cat_f1'])],
                                           columns=["epoch", "train_main_cat_acc", "val_main_cat_acc", "val_main_cat_f1"])
                wandb.log({"main_cat_acc_plot": wandb.plot.line(
                    main_cat_table, "epoch", ["train_main_cat_acc", "val_main_cat_acc"], 
                    title="Main Category Prediction Accuracy")})
                wandb.log({"main_cat_f1_plot": wandb.plot.line(
                    main_cat_table, "epoch", ["val_main_cat_f1"], 
                    title="Main Category Prediction F1 Score")})
            
            if 'train_sub_cat_acc' in training_history and len(training_history['train_sub_cat_acc']) > 0:
                sub_cat_table = wandb.Table(data=[[e, ta, va, vf1] for e, ta, va, vf1 in 
                                                 zip(epochs, 
                                                     training_history['train_sub_cat_acc'], 
                                                     training_history['val_sub_cat_acc'],
                                                     training_history['val_sub_cat_f1'])],
                                           columns=["epoch", "train_sub_cat_acc", "val_sub_cat_acc", "val_sub_cat_f1"])
                wandb.log({"sub_cat_acc_plot": wandb.plot.line(
                    sub_cat_table, "epoch", ["train_sub_cat_acc", "val_sub_cat_acc"], 
                    title="Sub Category Prediction Accuracy")})
                wandb.log({"sub_cat_f1_plot": wandb.plot.line(
                    sub_cat_table, "epoch", ["val_sub_cat_f1"], 
                    title="Sub Category Prediction F1 Score")})
            
            if 'val_third_level_acc' in training_history and len(training_history['val_third_level_acc']) > 0:
                third_level_table = wandb.Table(data=[[e, va, vf1] for e, va, vf1 in 
                                                     zip(epochs, 
                                                         training_history['val_third_level_acc'],
                                                         training_history['val_third_level_f1'])],
                                               columns=["epoch", "val_third_level_acc", "val_third_level_f1"])
                wandb.log({"third_level_acc_plot": wandb.plot.line(
                    third_level_table, "epoch", ["val_third_level_acc"], 
                    title="Third Level Prediction Accuracy")})
                wandb.log({"third_level_f1_plot": wandb.plot.line(
                    third_level_table, "epoch", ["val_third_level_f1"], 
                    title="Third Level Prediction F1 Score")})
            
            if 'val_fourth_level_acc' in training_history and len(training_history['val_fourth_level_acc']) > 0:
                fourth_level_table = wandb.Table(data=[[e, va, vf1] for e, va, vf1 in 
                                                      zip(epochs, 
                                                          training_history['val_fourth_level_acc'],
                                                          training_history['val_fourth_level_f1'])],
                                                columns=["epoch", "val_fourth_level_acc", "val_fourth_level_f1"])
                wandb.log({"fourth_level_acc_plot": wandb.plot.line(
                    fourth_level_table, "epoch", ["val_fourth_level_acc"], 
                    title="Fourth Level Prediction Accuracy")})
                wandb.log({"fourth_level_f1_plot": wandb.plot.line(
                    fourth_level_table, "epoch", ["val_fourth_level_f1"], 
                    title="Fourth Level Prediction F1 Score")})
        
        # Don't finish wandb here - let the main script handle wandb lifecycle
        # wandb.finish()
    
    return training_history
