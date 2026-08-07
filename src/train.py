"""
Training loop for CGR-based DNA sequence classifiers.

Features:
  - Binary and multi-class training modes
  - Adam optimizer with ReduceLROnPlateau scheduling
  - Early stopping on validation loss
  - TensorBoard logging
  - Stratified splits (70/15/15)

"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import build_dataloaders, CLASSES_BINARY, CLASSES_MULTICLASS
from models import build_model, build_loss_fn, save_checkpoint, ModelConfig


# Device setup

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("  Using CPU (training will be slow)")
    return device



# One epoch

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task: str,
    phase: str = "train",
) -> dict:
    """
    Run one training or validation epoch.

    Returns dict with: loss, accuracy, f1_macro
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=f"  {phase}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(imgs)

            if task == "binary":
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
            else:
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

            if is_train:
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "accuracy": float(accuracy), "f1_macro": float(f1)}


# Early stopping

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop



# Main training function

def train(config: ModelConfig, data_dir: str, output_dir: str):
    """Full training pipeline."""
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Data
    task = "binary" if config.n_classes == 2 else "multiclass"
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        task=task,
        batch_size=config.batch_size,
        num_workers=4,
    )

    # Model
    print("\nBuilding model...")
    model = build_model(
        config.model_name,
        n_classes=config.n_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
        freeze_backbone=config.freeze_backbone,
    )
    model = model.to(device)

    all_train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    from dataset import compute_class_weights
    cw = compute_class_weights(all_train_labels, config.n_classes).to(device)

    criterion = build_loss_fn(
        task=task,
        n_classes=config.n_classes,
        class_weights=cw,
        label_smoothing=config.label_smoothing,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        verbose=True,
    )
    early_stopper = EarlyStopping(patience=config.early_stop_patience)

    # CSV logger
    log_path = log_dir / f"{config.model_name}_{task}_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "train_f1",
                         "val_loss", "val_acc", "val_f1", "lr"])

    # Training loop
    best_val_f1 = 0.0
    best_ckpt_path = ckpt_dir / f"best_{task}_{config.model_name}.pt"
    history = {"train": [], "val": []}

    print(f"\n{'='*60}")
    print(f"Training {config.model_name} ({task})")
    print(f"  Classes: {class_names}")
    print(f"  Epochs:  {config.epochs}")
    print(f"  LR:      {config.lr}")
    print(f"  Device:  {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        if config.freeze_backbone and epoch == 6:
            print("  [Phase 2] Unfreezing full backbone...")
            model.unfreeze_all()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.lr * 0.1,
                weight_decay=config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=config.lr_factor,
                patience=config.lr_patience
            )

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, task, "train")
        val_metrics = run_epoch(model, val_loader, criterion, None, device, task, "val")

        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # CSV logging
        lr_now = optimizer.param_groups[0]["lr"]
        log_writer.writerow([
            epoch,
            f"{train_metrics['loss']:.4f}", f"{train_metrics['accuracy']:.4f}", f"{train_metrics['f1_macro']:.4f}",
            f"{val_metrics['loss']:.4f}", f"{val_metrics['accuracy']:.4f}", f"{val_metrics['f1_macro']:.4f}",
            f"{lr_now:.2e}"
        ])
        log_file.flush()

        print(
            f"Epoch {epoch:>3}/{config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} F1: {train_metrics['f1_macro']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} F1: {val_metrics['f1_macro']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt_path, config)
            print(f"  New best F1: {best_val_f1:.4f} -> {best_ckpt_path.name}")

        if early_stopper(val_metrics["loss"]):
            print(f"\n  Early stopping at epoch {epoch} (patience={config.early_stop_patience})")
            break

    log_file.close()

    history_path = output_dir / f"history_{task}_{config.model_name}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best val F1: {best_val_f1:.4f}")
    print(f"  Checkpoint:  {best_ckpt_path}")
    print(f"  Log:         {log_path}")
    print(f"  History:     {history_path}")

    return best_ckpt_path, history


# CLI

def main():
    parser = argparse.ArgumentParser(description="Train CGR DNA sequence classifier")
    parser.add_argument("--data_dir", required=True, help="CGR image directory")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone initially (2-phase training)")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from scratch (not recommended)")
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    n_classes = 2 if args.task == "binary" else 6

    config = ModelConfig(
        model_name=args.model,
        n_classes=n_classes,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    train(config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
