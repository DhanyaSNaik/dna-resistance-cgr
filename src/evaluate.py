"""
Model evaluation: accuracy, macro F1, per-class metrics, confusion matrix.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve
)
from tqdm import tqdm

from dataset import build_dataloaders, CLASSES_MULTICLASS, CLASSES_BINARY
from models import build_model, load_checkpoint


# Prediction collection

def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on entire dataloader and collect predictions.

    Returns:
        labels:  ground-truth labels (N,)
        preds:   predicted class indices (N,)
        probs:   softmax/sigmoid probabilities (N, n_classes) or (N,)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Evaluating"):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)

            if task == "binary":
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.flatten())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# Plotting utilities

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = True,
):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("True", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved -> {output_path}")


def plot_training_history(history_path: str, output_dir: Path):
    """Plot training and validation loss/accuracy/F1 over epochs."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train"]) + 1)
    metrics = ["loss", "accuracy", "f1_macro"]
    titles = ["Loss", "Accuracy", "Macro F1"]
    colors = {"train": "#2196F3", "val": "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        for split in ["train", "val"]:
            values = [m[metric] for m in history[split]]
            ax.plot(epochs, values, color=colors[split], label=split, linewidth=2, marker="o", markersize=3)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "training_history.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training history saved -> {out_path}")


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    output_path: Path,
    task: str = "multiclass",
):
    """Plot ROC curves (one-vs-rest for multiclass)."""
    n_classes = len(class_names)
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    fig, ax = plt.subplots(figsize=(8, 7))

    if task == "binary":
        fpr, tpr, _ = roc_curve(labels, probs if probs.ndim == 1 else probs[:, 1])
        auc = roc_auc_score(labels, probs if probs.ndim == 1 else probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        lb = label_binarize(labels, classes=list(range(n_classes)))
        for i, (name, color) in enumerate(zip(class_names, colors)):
            if lb[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
            auc = roc_auc_score(lb[:, i], probs[:, i])
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved -> {output_path}")


def plot_per_class_f1(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
):
    """Bar chart of per-class F1 scores."""
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    palette = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, f1s, color=palette, edgecolor="black", linewidth=0.8)

    # Annotate bars
    for bar, score in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center", fontsize=10, fontweight="bold"
        )

    ax.axhline(macro_f1, color="navy", linestyle="--", linewidth=2,
               label=f"Macro F1 = {macro_f1:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class F1 saved -> {output_path}")


# Main evaluation

def evaluate(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    task: str = "multiclass",
    model_name: str = "resnet50",
    batch_size: int = 32,
    history_path: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, _, test_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        task=task,
        batch_size=batch_size,
        num_workers=2,
    )

    # Load model
    n_classes = 2 if task == "binary" else 6
    model = build_model(model_name, n_classes=n_classes)
    model, saved_metrics = load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)

    print(f"\nEvaluating on test set...")
    labels, preds, probs = collect_predictions(model, test_loader, device, task)

    # Metrics
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, target_names=class_names, digits=4)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {model_name} ({task})")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(report)

    # Save metrics to JSON
    metrics = {
        "model": model_name,
        "task": task,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }
    with open(output_dir / f"metrics_{task}_{model_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_confusion_matrix(
        labels, preds, class_names,
        output_dir / f"confusion_matrix_{task}_{model_name}.png"
    )
    plot_per_class_f1(
        labels, preds, class_names,
        output_dir / f"per_class_f1_{task}_{model_name}.png"
    )
    plot_roc_curves(
        labels, probs, class_names,
        output_dir / f"roc_curves_{task}_{model_name}.png",
        task=task
    )

    if history_path:
        plot_training_history(history_path, output_dir)

    print(f"\nAll evaluation figures saved to {output_dir}")
    return metrics



# CLI

def main():
    parser = argparse.ArgumentParser(description="Evaluate CGR classifier")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint")
    parser.add_argument("--data_dir", required=True, help="CGR image directory")
    parser.add_argument("--output_dir", default="results/figures/")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--history", default=None, help="Path to training history JSON")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task=args.task,
        model_name=args.model,
        batch_size=args.batch_size,
        history_path=args.history,
    )


if __name__ == "__main__":
    main()
