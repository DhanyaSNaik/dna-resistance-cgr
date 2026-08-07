"""
Grad-CAM visualization for CGR-based DNA sequence classifiers.

Steps:
  1. Load model and test images
  2. Generate Grad-CAM activation maps (pytorch-grad-cam)
  3. Overlay heatmaps on CGR images
  4. Backproject activated pixels to genomic positions (k-mer mapping)
  5. Compare with known resistance-conferring motifs

"""

import argparse
import colorsys
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

from cgr_encoding import inverse_cgr, CGR_CORNERS
from dataset import build_dataloaders, CLASSES_MULTICLASS, CLASSES_BINARY, get_transforms
from models import build_model, load_checkpoint


# Known resistance-conferring motifs for biological validation

RESISTANCE_MOTIFS = {
    "beta_lactam": {
        "SXXK": ["SAMK", "STMK", "SVMK"],     # Active site serine motif
        "SDN":  ["SDN"],                          # SDN loop
        "KTG":  ["KTG"],                          # KTG motif
        "description": "β-lactamase active site: S70, K73, SDN loop, KTG box"
    },
    "tetracycline": {
        "TMD":  ["TMDS", "TMDA"],                 # TetM/TetO GTPase domain
        "GKT":  ["GKTT", "GKTS"],                 # P-loop NTPase
        "description": "TetM ribosomal protection: GTPase domain, P-loop motifs"
    },
    "aminoglycoside": {
        "GXXGX": ["GKAGK", "GDAGK"],             # Walker A motif (AAC/APH)
        "DEAD":  ["DEAH", "DEVH"],                # Walker B motif
        "description": "Aminoglycoside kinase/acetyltransferase ATP-binding"
    },
    "macrolide": {
        "CXXXC": ["CXXXC"],                       # Zinc finger (ErmC)
        "GXGXXG": ["GXGXXG"],                     # Rossmann fold (methyltransferases)
        "description": "ErmB/ErmC rRNA methyltransferase SAM-binding domain"
    },
    "fluoroquinolone": {
        "QKKG": ["QKKG"],                         # GyrA QRDR motif
        "DXXT": ["DXXE"],                         # Metal coordination
        "description": "GyrA QRDR: Ser83, Asp87 substitution hotspots"
    },
}

# k-mer length for backprojection (2^k pixels per k-mer cell)
KMER_LENGTH = 6  # 4^6 = 4096 cells


# Grad-CAM wrapper

class CGRGradCAM:
    """
    Wrapper around pytorch-grad-cam for CGR images.

    Supports GradCAM, GradCAM++, and EigenCAM methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        method: str = "gradcam",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        target_layers = model.get_gradcam_target_layer()

        cam_cls = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "eigencam": EigenCAM}
        cam_cls = cam_cls.get(method.lower(), GradCAM)

        self.cam = cam_cls(model=model, target_layers=target_layers)

    def compute(
        self,
        img_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute Grad-CAM heatmap for an image.

        Args:
            img_tensor:   (1, 3, H, W) normalized image tensor
            target_class: class index to visualize (None = predicted class)

        Returns:
            heatmap: (H, W) float array in [0, 1]
            pred_class: predicted class index
        """
        img_tensor = img_tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred_class = logits.argmax(dim=1).item()

        cls = target_class if target_class is not None else pred_class
        targets = [ClassifierOutputTarget(cls)]

        heatmap = self.cam(input_tensor=img_tensor, targets=targets)
        return heatmap[0], pred_class  # (H, W)


# Visualization

def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original CGR image."""
    # original_img: (H, W, 3) uint8
    # heatmap: (H, W) float [0,1]
    cam_img = show_cam_on_image(
        original_img.astype(np.float32) / 255.0,
        heatmap,
        use_rgb=True,
        image_weight=1 - alpha,
        colormap=cv2.COLORMAP_JET,
    )
    return cam_img


def plot_gradcam_grid(
    images: list[np.ndarray],
    heatmaps: list[np.ndarray],
    overlays: list[np.ndarray],
    titles: list[str],
    class_name: str,
    output_path: Path,
    n_cols: int = 5,
):
    """Plot grid of (original | heatmap | overlay) for a class."""
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"CGR Image {i+1}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmaps[i], cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlays[i])
        axes[i, 2].set_title("Overlay", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle(f"Grad-CAM Analysis: {class_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_mean_heatmap(
    mean_heatmap: np.ndarray,
    class_name: str,
    output_path: Path,
    top_kmers: list[tuple[str, float]] | None = None,
):
    """
    Plot mean Grad-CAM heatmap across all samples of a class,
    with CGR corner annotations and top k-mer labels.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(mean_heatmap, cmap="inferno", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Activation")

    # Annotate CGR corners
    size = mean_heatmap.shape[0]
    corner_labels = {"A": (5, size - 10), "T": (size - 20, size - 10),
                     "G": (size - 20, 10), "C": (5, 10)}
    for label, (x, y) in corner_labels.items():
        ax.text(x, y, label, color="white", fontsize=14, fontweight="bold",
                ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    ax.set_title(
        f"Mean Grad-CAM: {class_name}\n(darker = higher activation)",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    # Annotate top k-mers if provided
    if top_kmers:
        kmer_text = "Top activated k-mers:\n" + "\n".join(
            f"  {k}: {v:.3f}" for k, v in top_kmers[:8]
        )
        ax.text(
            1.02, 0.5, kmer_text,
            transform=ax.transAxes,
            fontsize=8, verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# Genomic backprojection

def heatmap_to_kmers(
    heatmap: np.ndarray,
    k: int = KMER_LENGTH,
    top_n: int = 20,
    threshold: float = 0.7,
) -> list[tuple[str, float]]:
    """
    Backproject high-activation pixels in a Grad-CAM heatmap
    to their corresponding k-mers.

    Args:
        heatmap:   (H, W) float activation map in [0, 1]
        k:         k-mer length to decode
        top_n:     number of top k-mers to return
        threshold: activation threshold (0–1)

    Returns:
        List of (k-mer, mean_activation) sorted by activation descending
    """
    H, W = heatmap.shape
    kmer_activations = defaultdict(list)

    # Sample pixel positions above threshold
    mask = heatmap >= threshold
    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        kmer = inverse_cgr(px=x, py=y, image_size=W, k=k)
        kmer_activations[kmer].append(float(heatmap[y, x]))

    # Aggregate: mean activation per k-mer
    kmer_scores = {
        kmer: np.mean(acts)
        for kmer, acts in kmer_activations.items()
        if len(acts) >= 2  # require at least 2 contributing pixels
    }

    sorted_kmers = sorted(kmer_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_kmers[:top_n]


def compare_kmers_to_motifs(
    top_kmers: list[tuple[str, float]],
    class_name: str,
) -> list[dict]:
    """
    Compare top activated k-mers against known resistance motifs.

    Returns list of matches with biological annotation.
    """
    if class_name not in RESISTANCE_MOTIFS:
        return []

    motif_info = RESISTANCE_MOTIFS[class_name]
    matches = []

    for kmer, score in top_kmers:
        for motif_name, motif_seqs in motif_info.items():
            if motif_name == "description":
                continue
            for motif_seq in motif_seqs:
                # count matching positions
                min_len = min(len(kmer), len(motif_seq))
                matches_pos = sum(
                    1 for a, b in zip(kmer[:min_len], motif_seq[:min_len])
                    if a == b or b == 'X'
                )
                similarity = matches_pos / min_len
                if similarity >= 0.6:
                    matches.append({
                        "kmer": kmer,
                        "activation": score,
                        "motif": motif_name,
                        "motif_seq": motif_seq,
                        "similarity": similarity,
                        "description": motif_info["description"],
                    })

    return sorted(matches, key=lambda x: x["activation"], reverse=True)


# Main analysis pipeline

def run_gradcam_analysis(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    task: str = "multiclass",
    model_name: str = "resnet50",
    cam_method: str = "gradcam",
    n_per_class: int = 10,
    kmer_k: int = KMER_LENGTH,
    batch_size: int = 8,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Load model
    n_classes = 2 if task == "binary" else 6
    model = build_model(model_name, n_classes=n_classes)
    model, _ = load_checkpoint(checkpoint_path, model, device=device_str)
    model = model.to(device)
    model.eval()

    cam_engine = CGRGradCAM(model, method=cam_method, device=device_str)

    class_names = CLASSES_BINARY if task == "binary" else CLASSES_MULTICLASS
    transform = get_transforms("test")

    # Collect per-class image paths
    data_dir_p = Path(data_dir)
    all_results = {}

    print(f"\nRunning Grad-CAM analysis ({cam_method.upper()})...")
    print(f"  k-mer length: {kmer_k}")
    print(f"  Samples/class: {n_per_class}\n")

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = data_dir_p / cls_name
        if not cls_dir.exists():
            print(f"  [SKIP] {cls_name}: directory not found")
            continue

        img_paths = sorted(cls_dir.glob("*.png"))[:n_per_class]
        if not img_paths:
            print(f"  [SKIP] {cls_name}: no images")
            continue

        print(f"\n  Class: {cls_name} ({len(img_paths)} images)")
        cls_out = output_dir / cls_name
        cls_out.mkdir(exist_ok=True)

        orig_images = []
        heatmaps_list = []
        overlays_list = []
        all_kmers = []

        for img_path in tqdm(img_paths, desc=f"    {cls_name}"):
            # Load original image (for display)
            orig = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
            orig_images.append(orig)

            # Transform for model
            pil_img = Image.open(img_path).convert("RGB")
            tensor = transform(pil_img).unsqueeze(0)

            # Grad-CAM
            heatmap, pred_cls = cam_engine.compute(tensor, target_class=cls_idx)
            heatmap_resized = cv2.resize(heatmap, (224, 224))

            heatmaps_list.append(heatmap_resized)
            overlays_list.append(overlay_heatmap(orig, heatmap_resized))

            # K-mer backprojection
            kmers = heatmap_to_kmers(heatmap_resized, k=kmer_k)
            all_kmers.extend(kmers)

        # Aggregate k-mer activations
        kmer_agg = defaultdict(list)
        for km, sc in all_kmers:
            kmer_agg[km].append(sc)
        top_kmers = sorted(
            [(k, np.mean(v)) for k, v in kmer_agg.items()],
            key=lambda x: x[1], reverse=True
        )[:20]

        # Biological interpretation
        bio_matches = compare_kmers_to_motifs(top_kmers, cls_name)

        # Mean heatmap
        mean_hmap = np.mean(heatmaps_list, axis=0)

        # Save outputs
        # Per-class grid
        plot_gradcam_grid(
            orig_images[:min(5, len(orig_images))],
            heatmaps_list[:5],
            overlays_list[:5],
            [cls_name] * 5,
            cls_name,
            cls_out / f"gradcam_grid_{cls_name}.png",
        )

        # Mean heatmap
        plot_mean_heatmap(
            mean_hmap,
            cls_name,
            cls_out / f"mean_heatmap_{cls_name}.png",
            top_kmers=top_kmers,
        )

        # JSON: top k-mers and biological matches
        results = {
            "class": cls_name,
            "n_images": len(img_paths),
            "top_kmers": [{"kmer": k, "mean_activation": v} for k, v in top_kmers],
            "biological_matches": bio_matches,
            "cam_method": cam_method,
            "kmer_k": kmer_k,
        }
        with open(cls_out / f"kmer_analysis_{cls_name}.json", "w") as f:
            json.dump(results, f, indent=2)

        all_results[cls_name] = results

        print(f"    Top 5 k-mers: {[k for k, _ in top_kmers[:5]]}")
        if bio_matches:
            print(f"    Bio matches: {len(bio_matches)} motif hits")
            for m in bio_matches[:2]:
                print(f"      {m['kmer']} ~ {m['motif']} ({m['similarity']:.2f} similarity)")

    # Summary report
    print(f"\n{'='*60}")
    print("GRAD-CAM BIOLOGICAL INTERPRETATION SUMMARY")
    print(f"{'='*60}")
    for cls_name, results in all_results.items():
        if results["biological_matches"]:
            top_match = results["biological_matches"][0]
            print(f"\n  {cls_name}:")
            print(f"    {top_match['description']}")
            print(f"    Top k-mer: {top_match['kmer']} -> motif {top_match['motif']}")
        else:
            print(f"\n  {cls_name}: top k-mers = {[k for k, _ in results['top_kmers'][:3]]}")

    # Save summary JSON
    with open(output_dir / "gradcam_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nGrad-CAM analysis complete. Results saved to {output_dir}")
    return all_results


# CLI

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM analysis for CGR classifiers")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="results/figures/gradcam/")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--method", choices=["gradcam", "gradcam++", "eigencam"],
                        default="gradcam")
    parser.add_argument("--n_per_class", type=int, default=10)
    parser.add_argument("--kmer_k", type=int, default=6)
    args = parser.parse_args()

    run_gradcam_analysis(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task=args.task,
        model_name=args.model,
        cam_method=args.method,
        n_per_class=args.n_per_class,
        kmer_k=args.kmer_k,
    )


if __name__ == "__main__":
    main()
