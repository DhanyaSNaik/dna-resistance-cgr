"""
Chaos Game Representation (CGR) encoder.

Each DNA sequence is converted to a 224×224 frequency image:
  - Four DNA bases map to corners of the unit square:
      A = (0, 0)  (bottom-left)
      T = (1, 0)  (bottom-right)
      G = (1, 1)  (top-right)
      C = (0, 1)  (top-left)
  - For each base b_i, the current position moves halfway toward corner(b_i)
  - The image is a 2D histogram of all visited positions

This encoding captures k-mer frequency structure at all scales simultaneously.
A 224×224 image encodes all k-mers up to k ≈ log4(224^2) ≈ 8.

"""

import argparse
import hashlib
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
from Bio import SeqIO
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# CGR corner coordinates (unit square)
CGR_CORNERS = {
    'A': np.array([0.0, 0.0]),
    'T': np.array([1.0, 0.0]),
    'G': np.array([1.0, 1.0]),
    'C': np.array([0.0, 1.0]),
    # Ambiguous bases: place at center
    'N': np.array([0.5, 0.5]),
    'R': np.array([0.5, 0.5]),  # A or G
    'Y': np.array([0.5, 0.5]),  # C or T
    'S': np.array([0.5, 0.5]),  # G or C
    'W': np.array([0.5, 0.5]),  # A or T
    'K': np.array([0.5, 0.5]),  # G or T
    'M': np.array([0.5, 0.5]),  # A or C
}


def sequence_to_cgr_trajectory(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute CGR trajectory for a DNA sequence.

    Returns:
        x_coords: array of shape (N,) in [0, 1]
        y_coords: array of shape (N,) in [0, 1]
    """
    seq = sequence.upper()
    n = len(seq)
    x = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    cx, cy = 0.5, 0.5  # Start at center
    for i, base in enumerate(seq):
        corner = CGR_CORNERS.get(base, CGR_CORNERS['N'])
        cx = (cx + corner[0]) / 2.0
        cy = (cy + corner[1]) / 2.0
        x[i] = cx
        y[i] = cy

    return x, y


def cgr_to_image(sequence: str, image_size: int = 224,
                 colormap: str = "viridis") -> np.ndarray:
    """
    Convert a DNA sequence to a CGR frequency image.

    Args:
        sequence:   DNA string (ATGC...)
        image_size: output image dimension (square)
        colormap:   matplotlib colormap name for RGB encoding

    Returns:
        RGB image as uint8 array of shape (image_size, image_size, 3)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = sequence_to_cgr_trajectory(sequence)

    # Build 2D frequency histogram
    # Invert y so that (0,0) = bottom-left matches corner convention
    bins = image_size
    freq_map, _, _ = np.histogram2d(
        x, 1.0 - y,   # flip y-axis for image coordinates
        bins=bins,
        range=[[0, 1], [0, 1]]
    )

    # Normalize to [0, 1] using log scaling for better contrast
    # Add 1 to avoid log(0)
    freq_map = np.log1p(freq_map)
    if freq_map.max() > 0:
        freq_map = freq_map / freq_map.max()

    # Apply colormap to get RGB
    cmap = plt.get_cmap(colormap)
    rgb = (cmap(freq_map)[:, :, :3] * 255).astype(np.uint8)

    # Transpose to get image orientation: (row, col) = (y, x)
    rgb = rgb.transpose(1, 0, 2)  # (H, W, C)

    return rgb


def cgr_to_image_fast(sequence: str, image_size: int = 224) -> np.ndarray:
    """
    Fast greyscale CGR image.
    Returns 3-channel image (identical channels) for CNN compatibility.
    """
    x, y = sequence_to_cgr_trajectory(sequence)

    # Scale to pixel indices
    xi = np.clip((x * image_size).astype(np.int32), 0, image_size - 1)
    yi = np.clip(((1.0 - y) * image_size).astype(np.int32), 0, image_size - 1)

    freq_map = np.zeros((image_size, image_size), dtype=np.float32)
    np.add.at(freq_map, (yi, xi), 1)

    # Log-normalize
    freq_map = np.log1p(freq_map)
    if freq_map.max() > 0:
        freq_map /= freq_map.max()

    pixel = (freq_map * 255).astype(np.uint8)
    rgb = np.stack([pixel, pixel, pixel], axis=-1)
    return rgb


def inverse_cgr(px: int, py: int, image_size: int, k: int) -> str:
    """
    Backproject a pixel (px, py) in a CGR image to its corresponding k-mer.

    The CGR image partitions [0,1]^2 into 4^k cells, each corresponding
    to a unique k-mer. This is the basis for Grad-CAM biological interpretation.

    Args:
        px, py:     pixel coordinates (column, row)
        image_size: image dimension
        k:          k-mer length to decode

    Returns:
        k-mer string of length k
    """
    # Normalize pixel to [0, 1]
    x = (px + 0.5) / image_size
    y = 1.0 - (py + 0.5) / image_size  # un-flip y

    # Decode k-mer by iteratively finding which quadrant
    # Corner encoding: A=(0,0), T=(1,0), G=(1,1), C=(0,1)
    quadrant_to_base = {
        (0, 0): 'A',   # x < 0.5, y < 0.5
        (1, 0): 'T',   # x >= 0.5, y < 0.5
        (1, 1): 'G',   # x >= 0.5, y >= 0.5
        (0, 1): 'C',   # x < 0.5, y >= 0.5
    }

    kmer = []
    for _ in range(k):
        qx = 1 if x >= 0.5 else 0
        qy = 1 if y >= 0.5 else 0
        base = quadrant_to_base.get((qx, qy), 'N')
        kmer.append(base)
        # Zoom into this quadrant
        x = x * 2 - qx
        y = y * 2 - qy

    return "".join(reversed(kmer))  # CGR reads k-mer in reverse order


# Batch processing

def _process_record(args):
    """Worker function for multiprocessing."""
    record, out_dir, image_size, use_color = args
    label = out_dir.name
    seq_id = hashlib.md5(str(record.id).encode()).hexdigest()[:8]
    out_path = out_dir / f"{seq_id}.png"

    if out_path.exists():
        return True

    seq = str(record.seq).upper()
    if len(seq) < 50:
        return False

    if use_color:
        img_array = cgr_to_image(seq, image_size)
    else:
        img_array = cgr_to_image_fast(seq, image_size)

    img = Image.fromarray(img_array, mode="RGB")
    img.save(out_path, format="PNG", optimize=False)
    return True


def encode_fasta_to_images(fasta_path: Path, output_dir: Path,
                           image_size: int = 224, use_color: bool = True,
                           n_workers: int = 4) -> int:
    """
    Encode all sequences in a FASTA file to CGR images.

    Args:
        fasta_path:  input FASTA file
        output_dir:  directory to save PNG images
        image_size:  output image dimension
        use_color:   use viridis colormap (True) or greyscale (False)
        n_workers:   number of parallel workers

    Returns:
        Number of images successfully created
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(SeqIO.parse(fasta_path, "fasta"))

    if not records:
        logger.warning(f"No sequences found in {fasta_path}")
        return 0

    logger.info(f"  Encoding {len(records)} sequences from {fasta_path.name}")

    args_list = [(r, output_dir, image_size, use_color) for r in records]

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_record, args_list),
                total=len(args_list),
                desc=f"  {output_dir.name}"
            ))
    else:
        results = [_process_record(a) for a in tqdm(args_list, desc=f"  {output_dir.name}")]

    n_success = sum(results)
    logger.info(f"  {n_success}/{len(records)} images saved to {output_dir}")
    return n_success


# EDA utilities

def compute_cgr_statistics(sequence: str) -> dict:
    """Compute compositional statistics from CGR trajectory."""
    seq = sequence.upper()
    n = len(seq)
    counts = {b: seq.count(b) for b in "ATGC"}
    gc = (counts.get('G', 0) + counts.get('C', 0)) / max(n, 1)

    x, y = sequence_to_cgr_trajectory(seq)

    return {
        "length": n,
        "gc_content": gc,
        "A_freq": counts.get('A', 0) / n,
        "T_freq": counts.get('T', 0) / n,
        "G_freq": counts.get('G', 0) / n,
        "C_freq": counts.get('C', 0) / n,
        "cgr_x_mean": float(x.mean()),
        "cgr_y_mean": float(y.mean()),
        "cgr_x_std": float(x.std()),
        "cgr_y_std": float(y.std()),
    }


# CLI

def main():
    parser = argparse.ArgumentParser(description="Generate CGR images from FASTA files")
    parser.add_argument("--fasta_dir", required=True, help="Directory with .fasta files")
    parser.add_argument("--output_dir", required=True, help="Output directory for CGR images")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size in pixels (default: 224)")
    parser.add_argument("--use_color", action="store_true", default=True,
                        help="Use viridis colormap (default: True)")
    parser.add_argument("--greyscale", action="store_false", dest="use_color",
                        help="Use greyscale instead of colormap")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    fasta_dir = Path(args.fasta_dir)
    output_dir = Path(args.output_dir)

    fasta_files = list(fasta_dir.glob("*.fasta")) + list(fasta_dir.glob("*.fa"))
    if not fasta_files:
        logger.error(f"No FASTA files found in {fasta_dir}")
        return

    logger.info(f"\nCGR Image Generation")
    logger.info(f"  Image size: {args.image_size}×{args.image_size}")
    logger.info(f"  Colormap:   {'viridis' if args.use_color else 'greyscale'}")
    logger.info(f"  Workers:    {args.n_workers}")
    logger.info(f"  FASTA dir:  {fasta_dir}")
    logger.info(f"  Output dir: {output_dir}\n")

    total = 0
    for fasta_path in sorted(fasta_files):
        label = fasta_path.stem
        class_out = output_dir / label
        n = encode_fasta_to_images(
            fasta_path, class_out,
            image_size=args.image_size,
            use_color=args.use_color,
            n_workers=args.n_workers
        )
        total += n

    logger.info(f"\n{'='*50}")
    logger.info(f"Total CGR images generated: {total}")
    logger.info(f"Saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
