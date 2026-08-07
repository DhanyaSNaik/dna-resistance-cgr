#!/usr/bin/env bash


# Runs Grad-CAM analysis on trained models

set -euo pipefail

CGR_DIR="data/cgr_images"
RESULTS_DIR="results"
N_PER_CLASS=10    # images per class for analysis
KMER_K=6          # k-mer length for backprojection

echo "  Grad-CAM Analysis + Genomic Backprojection"

for MODEL in resnet50 efficientnet_b0; do
    for METHOD in gradcam "gradcam++"; do
        CKPT="${RESULTS_DIR}/checkpoints/best_multiclass_${MODEL}.pt"
        OUT_DIR="${RESULTS_DIR}/figures/gradcam/${MODEL}_${METHOD}"

        if [ -f "$CKPT" ]; then
            echo ""
            echo "  Model: $MODEL | Method: $METHOD"
            python src/gradcam_analysis.py \
                --checkpoint "$CKPT" \
                --data_dir "$CGR_DIR" \
                --output_dir "$OUT_DIR" \
                --task multiclass \
                --model "$MODEL" \
                --method "$METHOD" \
                --n_per_class "$N_PER_CLASS" \
                --kmer_k "$KMER_K"
        else
            echo "  [SKIP] Checkpoint not found: $CKPT"
        fi
    done
done

echo ""
echo "Grad-CAM analysis complete."
echo "Heatmaps saved to: ${RESULTS_DIR}/figures/gradcam/"
