#!/usr/bin/env bash


# Runs full pipeline: data download, CGR encoding , training , evaluation

set -euo pipefail

# Configuration
DATA_DIR="data/raw"
CGR_DIR="data/cgr_images"
RESULTS_DIR="results"
N_PER_CLASS=1667       # ~10,000 total sequences across 6 classes
IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS_BINARY=30
EPOCHS_MULTI=50
N_WORKERS=4

echo "  DNA Sequence CGR Classification Pipeline"
echo ""

# Step 1: Download data
echo "[1/5] Downloading sequences from CARD + Ensembl Bacteria ..."
python src/data_download.py \
    --download \
    --parse \
    --output_dir "$DATA_DIR" \
    --n_per_class "$N_PER_CLASS" \
    --seed 42
echo ""

# Step 2: Generate CGR images
echo "[2/5] Generating CGR images..."
python src/cgr_encoding.py \
    --fasta_dir "$DATA_DIR" \
    --output_dir "$CGR_DIR" \
    --image_size "$IMAGE_SIZE" \
    --use_color \
    --n_workers "$N_WORKERS"
echo ""

# Step 3: Binary classification
echo "[3/5] Training binary classifiers (resistant vs non-resistant)..."

for MODEL in resnet50 efficientnet_b0; do
    echo ""
    echo "  Model: $MODEL"
    python src/train.py \
        --data_dir "$CGR_DIR" \
        --task binary \
        --model "$MODEL" \
        --epochs "$EPOCHS_BINARY" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$RESULTS_DIR"
done

# Step 4: Multi-class classification
echo ""
echo "[4/5] Training multi-class classifiers (6 resistance types)..."

for MODEL in resnet50 efficientnet_b0; do
    echo ""
    echo "  Model: $MODEL"
    python src/train.py \
        --data_dir "$CGR_DIR" \
        --task multiclass \
        --model "$MODEL" \
        --epochs "$EPOCHS_MULTI" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$RESULTS_DIR"
done

# Step 5: Evaluation
echo ""
echo "[5/5] Evaluating models..."

for MODEL in resnet50 efficientnet_b0; do
    for TASK in binary multiclass; do
        CKPT="${RESULTS_DIR}/checkpoints/best_${TASK}_${MODEL}.pt"
        HISTORY="${RESULTS_DIR}/history_${TASK}_${MODEL}.json"
        if [ -f "$CKPT" ]; then
            echo ""
            echo "  Evaluating: $MODEL ($TASK)"
            python src/evaluate.py \
                --checkpoint "$CKPT" \
                --data_dir "$CGR_DIR" \
                --task "$TASK" \
                --model "$MODEL" \
                --output_dir "${RESULTS_DIR}/figures" \
                --history "$HISTORY"
        fi
    done
done


echo "  Pipeline complete! Results in: $RESULTS_DIR"
