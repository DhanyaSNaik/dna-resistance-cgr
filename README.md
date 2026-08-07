# Chaos Game Representation-Based Convolutional Neural Network for Antibiotic Resistance Classification


## Overview

This project converts DNA sequences into 2D frequency images using Chaos Game Representation (CGR) and classifies them as resistant or non-resistant using fine-tuned CNNs (ResNet-50, EfficientNet-B0). Grad-CAM visualizations reveal which sequence patterns drive each prediction.

### Dataset
| Class | Source | Sequences |
|-------|--------|-----------|
| `resistant` | CARD (all resistance gene families) | 1,667 |
| `non_resistant` | Ensembl Bacteria (*E. coli* K-12 + *S. aureus* N315) | 1,667 |
| **Total** | | **3,334** |

### Results
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| ResNet-50 | 97.21% | 0.9721 |
| EfficientNet-B0 | 92.61% | 0.9261 |


---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data

Downloads the CARD resistance gene database and Ensembl Bacteria CDS files, then parses them into two balanced FASTA files.

```bash
python src/data_download.py --download --parse --output_dir data/raw
```

This creates:
- `data/raw/resistant.fasta` — 1,667 resistance genes from CARD
- `data/raw/non_resistant.fasta` — 1,667 housekeeping genes from *E. coli* + *S. aureus*

### 3. Generate CGR images

Converts each DNA sequence to a 224×224 frequency image.

```bash
python src/cgr_encoding.py \
  --fasta_dir data/raw \
  --output_dir data/cgr_images \
  --image_size 224 \
  --use_color \
  --n_workers 4
```

This creates `data/cgr_images/resistant/` and `data/cgr_images/non_resistant/` each containing PNG images.

### 4. Train

Train ResNet-50 and EfficientNet-B0 for binary classification (resistant vs non-resistant).

```bash
python src/train.py \
  --data_dir data/cgr_images \
  --task binary \
  --model resnet50 \
  --epochs 30 \
  --output_dir results/

python src/train.py \
  --data_dir data/cgr_images \
  --task binary \
  --model efficientnet_b0 \
  --epochs 30 \
  --output_dir results/
```

Best checkpoints are saved to `results/checkpoints/`. Training logs (loss, accuracy, F1 per epoch) are saved to `results/logs/`.

### 5. Evaluate

Computes accuracy, macro F1, confusion matrix, ROC curve, and per-class F1 on the test set.

```bash
python src/evaluate.py \
  --checkpoint results/checkpoints/best_binary_resnet50.pt \
  --data_dir data/cgr_images \
  --task binary \
  --model resnet50 \
  --output_dir results/figures/

python src/evaluate.py \
  --checkpoint results/checkpoints/best_binary_efficientnet_b0.pt \
  --data_dir data/cgr_images \
  --task binary \
  --model efficientnet_b0 \
  --output_dir results/figures/
```

Figures are saved to `results/figures/`.

### 6. Grad-CAM analysis

Generates activation heatmaps and maps high-activation pixels back to genomic k-mer positions via inverse CGR.

```bash
python src/gradcam_analysis.py \
  --checkpoint results/checkpoints/best_binary_resnet50.pt \
  --data_dir data/cgr_images \
  --output_dir results/figures/gradcam/resnet50 \
  --task binary \
  --model resnet50 \
  --n_per_class 10

python src/gradcam_analysis.py \
  --checkpoint results/checkpoints/best_binary_efficientnet_b0.pt \
  --data_dir data/cgr_images \
  --output_dir results/figures/gradcam/efficientnet_b0 \
  --task binary \
  --model efficientnet_b0 \
  --n_per_class 10
```

Results saved to `results/figures/gradcam/` including per-class mean heatmaps and k-mer analysis JSON files.

### Run everything at once

```bash
bash scripts/run_pipeline.sh   # steps 2-5
bash scripts/run_gradcam.sh    # step 6
```

---

## Methods

### Chaos Game Representation (CGR)
Each DNA sequence of length N generates one 224x224 frequency image. The four bases are assigned to corners of the unit square: A=(0,0), T=(1,0), G=(1,1), C=(0,1). For each base b_i, the current position moves halfway toward its corner. The final image is a 2D histogram of all visited positions, log-normalised and rendered with the viridis colormap.

This encoding captures k-mer frequency distributions at all scales - a 224x224 image encodes all k-mers up to k~8.

### CNN Fine-tuning
- **ResNet-50**: ImageNet pretrained, final FC replaced with Dropout(0.3) + Linear(2048, 2)
- **EfficientNet-B0**: ImageNet pretrained, classifier replaced with Dropout(0.3) + Linear(1280, 2)
- Optimizer: Adam, lr=1e-4, weight decay=1e-4
- Schedule: ReduceLROnPlateau (patience=5, factor=0.5)
- Split: 70/15/15 stratified train/val/test
- Augmentation: random flips, rotation +/-10 degrees, colour jitter

### Grad-CAM
Activation maps from the last convolutional layer are upsampled to 224x224 and overlaid on CGR images. High-activation pixels are mapped back to genomic positions via inverse CGR, identifying specific k-mers that drive predictions.
