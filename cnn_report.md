# Model Report – Custom CNN (Image)

**Modality:** Image only  
**Category:** Deep learning (convolutional, trained from scratch)  
**File:** `models/custom_cnn.py`

---

## Overview

A 4-block custom Convolutional Neural Network that learns visual features directly from book cover images. The model is trained from random initialisation — no pretrained weights — making it a fair baseline for what can be achieved from images alone on a limited dataset.

Book covers carry meaningful genre signals: colour palette, typography style, illustration vs. photography, and layout patterns all vary systematically by genre.

---

## Pipeline

```
cover image (any size)
      ↓
ImageDataGenerator      # rescale [0,1], rotation ±10°, h-flip, small shifts
      ↓
  Resize 224×224        # standard CNN input size
      ↓
 Conv(32)  → BN → MaxPool(2×2)
 Conv(64)  → BN → MaxPool(2×2)
 Conv(128) → BN → MaxPool(2×2)
 Conv(256) → BN → GlobalAvgPool
      ↓
Dense(256, relu, L2=1e-4) → Dropout(0.5)
      ↓
Dense(6, softmax)
```

---

## Architecture Details

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D(32, 3×3) | 224×224×32 | 896 |
| BatchNorm + MaxPool | 112×112×32 | — |
| Conv2D(64, 3×3) | 112×112×64 | 18 496 |
| BatchNorm + MaxPool | 56×56×64 | — |
| Conv2D(128, 3×3) | 56×56×128 | 73 856 |
| BatchNorm + MaxPool | 28×28×128 | — |
| Conv2D(256, 3×3) | 28×28×256 | 295 168 |
| BatchNorm + GlobalAvgPool | 256 | — |
| Dense(256) + Dropout | 256 | 65 792 |
| Dense(6) | 6 | 1 542 |

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image size | 224×224 | Matches ResNet50 input for fair comparison |
| Batch size | 16 | Small batch suits the dataset size |
| Learning rate | 1e-3 (Adam) | Standard starting point |
| Dropout | 0.5 | Aggressive regularisation due to limited data |
| L2 | 1e-4 | Small weight penalty |
| Epochs | 6 (+ EarlyStopping) | |
| Patience | 3 | More lenient than LSTM; image training is noisier |

---

## Data Augmentation

| Transform | Value | Purpose |
|-----------|-------|---------|
| `rescale` | 1/255 | Normalise pixel values to [0, 1] |
| `rotation_range` | ±10° | Orientation robustness |
| `width_shift_range` | 5% | Slight translation invariance |
| `height_shift_range` | 5% | Slight translation invariance |
| `horizontal_flip` | True | Doubles effective dataset size |

---

## Strengths

- **No external dependencies** – entirely self-contained, no pretrained weights needed.
- **Lightweight** – ~455 K trainable parameters, fast inference.
- **Augmentation** compensates for dataset size.

---

## Limitations

- **Limited feature depth** – shallow compared to ResNet50 (50 layers vs. 4 blocks).
- **Random initialisation** – must learn low-level features (edges, textures) from scratch.
- **Small dataset** – ~850 training images is marginal for an image CNN.

---

## Expected Results

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~0.48 | ~0.45 |
| Macro F1 | ~0.46 | ~0.43 |

> Image-only models generally underperform text-only models on this dataset due to the limited number of training images.

---

## How to Run

```bash
python models/custom_cnn.py
```

---

## Artefacts Produced

| Artefact | Description |
|----------|-------------|
| `cnn_model` | Trained Keras model |
| Training history plots | Loss and accuracy curves |
| Confusion matrix plot | Displayed via `matplotlib` |
| Classification report | Printed to stdout |
