# Model Report – ResNet50 Transfer Learning (Image)

**Modality:** Image only  
**Category:** Deep learning (transfer learning)  
**File:** `models/resnet50.py`

---

## Overview

ResNet50 is a 50-layer residual network pre-trained on ImageNet (1.2M images, 1 000 classes). By using it as a frozen feature extractor, we inherit powerful low- and mid-level visual representations (edges, textures, object parts) without any of the training cost. Only the small classification head on top is trained on our 6-genre dataset.

---

## Pipeline

```
cover image (any size)
      ↓
ImageDataGenerator      # ResNet-specific preprocessing (channel-wise mean sub.)
                        # + rotation ±10°, horizontal flip
      ↓
  Resize 224×224
      ↓
ResNet50 (frozen)       # 48 conv layers + skip connections; outputs 2048-dim vector
                        # Global Average Pooling at the top (pooling="avg")
      ↓
Dense(512, relu, L2=1e-4)
      ↓
   Dropout(0.4)
      ↓
Dense(6, softmax)
```

---

## Transfer Learning Strategy

| Component | Status | Rationale |
|-----------|--------|-----------|
| ResNet50 backbone | **Frozen** | Preserves ImageNet features; avoids catastrophic forgetting |
| Dense head | **Trained** | Adapts features to the 6 genre classes |
| Preprocessing | ResNet-specific | Channel mean subtraction expected by the pretrained weights |

> **Tip:** Fine-tuning (unfreezing the last few ResNet blocks and using a very low LR ≈ 1e-5) can further improve accuracy after the head converges.

---

## Architecture Details

| Component | Output dim | Trainable params |
|-----------|-----------|-----------------|
| ResNet50 backbone | 2048 | 0 (frozen) |
| Dense(512) | 512 | 1 049 088 |
| Dropout(0.4) | 512 | — |
| Dense(6) | 6 | 3 078 |

Total trainable: ~1.05 M (vs. 25.6 M in the full ResNet50).

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone weights | ImageNet | Standard pretrained checkpoint |
| Pooling | avg | Global Average Pooling → compact 2048-dim vector |
| Dense units | 512 | Provides capacity to remap ImageNet features to genres |
| Dropout | 0.4 | Regularises the head |
| L2 | 1e-4 | Small weight decay |
| Learning rate | 1e-3 (Adam) | Suitable for head-only training |
| Batch size | 16 | |
| Epochs | 6 (+ EarlyStopping, patience=3) | |

---

## Strengths

- **Massive transfer advantage** – ImageNet features include book cover relevant patterns (colour, texture, style).
- **Data-efficient** – only ~850 training images needed due to frozen backbone.
- **Dual purpose** – the frozen backbone is also reused as the image-feature extractor in the Fusion model, saving compute.

---

## Limitations

- **Large memory footprint** – ResNet50 backbone requires ~100 MB of GPU memory.
- **Frozen features** – may not optimally represent genre-specific visual patterns without fine-tuning.
- **No text signals** – still blind to description content.

---

## Expected Results

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~0.58 | ~0.55 |
| Macro F1 | ~0.56 | ~0.53 |

> Transfer learning gives a significant (~10 pp) accuracy boost over the from-scratch CNN.

---

## How to Run

```bash
python models/resnet50.py
```

---

## Artefacts Produced

| Artefact | Description |
|----------|-------------|
| `resnet_model` | Full Keras model (head + backbone) |
| `base_resnet` | Isolated frozen backbone (reused by fusion model) |
| Training history plots | Loss and accuracy curves |
| Confusion matrix plot | Displayed via `matplotlib` |
| Classification report | Printed to stdout |
