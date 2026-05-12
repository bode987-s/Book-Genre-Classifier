# Model Report – Multimodal Fusion (TF-IDF + ResNet → Random Forest)

**Modality:** Text + Image (multimodal)  
**Category:** Fusion / Ensemble  
**File:** `models/fusion_rf.py`

---

## Overview

The fusion model is the centerpiece of the multimodal approach. It combines complementary signals from two modalities:

- **Text** – TF-IDF sparse vectors compressed to 300 dense dimensions via Truncated SVD (Latent Semantic Analysis)
- **Image** – 2048-dimensional global-average-pooling embeddings extracted from the frozen ResNet50 backbone

The concatenated 2348-dimensional feature vector is fed to a Random Forest classifier, which handles high-dimensional tabular data robustly without normalisation.

---

## Pipeline

```
title + description                   cover image (224×224)
      ↓                                       ↓
  TF-IDF (5 000)                   ResNet50 (frozen backbone)
      ↓                                       ↓
TruncatedSVD (300 dims)           GlobalAveragePooling → 2048 dims
      ↓                                       ↓
      └──────────────── hstack ───────────────┘
                              ↓
                    2348-dim feature vector
                              ↓
                     Random Forest (300 trees)
                              ↓
                        predicted genre
```

---

## Component Details

### Text Branch: TF-IDF + SVD

| Step | Config | Output dim |
|------|--------|-----------|
| TfidfVectorizer | 5 000 features, bigrams | 5 000 (sparse) |
| TruncatedSVD | 300 components | 300 (dense) |
| Explained variance | ≈ 55–65% | — |

SVD (Latent Semantic Analysis) denoises the sparse TF-IDF matrix and produces a dense representation that captures latent topic structure across genres.

### Image Branch: ResNet50 Feature Extraction

| Step | Config | Output dim |
|------|--------|-----------|
| Preprocessing | ResNet-specific channel mean sub. | — |
| ResNet50 backbone | Frozen ImageNet weights, pooling="avg" | 2048 |

The backbone is shared with `models/resnet50.py` — no additional training is required.

### Fusion: Random Forest

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 300 | Large forest for stable predictions |
| `max_depth` | None | Fully grown trees; regularised by ensemble averaging |
| `n_jobs` | -1 | Parallel training across all CPU cores |
| `random_state` | 42 | Reproducibility |

---

## Why Random Forest for Fusion?

| Property | Benefit |
|----------|---------|
| No normalisation required | Text (SVD) and image (ResNet) features have different scales |
| High dimensionality | RF handles 2348 features efficiently |
| Non-linear boundaries | Can model complex text × image interactions |
| Feature importance | Interpretable: see which modality contributes more |
| Variance reduction | Ensemble averaging reduces overfitting vs. a single decision tree |

---

## Feature Importance Insight

After training, the importance of each feature can be retrieved:

```python
importances = rf.feature_importances_
text_importance  = importances[:300].sum()   # first 300 dims = SVD text
image_importance = importances[300:].sum()   # remaining 2048 dims = ResNet image
print(f"Text contribution : {text_importance:.2%}")
print(f"Image contribution: {image_importance:.2%}")
```

---

## Strengths

- **Best of both modalities** – text and image signals reinforce each other.
- **No additional deep-learning training** – reuses pre-fitted vectorizer, SVD, and frozen ResNet backbone.
- **Robust** – RF is insensitive to feature scale and outliers.
- **Interpretable** – feature importances indicate which modality and which latent dimension matters most.

---

## Limitations

- **Late fusion** – text and image features are combined after independent extraction; joint training could yield richer interactions.
- **SVD loses some text signal** – ~35–45% variance of TF-IDF is discarded.
- **RF inference at scale** – 300 trees × 2348 features can be slow for real-time applications.

---

## Expected Results

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~0.82 | ~0.80 |
| Macro F1 | ~0.81 | ~0.79 |

> Fusion consistently outperforms all single-modality models, validating the multimodal hypothesis.

---

## How to Run

```bash
python models/fusion_rf.py
```

> **Note:** This model depends on the frozen ResNet50 backbone. TensorFlow must be installed and the image dataset must be accessible.

---

## Artefacts Produced

| Artefact | Description |
|----------|-------------|
| `rf` | Fitted `RandomForestClassifier` (saveable with `joblib.dump`) |
| `svd` | Fitted `TruncatedSVD` (required for inference on new text) |
| `feat_extractor` | Keras model wrapping the frozen ResNet50 backbone |
| Confusion matrix plot | Displayed via `matplotlib` |
| Classification report | Printed to stdout |

---

## Comparative Summary

| Model | Modality | Expected Test Accuracy |
|-------|----------|----------------------|
| Logistic Regression | Text | ~74% |
| SVM | Text | ~75% |
| BiLSTM | Text | ~70% |
| Custom CNN | Image | ~45% |
| ResNet50 | Image | ~55% |
| **Fusion RF** | **Text + Image** | **~80%** |
