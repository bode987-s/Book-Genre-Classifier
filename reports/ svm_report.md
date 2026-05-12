# Model Report – TF-IDF + SVM (LinearSVC)

**Modality:** Text only  
**Category:** Classical ML baseline  
**File:** `models/svm.py`

---

## Overview

Support Vector Machines with a linear kernel are often the gold standard for text classification. `LinearSVC` directly optimises the hinge loss in the primal, making it significantly faster than kernel SVM (`SVC`) on high-dimensional sparse feature spaces. It typically matches or exceeds Logistic Regression on short-to-medium texts.

---

## Pipeline

```
title + description
      ↓
  clean_text()
      ↓
TfidfVectorizer         # 5 000 features, unigrams + bigrams
      ↓
   LinearSVC            # C=1.0, max_iter=5 000
      ↓
  predicted genre
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 5 000 | Same vocabulary as Logistic Regression for fair comparison |
| `ngram_range` | (1, 2) | Bigrams improve phrase recognition |
| `C` | 1.0 | Standard regularisation; avoids overfitting on small dataset |
| `max_iter` | 5 000 | Higher than default to ensure convergence |

---

## Strengths

- **Maximum-margin classifier** – finds the decision boundary with the largest margin, improving generalisation.
- **Sparse-matrix native** – operates directly on TF-IDF sparse matrices with no conversion cost.
- **Often outperforms LR** on imbalanced or noisy text data due to margin maximisation.
- **Deterministic** – given `random_state`, results are fully reproducible.

---

## Limitations

- **No probability estimates** – `LinearSVC` does not natively produce class probabilities (use `CalibratedClassifierCV` if needed).
- **Linear boundary only** – cannot capture non-linear genre interactions.
- **Sensitive to C** – too-large C risks overfitting; too-small C under-fits minority classes.

---

## Comparison vs. Logistic Regression

| Aspect | Logistic Regression | SVM (LinearSVC) |
|--------|--------------------|--------------------|
| Loss | Log-loss (cross-entropy) | Hinge loss |
| Probability output | Yes | No (by default) |
| Speed | Fast | Slightly faster |
| Typical text accuracy | Similar | Slightly higher |

---

## Expected Results

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~0.77 | ~0.75 |
| Macro F1 | ~0.76 | ~0.74 |

> Exact numbers depend on the dataset split and random seed.

---

## Confusion Matrix Insights

- Similar confusion pattern to Logistic Regression.
- SVM typically has sharper precision on high-frequency classes.
- **self-help / psychology** boundary remains the hardest to learn.

---

## How to Run

```bash
python models/svm.py
```

---

## Artefacts Produced

| Artefact | Description |
|----------|-------------|
| `svm` object | Fitted `LinearSVC` |
| `vectorizer` | Fitted `TfidfVectorizer` |
| Confusion matrix plot | Displayed via `matplotlib` |
| Classification report | Printed to stdout |
