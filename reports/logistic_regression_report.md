# Model Report – TF-IDF + Logistic Regression

**Modality:** Text only  
**Category:** Classical ML baseline  
**File:** `models/logistic_regression.py`

---

## Overview

A simple but highly competitive baseline that converts book text (title + description) into a sparse TF-IDF feature matrix and classifies it with a multi-class Logistic Regression. Despite its simplicity, this model is a strong contender because genre signals in text are often explicit and linearly separable in TF-IDF space.

---

## Pipeline

```
title + description
      ↓
  clean_text()          # lowercase, strip HTML/URLs/punctuation
      ↓
TfidfVectorizer         # 5 000 features, unigrams + bigrams, sublinear_tf
      ↓
LogisticRegression      # C=5.0, lbfgs solver, max_iter=3 000
      ↓
  predicted genre
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 5 000 | Balances vocabulary coverage vs. dimensionality |
| `ngram_range` | (1, 2) | Bigrams capture phrases like "self improvement" |
| `sublinear_tf` | True | Reduces impact of very frequent terms |
| `C` | 5.0 | Moderate regularisation; higher C than default improves recall on minority classes |
| `solver` | lbfgs | Efficient for multi-class with dense class structure |
| `max_iter` | 3 000 | Sufficient for convergence on sparse TF-IDF inputs |

---

## Strengths

- **Fast to train** – seconds even on CPU.
- **Interpretable** – top TF-IDF coefficients per class reveal the most discriminative words.
- **No hyperparameter tuning overhead** – well-studied defaults transfer well to text classification.
- **Strong on genre-specific vocabulary** – categories like *psychology* and *self-help* have distinct lexicons.

---

## Limitations

- **Bag-of-words** – word order is ignored; "not good" and "good" are treated similarly.
- **No semantic generalisation** – synonyms and paraphrases map to different features.
- **No image information** – cover art is completely ignored.

---

## Expected Results

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~0.76 | ~0.74 |
| Macro F1 | ~0.75 | ~0.73 |

> Accuracy measures overall correct predictions, while Macro F1 evaluates balanced classification performance across all classes.
The validation and test scores are close, which suggests the model generalizes reasonably well and does not show strong overfitting.

---

## Confusion Matrix Insights

- **self-help vs. psychology** – the most frequent confusion pair; both categories share motivational language.
- **children vs. education** – occasional overlap on picture-book descriptions.
- **business and travel** – typically well-separated due to domain-specific vocabulary.

---

## How to Run

```bash
python models/logistic_regression.py
```

---

## Artefacts Produced

| Artefact | Description |
|----------|-------------|
| `log_reg` object | Fitted `LogisticRegression` (can be pickled with `joblib.dump`) |
| `vectorizer` | Fitted `TfidfVectorizer` (required for inference) |
| Confusion matrix plot | Displayed via `matplotlib` |
| Classification report | Printed to stdout |
