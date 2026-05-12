# Model Report – Embedding + Bidirectional LSTM

**Modality:** Text only  
**Category:** Deep learning (sequential)  
**File:** `models/bilstm.py`

---

## Overview

The BiLSTM model learns to classify genres from the raw sequence of words rather than a bag-of-words representation. By reading the text in both directions simultaneously, it captures contextual dependencies that span across the entire description — something TF-IDF models cannot do.

---

## Pipeline

```
title + description
      ↓
  clean_text()
      ↓
 Keras Tokenizer        # vocabulary of 20 000 tokens, <OOV> for unknowns
      ↓
 pad_sequences          # fixed length = 200 tokens (post-pad / post-truncate)
      ↓
  Embedding(20000, 128) # learned word vectors (not pretrained)
      ↓
SpatialDropout1D(0.2)   # drops entire feature maps to regularise embeddings
      ↓
Bidirectional LSTM(64)  # reads sequence left→right and right→left
      ↓
   Dropout(0.4)
      ↓
 Dense(64, relu, L2)
      ↓
   Dropout(0.3)
      ↓
Dense(6, softmax)
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MAX_WORDS` | 20 000 | Captures most English vocabulary in book descriptions |
| `MAX_LEN` | 200 | Covers typical description length without wasted padding |
| Embedding dim | 128 | Sufficient for 6-class genre semantics |
| LSTM units | 64 | Balances capacity vs. overfitting on ~850 training samples |
| `SpatialDropout1D` | 0.2 | Regularises the embedding layer |
| `Dropout` | 0.4 / 0.3 | Prevents memorisation of training examples |
| L2 regularisation | 1e-4 | Small weight decay in the dense layer |
| Batch size | 32 | Provides noisy but informative gradient estimates |
| Epochs | 6 (+ EarlyStopping) | Avoids overfitting; best weights restored |
| Patience | 2 | Stops quickly if val_loss stagnates |

---

## Training Callbacks

| Callback | Setting | Effect |
|----------|---------|--------|
| `EarlyStopping` | patience=2, restore_best_weights=True | Prevents overfitting |
| `ReduceLROnPlateau` | factor=0.5, patience=1, min_lr=1e-6 | Fine-tunes learning rate when progress stalls |

---

## Strengths

- **Captures word order** – genre-relevant phrases and sentence structures are modelled.
- **Bidirectional context** – each word is influenced by both preceding and following words.
- **Learns task-specific embeddings** – unlike TF-IDF, embeddings are fine-tuned for genre classification.

---

## Limitations

- **Slow to train** on CPU — LSTM is sequential by nature (no parallelism within a sequence).
- **Requires more data** to outperform TF-IDF baselines consistently.
- **No pretrained embeddings** – using GloVe or FastText could improve performance significantly.
- **Still text-only** – ignores cover image signals.

---

## Expected Results

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | ~0.72 | ~0.70 |
| Macro F1 | ~0.71 | ~0.69 |

> On small datasets, TF-IDF baselines sometimes outperform LSTM. Adding pretrained embeddings (GloVe/FastText) typically closes this gap.

---

## How to Run

```bash
python models/bilstm.py
```

---

## Artefacts Produced

| Artefact | Description |
|----------|-------------|
| `lstm_model` | Trained Keras model (saveable with `model.save()`) |
| `tokenizer_keras` | Fitted Keras Tokenizer (required for inference) |
| Training history plots | Loss and accuracy curves |
| Confusion matrix plot | Displayed via `matplotlib` |
| Classification report | Printed to stdout |
