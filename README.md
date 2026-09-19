# 📚 Multimodal Book Genre Classification

> Deep Learning & Neural Networks – Fall 2026

A complete multimodal classification pipeline that predicts book genres from **cover images** and **textual descriptions**, combining six models across three modalities.

---

# Multimodal Book Classification
Pipeline Architecture

TEXT
├── Clean Text
├── TF-IDF 
│   ├── Logistic Regression 
│   └── SVM 
└── Tokenization → Embedding → LSTM 
IMAGE
├── Resize (224×224)
├── CNN 
└── ResNet50 

FUSION 
├── Text Features (TF-IDF)
├── Image Features (CNN + ResNet)
└── Random Forest

EVALUATION 
├── Accuracy
├── Precision / Recall / F1
└── Confusion Matrix

## 🎯 Task

**Multi-class book genre classification** across 6 categories:

| Genre | Description |
|-------|-------------|
| `business` | Finance, management, entrepreneurship |
| `children` | Children's fiction and non-fiction |
| `education` | Textbooks, academic, pedagogy |
| `psychology` | Behavioral science, self-understanding |
| `self-help` | Personal development, motivation |
| `travel` | Travel guides, memoirs, geography |

---

## 🧠 Models Overview

### Text Modality
| Model | Key Idea |
|-------|----------|
| **TF-IDF + Logistic Regression** | Sparse bag-of-words with a fast linear classifier |
| **TF-IDF + SVM (LinearSVC)** | Maximum-margin classifier on TF-IDF features |
| **Embedding + BiLSTM** | Sequential deep model capturing bidirectional context |

### Image Modality
| Model | Key Idea |
|-------|----------|
| **Custom CNN** | 4-block convolutional network trained from scratch |
| **ResNet50 (Transfer)** | Frozen ImageNet backbone + fine-tuned classification head |

### Fusion Modality
| Model | Key Idea |
|-------|----------|
| **TF-IDF + ResNet → Random Forest** | SVD-reduced text features concatenated with ResNet image embeddings |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set dataset paths
Edit `src/preprocessing.py` and update:
```python
DATA_DIR  = "path/to/dataset"
CSV_PATH  = os.path.join(DATA_DIR, "books_dataset.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
```

### 3. Run the full pipeline
```bash
jupyter notebook notebooks/full_project.ipynb
```

Or run individual model files:
```bash
python models/logistic_regression.py
python models/bilstm.py
python models/resnet50.py
python models/fusion_rf.py
```

---

## 📊 Data Split Strategy

| Split | Proportion | Purpose |
|-------|-----------|---------|
| **Train** | ~72% | Fit all models |
| **Validation** | ~8% | Early stopping & hyperparameter tuning |
| **Test** | 20% | Final unbiased evaluation |

Splits are **stratified** to preserve class balance at every stage.

---

## 🔁 Pipeline Architecture

```
INPUT
├── Book Description + Title  ──► clean_text() ──► TF-IDF ──┐
│                                                              ├──► LR / SVM
│                                                              │
│                               Tokenize ──► Embedding ──► BiLSTM
│
└── Book Cover Image (224×224) ──► rescale ──► Custom CNN
                                ──► ResNet preprocess ──► ResNet50 (frozen)

FUSION
  TF-IDF (SVD-300) + ResNet features (2048) ──► concat ──► Random Forest
```

---

## 📋 Evaluation Metrics

- **Accuracy** (primary)
- **Precision / Recall / F1-score** (per class, macro, weighted)
- **Confusion Matrix**
- **Training & Validation curves** (loss + accuracy per epoch)

---

## 📦 Dataset Source

Book metadata and cover images scraped from the [Open Library API](https://openlibrary.org/developers/api).  
Up to **200 books per category** were collected (~1,200 total).

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Data wrangling | `pandas`, `numpy` |
| ML models | `scikit-learn` |
| Deep learning | `tensorflow` / `keras` |
| Image processing | `Pillow` |
| Visualization | `matplotlib`, `seaborn` |
| Dimensionality reduction | `TruncatedSVD` |
