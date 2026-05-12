"""
preprocessing.py
================
Data loading, text cleaning, image path resolution, label encoding,
and train / validation / test splitting.

All downstream models import from this module to guarantee a consistent
pipeline with no data leakage between splits.
"""

import os
import re
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths – update these to match your local setup ───────────────────────
DATA_DIR  = "/kaggle/input/datasets/xzt1abdelrahman/books-dataset/dataset"
CSV_PATH  = os.path.join(DATA_DIR, "books_dataset.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# ── Column names ─────────────────────────────────────────────────────────
TEXT_COL  = "description"
TITLE_COL = "title"
IMAGE_COL = "image_path"
LABEL_COL = "category"

# ── Model hyperparameters ─────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
MAX_WORDS  = 20_000   # TF-IDF / Tokenizer vocabulary size
MAX_LEN    = 200      # max token sequence length for LSTM

EPOCHS_TEXT  = 6
EPOCHS_IMAGE = 6


# ─────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs, remove punctuation, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>",          " ", text)   # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z0-9\s]",    " ", text)   # punctuation
    text = re.sub(r"\s+",             " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────

def ensure_image_path(path: str) -> str:
    """
    Resolve a stored image path to an absolute path under IMAGE_DIR.
    Handles both Unix and Windows path separators in the stored value.
    """
    if pd.isna(path):
        return ""
    filename  = os.path.basename(str(path).replace("\\", "/"))
    full_path = os.path.join(IMAGE_DIR, filename)
    return full_path


# ─────────────────────────────────────────────────────────────────────────
# DataFrame loading & preparation
# ─────────────────────────────────────────────────────────────────────────

def load_and_prepare_dataframe(csv_path: str = CSV_PATH):
    """
    Load the CSV, clean text, resolve image paths, and encode labels.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame with extra columns:
        - ``text_input``  : cleaned title + description
        - ``label``       : integer class label
    le : LabelEncoder
        Fitted label encoder; use ``le.classes_`` for class names.
    """
    df = pd.read_csv(csv_path)

    required = {TITLE_COL, TEXT_COL, IMAGE_COL, LABEL_COL}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df[TEXT_COL]  = df[TEXT_COL].fillna("")
    df[TITLE_COL] = df[TITLE_COL].fillna("")
    df[IMAGE_COL] = df[IMAGE_COL].apply(ensure_image_path)

    # Combine title + description → single text input
    df["text_input"] = (
        df[TITLE_COL].astype(str) + " " + df[TEXT_COL].astype(str)
    ).apply(clean_text)

    # Encode string labels → integers
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[LABEL_COL])

    return df, le


# ─────────────────────────────────────────────────────────────────────────
# Train / Validation / Test split
# ─────────────────────────────────────────────────────────────────────────

def split_dataframe(df: pd.DataFrame):
    """
    Stratified split:
      - Test  : 20 % of total
      - Val   :  8 % of total  (10 % of the remaining 80 %)
      - Train : 72 % of total

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame (indices reset)
    y_train, y_val, y_test    : np.ndarray of integer labels
    """
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    print(f"Train : {len(train_df):>5} rows")
    print(f"Val   : {len(val_df):>5} rows")
    print(f"Test  : {len(test_df):>5} rows")
    print(f"Total : {len(train_df) + len(val_df) + len(test_df):>5} rows")

    return train_df, val_df, test_df, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def print_report(y_true, y_pred, label_encoder, title="Model"):
    """Print accuracy + full classification report; return (accuracy, cm)."""
    acc    = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f} %)\n")
    print(
        classification_report(
            y_true, y_pred,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )
    cm_mat = confusion_matrix(y_true, y_pred)
    return acc, cm_mat


def plot_confusion(cm_mat, labels, title="Confusion Matrix"):
    """Display a colour-mapped confusion matrix with per-cell counts."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks);       ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(ticks);       ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title(title,              fontsize=13)

    thresh = cm_mat.max() / 2
    for i in range(cm_mat.shape[0]):
        for j in range(cm_mat.shape[1]):
            ax.text(
                j, i, format(cm_mat[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_mat[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.show()


def plot_history(history, title="Training History"):
    """Plot train vs. validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val  loss")
    axes[0].set_title(f"{title} – Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["accuracy"],     label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val  acc")
    axes[1].set_title(f"{title} – Accuracy")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────
# Script entry point – quick sanity check
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, le = load_and_prepare_dataframe()
    print(f"Loaded {len(df)} rows | {len(le.classes_)} classes: {list(le.classes_)}")
    train_df, val_df, test_df, y_train, y_val, y_test = split_dataframe(df)
