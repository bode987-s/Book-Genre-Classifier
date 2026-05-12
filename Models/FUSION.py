"""
fusion_rf.py
============
Model 6 – Fusion Modality
TF-IDF (SVD-300) + ResNet50 Features (2048) → Random Forest

Modality  : Text + Image (multimodal)
Inputs    : TF-IDF sparse text features  →  SVD-reduced to 300 dims
            ResNet50 image embeddings    →  2048 dims (from frozen backbone)
            Concatenated feature vector  →  2348 dims
Output    : genre label (6 classes)

Design rationale
----------------
Random Forest is chosen as the fusion classifier because:
  - It handles high-dimensional tabular features without normalisation.
  - It is robust to feature scale differences between text (SVD) and
    image (ResNet) embeddings.
  - It provides feature importance rankings for interpretability.
  - It is deterministic (random_state) and parallelisable (n_jobs=-1).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from src.preprocessing import (
    load_and_prepare_dataframe, split_dataframe,
    print_report, plot_confusion,
    SEED, CSV_PATH,
)
from src.text_models import build_tfidf_features
from src.image_models import build_resnet50, make_resnet_generators
from src.fusion import build_fusion_features, build_random_forest


def train_fusion_rf(train_df, val_df, test_df,
                    y_train, y_val, y_test, label_encoder, num_classes: int):
    """
    Full training and evaluation pipeline for the Fusion (RF) model.

    Pipeline steps
    --------------
    1. TF-IDF feature extraction (fit on train only)
    2. Truncated SVD → 300-dim dense text vectors
    3. ResNet50 backbone feature extraction → 2048-dim image vectors
    4. Horizontal concatenation → 2348-dim joint feature matrix
    5. Random Forest trained on the joint features

    Returns
    -------
    rf          : fitted RandomForestClassifier
    acc_rf      : test accuracy (float)
    cm_rf       : confusion matrix (ndarray)
    """
    # ── Step 1: TF-IDF ────────────────────────────────────────────────────
    vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = build_tfidf_features(
        train_df["text_input"].values,
        val_df["text_input"].values,
        test_df["text_input"].values,
        max_features=5_000,
    )

    # ── Steps 2–4: Build fusion features (SVD + ResNet) ───────────────────
    # We need the frozen ResNet50 backbone.  Build the transfer model and
    # reuse its backbone as a feature extractor – no additional training.
    _, base_resnet = build_resnet50(num_classes)
    # (The classification head is discarded; only base_resnet is used below.)

    X_train_fusion, X_val_fusion, X_test_fusion, svd, feat_extractor = \
        build_fusion_features(
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            train_df, val_df, test_df,
            base_resnet,
            n_svd_components=300,
        )

    # ── Step 5: Random Forest ─────────────────────────────────────────────
    print("Training Random Forest …")
    rf = build_random_forest(n_estimators=300)
    rf.fit(X_train_fusion, y_train)

    # Validation
    rf_pred_val = rf.predict(X_val_fusion)
    print("── Validation ──")
    print_report(y_val, rf_pred_val, label_encoder, title="Fusion RF [Val]")

    # Test
    rf_pred = rf.predict(X_test_fusion)
    print("\n── Test ──")
    acc_rf, cm_rf = print_report(
        y_test, rf_pred, label_encoder,
        title="Fusion: TF-IDF + ResNet → Random Forest [Test]",
    )
    plot_confusion(cm_rf, label_encoder.classes_,
                   title="Fusion RF – Confusion Matrix (Test)")

    return rf, acc_rf, cm_rf


# ─────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tf.random.set_seed(SEED)

    df, label_encoder = load_and_prepare_dataframe(CSV_PATH)
    num_classes = len(label_encoder.classes_)

    train_df, val_df, test_df, y_train, y_val, y_test = split_dataframe(df)

    rf, acc, cm = train_fusion_rf(
        train_df, val_df, test_df,
        y_train, y_val, y_test,
        label_encoder, num_classes,
    )
    print(f"\nFinal Test Accuracy: {acc * 100:.2f} %")
