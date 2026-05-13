"""
custom_cnn.py
=============
Model 4 – Image Modality
Custom 4-Block Convolutional Neural Network (trained from scratch)

Modality  : Image only
Inputs    : book cover image resized to 224×224 RGB
Output    : genre label (6 classes)

Design rationale
----------------
- Four convolutional blocks with progressively deeper filters
  (32 → 64 → 128 → 256) to capture increasingly abstract visual features.
- Batch Normalisation after each convolution stabilises training.
- Global Average Pooling (instead of Flatten) reduces the parameter count
  and acts as a structural regulariser.
- Light augmentation (rotation ±10°, horizontal flip, small shifts)
  compensates for the limited dataset size (~850 training images).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from src.preprocessing import (
    load_and_prepare_dataframe, split_dataframe,
    print_report, plot_confusion, plot_history,
    SEED, CSV_PATH, EPOCHS_IMAGE,
)
from src.image_models import build_custom_cnn, make_cnn_generators


def train_custom_cnn(train_df, val_df, test_df,
                     y_test, label_encoder, num_classes: int):
    """
    Full training and evaluation pipeline for the custom CNN.

    Returns
    -------
    cnn_model : trained Keras model
    history   : Keras History object
    acc_cnn   : test accuracy (float)
    cm_cnn    : confusion matrix (ndarray)
    """
    # ── Data generators ───────────────────────────────────────────────────
    train_gen, val_gen, test_gen = make_cnn_generators(train_df, val_df, test_df)

    # ── Model ─────────────────────────────────────────────────────────────
    cnn_model = build_custom_cnn(num_classes)
    cnn_model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    cnn_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=1, min_lr=1e-6, verbose=1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────
    history = cnn_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_IMAGE,
        callbacks=cnn_callbacks,
        verbose=1,
    )
    plot_history(history, title="Custom CNN")

    # ── Test ──────────────────────────────────────────────────────────────
    probs  = cnn_model.predict(test_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc_cnn, cm_cnn = print_report(
        y_test, y_pred, label_encoder,
        title="Custom CNN [Test]",
    )
    plot_confusion(cm_cnn, label_encoder.classes_,
                   title="Custom CNN – Confusion Matrix (Test)")

    return cnn_model, history, acc_cnn, cm_cnn


# ─────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tf.random.set_seed(SEED)

    df, label_encoder = load_and_prepare_dataframe(CSV_PATH)
    num_classes = len(label_encoder.classes_)

    train_df, val_df, test_df, y_train, y_val, y_test = split_dataframe(df)

    model, history, acc, cm = train_custom_cnn(
        train_df, val_df, test_df,
        y_test, label_encoder, num_classes,
    )
    print(f"\nFinal Test Accuracy: {acc * 100:.2f} %")
