"""
resnet50.py
===========
Model 5 – Image Modality
ResNet50 Transfer Learning (frozen ImageNet backbone)

Modality  : Image only
Inputs    : book cover image resized to 224×224 RGB, ResNet-preprocessed
Output    : genre label (6 classes)

Transfer-learning strategy
--------------------------
The ResNet50 backbone pre-trained on ImageNet is used as a frozen feature
extractor.  Only the added classification head (Dense 512 → Dropout 0.4 →
Dense num_classes) is trained.  This is efficient on small datasets and
leverages deep visual representations learned from millions of images,
giving a significant accuracy boost over the from-scratch CNN.

The backbone is exposed as `base_resnet` so the fusion model can reuse it
as an image-feature extractor without re-loading weights.
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
from src.image_models import build_resnet50, make_resnet_generators


def train_resnet50(train_df, val_df, test_df,
                   y_test, label_encoder, num_classes: int):
    """
    Full training and evaluation pipeline for ResNet50 Transfer Learning.

    Returns
    -------
    resnet_model : trained Keras Sequential model
    base_resnet  : frozen ResNet50 backbone (reused by fusion model)
    history      : Keras History object
    acc_resnet   : test accuracy (float)
    cm_resnet    : confusion matrix (ndarray)
    """
    # ── Data generators ───────────────────────────────────────────────────
    train_gen, val_gen, test_gen = make_resnet_generators(train_df, val_df, test_df)

    # ── Model ─────────────────────────────────────────────────────────────
    resnet_model, base_resnet = build_resnet50(num_classes)
    resnet_model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    resnet_callbacks = [
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
    history = resnet_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_IMAGE,
        callbacks=resnet_callbacks,
        verbose=1,
    )
    plot_history(history, title="ResNet50 Transfer Learning")

    # ── Test ──────────────────────────────────────────────────────────────
    probs  = resnet_model.predict(test_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc_resnet, cm_resnet = print_report(
        y_test, y_pred, label_encoder,
        title="ResNet50 [Test]",
    )
    plot_confusion(cm_resnet, label_encoder.classes_,
                   title="ResNet50 – Confusion Matrix (Test)")

    return resnet_model, base_resnet, history, acc_resnet, cm_resnet


# ─────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tf.random.set_seed(SEED)

    df, label_encoder = load_and_prepare_dataframe(CSV_PATH)
    num_classes = len(label_encoder.classes_)

    train_df, val_df, test_df, y_train, y_val, y_test = split_dataframe(df)

    model, base_resnet, history, acc, cm = train_resnet50(
        train_df, val_df, test_df,
        y_test, label_encoder, num_classes,
    )
    print(f"\nFinal Test Accuracy: {acc * 100:.2f} %")
