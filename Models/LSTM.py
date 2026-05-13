"""
bilstm.py
=========
Model 3 – Text Modality
Embedding Layer → Bidirectional LSTM

Modality  : Text only
Inputs    : title + description → cleaned → integer-padded sequences (len=200)
Output    : genre label (6 classes)

The Bidirectional LSTM reads the sequence in both directions (left-to-right
and right-to-left), capturing long-range dependencies that bag-of-words
models miss.  SpatialDropout1D and L2 regularisation mitigate overfitting
on the ~850-sample training set.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

from src.preprocessing import (
    load_and_prepare_dataframe, split_dataframe,
    print_report, plot_confusion, plot_history,
    SEED, CSV_PATH, MAX_WORDS, MAX_LEN, EPOCHS_TEXT,
)
from src.text_models import build_padded_sequences


def build_bilstm(num_classes: int,
                 max_words: int = MAX_WORDS,
                 max_len: int = MAX_LEN) -> tf.keras.Model:
    """
    Construct and compile the BiLSTM model.

    Architecture
    ------------
    Embedding(MAX_WORDS, 128) → SpatialDropout1D(0.2)
    → Bidirectional(LSTM(64)) → Dropout(0.4)
    → Dense(64, relu, L2) → Dropout(0.3)
    → Dense(num_classes, softmax)
    """
    model = models.Sequential([
        layers.Embedding(input_dim=max_words, output_dim=128,
                         input_length=max_len),
        layers.SpatialDropout1D(0.2),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="BiLSTM")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_bilstm(train_df, val_df, test_df,
                 y_train, y_val, y_test,
                 label_encoder, num_classes: int):
    """
    Full training and evaluation pipeline for the BiLSTM model.

    Returns
    -------
    lstm_model : trained Keras model
    history    : Keras History object
    acc_lstm   : test accuracy (float)
    cm_lstm    : confusion matrix (ndarray)
    """
    # ── Tokenisation ─────────────────────────────────────────────────────
    tokenizer_keras, X_train_pad, X_val_pad, X_test_pad = build_padded_sequences(
        train_df["text_input"].values,
        val_df["text_input"].values,
        test_df["text_input"].values,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    lstm_model = build_bilstm(num_classes)
    lstm_model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    lstm_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=2,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=1, min_lr=1e-6, verbose=1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────
    history = lstm_model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=EPOCHS_TEXT,
        batch_size=32,
        callbacks=lstm_callbacks,
        verbose=1,
    )
    plot_history(history, title="BiLSTM")

    # ── Test ──────────────────────────────────────────────────────────────
    probs    = lstm_model.predict(X_test_pad, verbose=0)
    y_pred   = np.argmax(probs, axis=1)
    acc_lstm, cm_lstm = print_report(
        y_test, y_pred, label_encoder,
        title="Embedding + BiLSTM [Test]",
    )
    plot_confusion(cm_lstm, label_encoder.classes_,
                   title="BiLSTM – Confusion Matrix (Test)")

    return lstm_model, history, acc_lstm, cm_lstm


# ─────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tf.random.set_seed(SEED)

    df, label_encoder = load_and_prepare_dataframe(CSV_PATH)
    num_classes = len(label_encoder.classes_)

    train_df, val_df, test_df, y_train, y_val, y_test = split_dataframe(df)

    model, history, acc, cm = train_bilstm(
        train_df, val_df, test_df,
        y_train, y_val, y_test,
        label_encoder, num_classes,
    )
    print(f"\nFinal Test Accuracy: {acc * 100:.2f} %")
