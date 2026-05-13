"""
logistic_regression.py
======================
Model 1 – Text Modality
TF-IDF (unigrams + bigrams) + Logistic Regression

Modality  : Text only
Inputs    : title + description → cleaned → TF-IDF (5 000 features)
Output    : genre label (6 classes)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    load_and_prepare_dataframe, split_dataframe,
    print_report, plot_confusion, SEED, CSV_PATH,
)
from src.text_models import build_tfidf_features


def train_logistic_regression(train_df, val_df, test_df,
                               y_train, y_val, y_test, label_encoder):
    """
    Full training and evaluation pipeline for TF-IDF + Logistic Regression.

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames with 'text_input' and 'label'
    y_train, y_val, y_test    : integer label arrays
    label_encoder             : fitted LabelEncoder

    Returns
    -------
    log_reg     : fitted LogisticRegression
    vectorizer  : fitted TfidfVectorizer
    acc_lr      : test accuracy (float)
    cm_lr       : confusion matrix (ndarray)
    """
    # ── Feature extraction ────────────────────────────────────────────────
    vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = build_tfidf_features(
        train_df["text_input"].values,
        val_df["text_input"].values,
        test_df["text_input"].values,
        max_features=5_000,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    log_reg = LogisticRegression(
        C=5.0,
        max_iter=3_000,
        solver="lbfgs",
        multi_class="auto",
        random_state=SEED,
    )
    log_reg.fit(X_train_tfidf, y_train)

    # ── Validation ────────────────────────────────────────────────────────
    y_pred_val = log_reg.predict(X_val_tfidf)
    print("── Validation ──")
    print_report(y_val, y_pred_val, label_encoder,
                 title="TF-IDF + Logistic Regression [Val]")

    # ── Test ──────────────────────────────────────────────────────────────
    y_pred = log_reg.predict(X_test_tfidf)
    print("\n── Test ──")
    acc_lr, cm_lr = print_report(
        y_test, y_pred, label_encoder,
        title="TF-IDF + Logistic Regression [Test]",
    )
    plot_confusion(cm_lr, label_encoder.classes_,
                   title="Logistic Regression – Confusion Matrix (Test)")

    return log_reg, vectorizer, acc_lr, cm_lr


# ─────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, label_encoder = load_and_prepare_dataframe(CSV_PATH)
    train_df, val_df, test_df, y_train, y_val, y_test = split_dataframe(df)

    log_reg, vectorizer, acc, cm = train_logistic_regression(
        train_df, val_df, test_df,
        y_train, y_val, y_test,
        label_encoder,
    )
    print(f"\nFinal Test Accuracy: {acc * 100:.2f} %")
