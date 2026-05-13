"""
svm.py
======
Model 2 – Text Modality
TF-IDF (unigrams + bigrams) + SVM (LinearSVC)

Modality  : Text only
Inputs    : title + description → cleaned → TF-IDF (5 000 features)
Output    : genre label (6 classes)

LinearSVC is preferred over kernel SVC for large sparse feature spaces;
it optimises the same hinge-loss objective but scales to tens of thousands
of features efficiently.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import LinearSVC

from src.preprocessing import (
    load_and_prepare_dataframe, split_dataframe,
    print_report, plot_confusion, SEED, CSV_PATH,
)
from src.text_models import build_tfidf_features


def train_svm(train_df, val_df, test_df,
              y_train, y_val, y_test, label_encoder):
    """
    Full training and evaluation pipeline for TF-IDF + LinearSVC.

    Returns
    -------
    svm        : fitted LinearSVC
    vectorizer : fitted TfidfVectorizer
    acc_svm    : test accuracy (float)
    cm_svm     : confusion matrix (ndarray)
    """
    # ── Feature extraction ────────────────────────────────────────────────
    vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = build_tfidf_features(
        train_df["text_input"].values,
        val_df["text_input"].values,
        test_df["text_input"].values,
        max_features=5_000,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    # C=1.0 is a strong default for balanced datasets; increase for lower
    # regularisation if training accuracy is noticeably below val accuracy.
    svm = LinearSVC(C=1.0, max_iter=5_000, random_state=SEED)
    svm.fit(X_train_tfidf, y_train)

    # ── Validation ────────────────────────────────────────────────────────
    y_pred_val = svm.predict(X_val_tfidf)
    print("── Validation ──")
    print_report(y_val, y_pred_val, label_encoder,
                 title="TF-IDF + SVM [Val]")

    # ── Test ──────────────────────────────────────────────────────────────
    y_pred = svm.predict(X_test_tfidf)
    print("\n── Test ──")
    acc_svm, cm_svm = print_report(
        y_test, y_pred, label_encoder,
        title="TF-IDF + SVM [Test]",
    )
    plot_confusion(cm_svm, label_encoder.classes_,
                   title="SVM – Confusion Matrix (Test)")

    return svm, vectorizer, acc_svm, cm_svm


# ─────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, label_encoder = load_and_prepare_dataframe(CSV_PATH)
    train_df, val_df, test_df, y_train, y_val, y_test = split_dataframe(df)

    svm, vectorizer, acc, cm = train_svm(
        train_df, val_df, test_df,
        y_train, y_val, y_test,
        label_encoder,
    )
    print(f"\nFinal Test Accuracy: {acc * 100:.2f} %")
