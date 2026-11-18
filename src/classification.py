"""Exp. 4, 5, 6, 7, 8 — classification: popular (>= training median) vs not.

Labels come from ``Dataset.binary_labels()``: the threshold is the *training* median of shares,
applied unchanged to the test rows. The split is balanced by construction (~50/50), so accuracy is
not a misleading metric for once, but F1 on the popular class is what the brief asks for and is
reported first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config
from .preprocess import Dataset


@dataclass
class ClassificationResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fit_seconds: float
    notes: str = ""
    model: object = None
    y_pred: np.ndarray | None = None
    y_score: np.ndarray | None = None

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "F1 (popular)": round(self.f1, 4),
            "ROC-AUC": round(self.roc_auc, 4),
            "fit (s)": round(self.fit_seconds, 1),
            "notes": self.notes,
        }


def evaluate(name: str, model, ds: Dataset, X_train, X_test, notes: str = "",
             y_train=None) -> ClassificationResult:
    y_train_full, y_test = ds.binary_labels()
    y_train = y_train_full if y_train is None else y_train
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:  # SVC(probability=False) exposes decision_function instead
        y_score = model.decision_function(X_test)

    return ClassificationResult(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_score)),
        fit_seconds=fit_s,
        notes=notes,
        model=model,
        y_pred=y_pred,
        y_score=y_score,
    )


def run_all(ds: Dataset, knn_k: int = 25, svm_cap: int | None = 15000) -> tuple[pd.DataFrame, dict]:
    """Fit the five classification families.

    ``svm_cap``: an RBF SVM is roughly O(n^2..n^3) in the number of training rows. If the cap is
    smaller than the training set, the SVM sees a stratified subsample and the row says so -
    the alternative (silently training a 31k-row RBF SVM for half an hour) would be worse.
    """
    y_train, _ = ds.binary_labels()

    svm_notes = "RBF kernel, C=1, gamma='scale'"
    X_svm, y_svm = ds.X_train_scaled, y_train
    if svm_cap is not None and svm_cap < len(y_train):
        rng = np.random.default_rng(config.RANDOM_STATE)
        pos = np.flatnonzero(y_train == 1)
        neg = np.flatnonzero(y_train == 0)
        n_pos = int(round(svm_cap * len(pos) / len(y_train)))
        pick = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, svm_cap - n_pos, replace=False)]
        )
        X_svm, y_svm = ds.X_train_scaled[pick], y_train[pick]
        svm_notes += f"; trained on a stratified {svm_cap:,}-row subsample"

    results = [
        evaluate(
            "Logistic Regression",
            LogisticRegression(max_iter=2000, C=1.0, random_state=config.RANDOM_STATE),
            ds, ds.X_train_scaled, ds.X_test_scaled, notes="scaled features, C=1",
        ),
        evaluate(
            "Decision Tree",
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=config.RANDOM_STATE),
            ds, ds.X_train, ds.X_test, notes="max_depth=8, min_samples_leaf=20",
        ),
        evaluate(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, max_features=0.5,
                random_state=config.RANDOM_STATE, n_jobs=-1,
            ),
            ds, ds.X_train, ds.X_test, notes="300 trees, min_samples_leaf=2",
        ),
        evaluate(
            f"KNN (k={knn_k})",
            KNeighborsClassifier(n_neighbors=knn_k, weights="distance", n_jobs=-1),
            ds, ds.X_train_scaled, ds.X_test_scaled, notes=f"k={knn_k}, distance weights",
        ),
        evaluate(
            "SVM (RBF)", SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000),
            ds, X_svm, ds.X_test_scaled, notes=svm_notes, y_train=y_svm,
        ),
    ]
    table = pd.DataFrame([r.as_row() for r in results])
    return table, {r.name: r for r in results}


def confusion(results: dict, ds: Dataset, name: str) -> pd.DataFrame:
    _, y_test = ds.binary_labels()
    cm = confusion_matrix(y_test, results[name].y_pred)
    return pd.DataFrame(
        cm, index=["actual: not popular", "actual: popular"],
        columns=["pred: not popular", "pred: popular"],
    )
