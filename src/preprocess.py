"""Exp. 2 — preprocessing: what we drop, why we model log(shares), and how we split."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class Dataset:
    """Everything the modelling code needs, all derived from one leak-free split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    feature_names: list[str]
    popularity_threshold: float  # training-set median of shares
    _X_train_scaled: np.ndarray = field(repr=False)
    _X_test_scaled: np.ndarray = field(repr=False)

    @property
    def y_train_log(self) -> np.ndarray:
        return np.log(self.y_train_raw)

    @property
    def y_test_log(self) -> np.ndarray:
        return np.log(self.y_test_raw)

    # scaled copies are used by Linear Regression / KNN / SVM only; trees never touch them
    @property
    def X_train_scaled(self) -> np.ndarray:
        return self._X_train_scaled

    @property
    def X_test_scaled(self) -> np.ndarray:
        return self._X_test_scaled

    def binary_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """popular == shares >= *training* median (Exp. 4).

        The threshold comes from the training rows only and is applied unchanged to the test
        rows, so the test labels never see the test distribution.
        """
        thr = self.popularity_threshold
        return (self.y_train_raw >= thr).astype(int), (self.y_test_raw >= thr).astype(int)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop identifiers, separate the target.

    58 of the 61 columns survive: every column except ``url``, ``timedelta`` and ``shares``.
    """
    missing = [c for c in config.DROP_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns not found: {missing}")
    X = df.drop(columns=config.DROP_COLS + [config.TARGET])
    y = df[config.TARGET].astype(float)
    return X, y


def prepare(df: pd.DataFrame) -> Dataset:
    """One 80/20 split, log target, training median for the popular/not split."""
    X, y_raw = build_features(df)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y_raw.to_numpy(),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )

    # Fitted on the training rows only -> no information from the test part leaks in.
    scaler = StandardScaler().fit(X_train)

    return Dataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        feature_names=list(X.columns),
        popularity_threshold=float(np.median(y_train_raw)),
        _X_train_scaled=scaler.transform(X_train),
        _X_test_scaled=scaler.transform(X_test),
    )
