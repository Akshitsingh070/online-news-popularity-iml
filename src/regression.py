"""Exp. 3, 5, 6, 7 — regression on log(shares), scored on the original share scale.

Every model here predicts log(shares). Predictions are mapped back with expm1 and clipped at
1 (no article can be shared a fractional number of times, and the data minimum is 1), so MAE is
reported in shares and R² is the honest "share of variance explained on the real scale".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from . import config
from .preprocess import Dataset


def inverse_log(y_log: np.ndarray) -> np.ndarray:
    """Back to shares on the real scale, clipped at the observed minimum of 1."""
    return np.clip(np.expm1(y_log), 1.0, None)


@dataclass
class RegressionResult:
    name: str
    mae: float
    rmse: float
    r2: float
    r2_log: float
    medae: float
    fit_seconds: float
    notes: str = ""
    model: object = None

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "MAE (shares)": round(self.mae, 1),
            "RMSE (shares)": round(self.rmse, 1),
            "R2 (share scale)": round(self.r2, 4),
            "R2 (log scale)": round(self.r2_log, 4),
            "median abs err": round(self.medae, 1),
            "fit (s)": round(self.fit_seconds, 1),
            "notes": self.notes,
        }


def evaluate(name: str, model, ds: Dataset, X_train, X_test, notes: str = "") -> RegressionResult:
    t0 = time.perf_counter()
    model.fit(X_train, ds.y_train_log)
    fit_s = time.perf_counter() - t0

    pred_raw = inverse_log(model.predict(X_test))
    return RegressionResult(
        name=name,
        mae=float(mean_absolute_error(ds.y_test_raw, pred_raw)),
        rmse=float(np.sqrt(mean_squared_error(ds.y_test_raw, pred_raw))),
        r2=float(r2_score(ds.y_test_raw, pred_raw)),
        r2_log=float(r2_score(ds.y_test_log, model.predict(X_test))),
        medae=float(np.median(np.abs(ds.y_test_raw - pred_raw))),
        fit_seconds=fit_s,
        notes=notes,
        model=model,
    )


def tune_knn_k(ds: Dataset, candidates=(5, 15, 35, 75)) -> tuple[int, pd.DataFrame]:
    """Choose k on a validation slice carved out of the *training* data only."""
    Xtr, Xva, ytr, yva = train_test_split(
        ds.X_train_scaled, ds.y_train_log, test_size=0.15,
        random_state=config.RANDOM_STATE, shuffle=True,
    )
    rows = []
    for k in candidates:
        m = KNeighborsRegressor(n_neighbors=k, n_jobs=-1).fit(Xtr, ytr)
        rows.append({"k": k, "val_MAE(log)": round(float(mean_absolute_error(yva, m.predict(Xva))), 4)})
    tbl = pd.DataFrame(rows).sort_values("val_MAE(log)").reset_index(drop=True)
    return int(tbl.iloc[0]["k"]), tbl


def run_all(ds: Dataset, knn_k: int = 25) -> tuple[pd.DataFrame, dict]:
    """Fit the four regression families; return a comparison table and the fitted models."""
    results = [
        evaluate(
            "Linear Regression", LinearRegression(), ds,
            ds.X_train_scaled, ds.X_test_scaled,
            notes="scaled features",
        ),
        evaluate(
            "Decision Tree", DecisionTreeRegressor(random_state=config.RANDOM_STATE, min_samples_leaf=5),
            ds, ds.X_train, ds.X_test,
            notes="min_samples_leaf=5",
        ),
        evaluate(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, max_features=0.5,
                random_state=config.RANDOM_STATE, n_jobs=-1,
            ),
            ds, ds.X_train, ds.X_test,
            notes="300 trees, min_samples_leaf=2",
        ),
        evaluate(
            f"KNN Regressor (k={knn_k})", KNeighborsRegressor(n_neighbors=knn_k, weights="distance", n_jobs=-1),
            ds, ds.X_train_scaled, ds.X_test_scaled,
            notes=f"k={knn_k} chosen on training-only validation",
        ),
    ]
    table = pd.DataFrame([r.as_row() for r in results])
    models = {r.name: r.model for r in results}
    return table, models


def fig_predicted_vs_actual(models: dict, ds: Dataset, path=None) -> str:
    import matplotlib.pyplot as plt

    names = list(models)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.4), sharey=True)
    for ax, name in zip(axes, names):
        m = models[name]
        X = ds.X_test_scaled if "Linear" in name or "KNN" in name else ds.X_test
        pred = inverse_log(m.predict(X))
        ax.scatter(ds.y_test_raw, pred, s=4, alpha=0.2, color="#4c72b0", edgecolors="none")
        lim = [1, np.percentile(ds.y_test_raw, 99.5)]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(title=name.replace(" Regressor", ""), xlabel="actual shares")
    axes[0].set_ylabel("predicted shares")
    fig.suptitle("Predicted vs actual shares (test set, log axes, 99.5th pct clipped)", fontsize=10)
    fig.tight_layout()
    path = path or config.FIG_DIR / "exp3_pred_vs_actual.png"
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    return str(path)


def log_vs_raw_comparison(ds: Dataset) -> pd.DataFrame:
    """Exp. 2 evidence: same model, same split, target raw shares vs log(shares).

    Scored on the original share scale in both cases, so the two rows are directly comparable.
    The raw-target model buys a slightly better R² by predicting the conditional mean of a
    heavy-tailed variable - it is terrible on the typical article (median absolute error) and
    still misses every viral hit. The log-target model is far better in the typical case, which
    is what a newsroom actually uses.
    """
    rows = []
    for label, y_train, y_test in (
        ("Linear Regression on raw shares", ds.y_train_raw, ds.y_test_raw),
        ("Linear Regression on log(shares)", ds.y_train_log, np.log(ds.y_test_raw)),
    ):
        m = LinearRegression().fit(ds.X_train_scaled, y_train)
        pred = m.predict(ds.X_test_scaled)
        # map back to shares: expm1 for the log-target model, identity for the raw-target one
        pred_raw = np.clip(np.expm1(pred) if "log" in label else pred, 1.0, None)
        rows.append(
            {
                "target": label,
                "MAE (shares)": round(float(mean_absolute_error(ds.y_test_raw, pred_raw)), 1),
                "median abs err (shares)": round(float(np.median(np.abs(ds.y_test_raw - pred_raw))), 1),
                "R2 (share scale)": round(float(r2_score(ds.y_test_raw, pred_raw)), 4),
                "R2 (log scale)": round(float(r2_score(ds.y_test_log, np.log(pred_raw))), 4),
            }
        )
    return pd.DataFrame(rows)


def median_baseline(ds: Dataset) -> dict:
    """The 'model' a newsroom already has: always predict the training median.

    Any regression model must beat this on MAE to be worth deploying; its R² is ~0 by
    construction, which is exactly why R² on the raw share scale is a hard metric here.
    """
    const = float(np.median(ds.y_train_raw))
    pred = np.full_like(ds.y_test_raw, const)
    return {
        "model": "Baseline: training median",
        "MAE (shares)": round(float(mean_absolute_error(ds.y_test_raw, pred)), 1),
        "RMSE (shares)": round(float(np.sqrt(mean_squared_error(ds.y_test_raw, pred))), 1),
        "R2 (share scale)": round(float(r2_score(ds.y_test_raw, pred)), 4),
        "R2 (log scale)": round(float(r2_score(ds.y_test_log, np.log(pred))), 4),
        "median abs err": round(float(np.median(np.abs(ds.y_test_raw - pred))), 1),
        "fit (s)": 0.0,
        "notes": f"constant {const:.0f} shares",
    }


def quintile_calibration(model, ds: Dataset, n_bins: int = 5) -> pd.DataFrame:
    """Ranking quality: sort test articles by prediction, look at what actually happened.

    This is the practical test for a newsroom. Absolute share predictions are hopeless
    (R² ~ 0 on the raw scale), but if median actual shares climb monotonically across predicted
    quintiles, the model can still triage a content calendar.
    """
    pred = inverse_log(model.predict(ds.X_test))
    q = pd.qcut(pred, n_bins, labels=[f"Q{i + 1}" for i in range(n_bins)])
    tbl = (
        pd.DataFrame({"predicted_quintile": q, "predicted_shares": pred, "actual_shares": ds.y_test_raw})
        .groupby("predicted_quintile", observed=True)
        .agg(
            articles=("actual_shares", "size"),
            predicted_median=("predicted_shares", "median"),
            actual_median=("actual_shares", "median"),
            actual_mean=("actual_shares", "mean"),
        )
        .round(0)
        .reset_index()
    )
    return tbl
