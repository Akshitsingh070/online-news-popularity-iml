"""Exp. 8 — interpretation: which features actually carry the signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .preprocess import Dataset

# Features whose names encode the article's *own* history on Mashable rather than its content.
HISTORY_PREFIXES = ("kw_", "self_reference_")


def linear_coefficients(model, ds: Dataset, top: int = 15) -> pd.DataFrame:
    """Standardised linear coefficients: change in log(shares) per 1 SD of the feature.

    Note the trap: the features are standardised on *unscaled* one-hot columns, so a 0/1 channel
    flag moves a small fraction of a standard deviation - coefficients for those are not
    comparable in magnitude with continuous features. Rank is read with that in mind.
    """
    coefs = pd.Series(np.ravel(model.coef_), index=ds.feature_names)
    tbl = (
        coefs.rename("coef_log_shares")
        .to_frame()
        .assign(abs_coef=lambda d: d["coef_log_shares"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(top)
        .drop(columns="abs_coef")
        .reset_index(names="feature")
    )
    return tbl


def forest_importances(model, ds: Dataset, top: int = 15) -> pd.DataFrame:
    """Impurity-based importances from a fitted tree ensemble (fast, but biased towards
    high-cardinality continuous features)."""
    imp = pd.Series(model.feature_importances_, index=ds.feature_names)
    return (
        imp.rename("importance")
        .to_frame()
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(names="feature")
    )


def permutation_table(model, ds: Dataset, n_repeats: int = 5, top: int = 15) -> pd.DataFrame:
    """Permutation importance on the *test* set: drop in R² when a feature is shuffled.

    This is the honest counterpart to impurity importance because it measures out-of-sample
    usefulness, not how often a split used the column.
    """
    use_scaled = "Linear" in type(model).__name__ or "KNN" in type(model).__name__
    X = ds.X_test_scaled if use_scaled else ds.X_test
    r = permutation_importance(
        model, X, ds.y_test_log, n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, scoring="r2", n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": ds.feature_names,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def group_feature_importance(model, ds: Dataset) -> pd.DataFrame:
    """Coarse view used in the report: content vs links vs keywords vs timing."""
    imp = pd.Series(model.feature_importances_, index=ds.feature_names)

    def group(f: str) -> str:
        if f.startswith("kw_"):
            return "keywords (kw_*)"
        if f.startswith("self_reference_"):
            return "self-reference shares"
        if f.startswith(("weekday_", "is_weekend")):
            return "publishing day"
        if f.startswith("data_channel_"):
            return "channel"
        if f.startswith("LDA_"):
            return "LDA topics"
        if f.startswith(("global_", "title_")):
            return "title / subjectivity / polarity"
        return "content & links"

    out = (
        imp.groupby(imp.index.map(group))
        .sum()
        .sort_values(ascending=False)
        .rename("share_of_importance")
        .to_frame()
    )
    out["share_of_importance"] = (out["share_of_importance"] * 100).round(1)
    return out.reset_index(names="feature_group")


def fig_importance(perm: pd.DataFrame, path=None) -> str:
    import matplotlib.pyplot as plt

    d = perm.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#4c72b0")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(
        title="Exp. 8 — permutation importance on test R² (Random Forest)",
        xlabel="drop in R² when shuffled",
    )
    fig.tight_layout()
    path = path or config.FIG_DIR / "exp8_permutation_importance.png"
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    return str(path)


def fig_group_importance(grouped: pd.DataFrame, path=None) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(grouped["feature_group"], grouped["share_of_importance"], color="#55a868")
    ax.invert_yaxis()
    ax.set(title="Where the Random Forest's signal comes from", xlabel="% of total importance")
    fig.tight_layout()
    path = path or config.FIG_DIR / "exp8_group_importance.png"
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    return str(path)


def history_vs_content_split(grouped: pd.DataFrame) -> pd.DataFrame:
    """How much of the model's signal is Mashable's own past performance, not the article."""
    hist = grouped.loc[grouped["feature_group"].isin(["keywords (kw_*)", "self-reference shares"]),
                       "share_of_importance"].sum()
    return pd.DataFrame(
        {
            "bucket": ["article history on Mashable (kw_*, self_reference_*)",
                       "article content, links, topics, timing"],
            "share_of_importance_%": [round(float(hist), 1), round(100 - float(hist), 1)],
        }
    )
