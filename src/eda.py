"""Exp. 1 — exploratory analysis of shares and of the features that track log(shares)."""

from __future__ import annotations

import matplotlib

# Head-less PNG output for the pipeline; inside Jupyter the inline backend is kept so figures
# show up in the notebook as usual.
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

CHANNEL_COLS = [
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
]


def shares_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quantiles + shape statistics: the numbers behind 'heavily skewed'."""
    s = df[config.TARGET]
    q = {f"p{p}": float(np.percentile(s, p)) for p in (1, 5, 25, 50, 75, 90, 95, 99, 99.9)}
    rows = [("count", int(s.size)), ("mean", round(float(s.mean()), 1)), ("std", round(float(s.std()), 1))]
    rows += [(k, round(v, 1)) for k, v in q.items()]
    rows += [
        ("max", int(s.max())),
        ("skew", round(float(s.skew()), 2)),
        ("kurtosis", round(float(s.kurt()), 1)),
        ("mean / median", round(float(s.mean() / s.median()), 2)),
        ("share of total shares in top 1% of articles", round(float(s.nlargest(int(len(s) * 0.01)).sum() / s.sum()), 3)),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def correlation_table(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Spearman and Pearson correlation of every feature with log(shares).

    Spearman is quoted first because the predictors are mostly counted, bounded or 0/1
    variables for which a rank correlation is the fairer summary.
    """
    y = np.log(df[config.TARGET].astype(float))
    X = df.drop(columns=config.DROP_COLS + [config.TARGET])
    tbl = pd.DataFrame(
        {
            "feature": X.columns,
            "spearman": [float(X[c].corr(y, method="spearman")) for c in X.columns],
            "pearson": [float(X[c].corr(y, method="pearson")) for c in X.columns],
        }
    )
    tbl["abs_spearman"] = tbl["spearman"].abs()
    return tbl.sort_values("abs_spearman", ascending=False).head(top).drop(columns="abs_spearman").reset_index(drop=True)


def channel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median shares per Mashable channel — the strongest *categorical* signal in the data."""
    tmp = df.copy()
    tmp["channel"] = np.select(
        [tmp[c] == 1 for c in CHANNEL_COLS],
        [c.replace("data_channel_is_", "") for c in CHANNEL_COLS],
        default="none",
    )
    out = (
        tmp.groupby("channel")[config.TARGET]
        .agg(articles="size", median_shares="median", mean_shares="mean")
        .assign(mean_shares=lambda d: d["mean_shares"].round(0))
        .sort_values("median_shares", ascending=False)
        .reset_index()
    )
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_shares_distribution(df: pd.DataFrame, path=None) -> str:
    s = df[config.TARGET]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    ax[0].hist(s, bins=80, color="#4c72b0")
    ax[0].set(title="shares (raw)", xlabel="shares", ylabel="articles")
    ax[0].set_xlim(0, np.percentile(s, 99.5))

    ax[1].hist(np.log(s), bins=60, color="#55a868")
    ax[1].set(title="log(shares)", xlabel="log(shares)", ylabel="articles")

    ax[2].boxplot(np.log(s), vert=False, widths=0.6)
    ax[2].set(title="log(shares), spread", xlabel="log(shares)", yticks=[])

    fig.suptitle(
        f"shares is heavy-tailed: skew {s.skew():.1f} raw vs {np.log(s).skew():.2f} after log",
        fontsize=10,
    )
    fig.tight_layout()
    path = path or config.FIG_DIR / "exp1_shares_distribution.png"
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    return str(path)


def fig_top_correlations(tbl: pd.DataFrame, path=None) -> str:
    d = tbl.sort_values("spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in d["spearman"]]
    ax.barh(d["feature"], d["spearman"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Exp. 1 — Spearman correlation with log(shares), top 20", xlabel="rho")
    fig.tight_layout()
    path = path or config.FIG_DIR / "exp1_top_correlations.png"
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    return str(path)


def fig_channel_box(df: pd.DataFrame, path=None) -> str:
    tmp = df.copy()
    tmp["channel"] = np.select(
        [tmp[c] == 1 for c in CHANNEL_COLS],
        [c.replace("data_channel_is_", "") for c in CHANNEL_COLS],
        default="none",
    )
    order = (
        tmp.groupby("channel")[config.TARGET].median().sort_values(ascending=False).index.tolist()
    )
    data = [np.log(tmp.loc[tmp["channel"] == c, config.TARGET]) for c in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=order, widths=0.6)
    ax.set(title="log(shares) by Mashable channel", ylabel="log(shares)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = path or config.FIG_DIR / "exp1_channel_box.png"
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    return str(path)
