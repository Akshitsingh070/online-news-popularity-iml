"""Live demo for the 10-minute presentation: score an article *before* it is published.

The model bundle produced by ``python -m src.run_all`` holds the Random Forest regressor
(predicts log shares, mapped back to shares) and classifier (probability of being above the
training median of 1 400 shares).

    python demo/predict_demo.py                     # 5 random real articles + a "safe" and a "viral" one
    python demo/predict_demo.py --index 42          # one specific row of the CSV (0-based)
    python demo/predict_demo.py --quintiles         # ranking quality on the held-out test set
    python demo/predict_demo.py --interactive       # describe your own article on the command line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data, preprocess  # noqa: E402

BUNDLE = config.MODEL_DIR / "demo_bundle.joblib"
if not BUNDLE.exists():
    BUNDLE = config.MODEL_DIR / "model_bundle.joblib"

# Features the interactive mode lets you edit, with the median article as the starting point.
EDITABLE = [
    "n_tokens_title",
    "n_tokens_content",
    "num_hrefs",
    "num_imgs",
    "num_videos",
    "num_keywords",
    "average_token_length",
    "global_sentiment_polarity",
    "kw_avg_avg",
    "self_reference_avg_sharess",
    "is_weekend",
]

CHANNELS = {
    "lifestyle": "data_channel_is_lifestyle",
    "entertainment": "data_channel_is_entertainment",
    "business": "data_channel_is_bus",
    "social": "data_channel_is_socmed",
    "tech": "data_channel_is_tech",
    "world": "data_channel_is_world",
}


def load_bundle() -> dict:
    if not BUNDLE.exists():
        raise SystemExit(
            f"{BUNDLE} not found - run `python -m src.run_all` first (it writes the bundle)."
        )
    return joblib.load(BUNDLE)


def score(bundle: dict, row: pd.Series) -> tuple[float, float]:
    """Return (predicted shares, P(popular)) for one article row."""
    x = row[bundle["feature_names"]].to_frame().T.astype(float)
    pred_shares = float(np.clip(np.expm1(bundle["rf_regressor"].predict(x)[0]), 1.0, None))
    prob_popular = float(bundle["rf_classifier"].predict_proba(x)[0, 1])
    return pred_shares, prob_popular


def report(bundle: dict, row: pd.Series, label: str = "") -> None:
    pred, prob = score(bundle, row)
    actual = row.get(config.TARGET, np.nan)
    thr = bundle["popularity_threshold"]
    verdict = "POPULAR" if prob >= 0.5 else "not popular"
    title = label or str(row.get("url", "")).rsplit("/", 1)[-1]
    print(f"\n  article      : {title[:60]}")
    print(f"  words        : {int(row['n_tokens_content']):>6}   "
          f"links: {int(row['num_hrefs']):>3}   images: {int(row['num_imgs']):>3}   "
          f"keywords: {int(row['num_keywords']):>2}")
    print(f"  channel      : {channel_of(row):<14} weekend: {int(row['is_weekend'])}   "
          f"kw_avg_avg: {row['kw_avg_avg']:,.0f}")
    print(f"  -> predicted : {pred:>9,.0f} shares      P(popular >= {thr:,.0f}) = {prob:.2f}  [{verdict}]")
    if not np.isnan(actual):
        err = pred - float(actual)
        print(f"  -> actual    : {float(actual):>9,.0f} shares      error {err:+,.0f} "
              f"({err / float(actual) * 100:+.0f}%)")


def channel_of(row: pd.Series) -> str:
    for name, col in CHANNELS.items():
        if row.get(col, 0) == 1:
            return name
    return "none"


def mode_examples(bundle: dict, df: pd.DataFrame, n: int = 5) -> None:
    rng = np.random.default_rng(7)
    print("=" * 78)
    print("Random real articles from the corpus")
    print("=" * 78)
    for i in rng.choice(len(df), size=n, replace=False):
        report(bundle, df.iloc[int(i)], label=f"row {int(i)}")
    print("\n" + "=" * 78)
    print("Two extremes the model was NOT trained to worship")
    print("=" * 78)
    report(bundle, df.loc[df[config.TARGET].idxmax()], label="the most-shared article (843k shares)")
    report(bundle, df.loc[(df[config.TARGET] - 1400).abs().idxmin()], label="the most typical article (~1.4k shares)")


def mode_index(bundle: dict, df: pd.DataFrame, idx: int) -> None:
    if not 0 <= idx < len(df):
        raise SystemExit(f"--index must be between 0 and {len(df) - 1}")
    report(bundle, df.iloc[idx], label=f"row {idx}")


def mode_quintiles(bundle: dict) -> None:
    """Does the ranking work even though the magnitudes do not? Scored on the test split."""
    from src import regression

    ds = preprocess.prepare(data.load_raw())
    tbl = regression.quintile_calibration(bundle["rf_regressor"], ds)
    print("\nTest-set ranking quality (Random Forest, held-out rows)")
    print(tbl.to_string(index=False))
    print(
        "\nIf the median climbs from Q1 to Q5 the model ranks articles usefully even though the\n"
        "absolute numbers in section 'predicted' are unreliable."
    )


def mode_interactive(bundle: dict, df: pd.DataFrame) -> None:
    base = df[bundle["feature_names"]].median(numeric_only=True).copy()
    base["weekday_is_monday"] = 0
    base["is_weekend"] = 0
    print("Interactive mode - press Enter to keep the median-article default.")
    print(f"Defaults: {int(base['n_tokens_content'])} words, {int(base['num_hrefs'])} links, "
          f"{int(base['num_imgs'])} images, {int(base['num_keywords'])} keywords, "
          f"kw_avg_avg {base['kw_avg_avg']:,.0f}\n")

    for col in EDITABLE:
        raw = input(f"  {col} [{base[col]:,.3f}]: ").strip()
        if raw:
            try:
                base[col] = float(raw)
            except ValueError:
                print(f"    '{raw}' is not a number - keeping default")

    ch = input(f"  channel {list(CHANNELS)} [none]: ").strip().lower()
    if ch in CHANNELS:
        for col in CHANNELS.values():
            base[col] = 0
        base[CHANNELS[ch]] = 1

    report(bundle, base, label="your article")
    print("\nNote: only the features above were changed; every other feature stays at the corpus median.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, help="score this row of the CSV (0-based)")
    ap.add_argument("--interactive", action="store_true", help="describe your own article")
    ap.add_argument("--quintiles", action="store_true", help="show ranking quality on the test set")
    ap.add_argument("--n", type=int, default=5, help="how many random articles to score")
    args = ap.parse_args()

    bundle = load_bundle()
    print(f"model bundle: {len(bundle['feature_names'])} features, trained on {bundle['trained_on_rows']:,} rows")
    print(f"popularity threshold: {bundle['popularity_threshold']:,.0f} shares "
          f"(training median)")

    df = data.load_raw()
    if args.index is not None:
        mode_index(bundle, df, args.index)
    elif args.interactive:
        mode_interactive(bundle, df)
    elif args.quintiles:
        mode_quintiles(bundle)
    else:
        mode_examples(bundle, df, args.n)
        mode_quintiles(bundle)


if __name__ == "__main__":
    main()
