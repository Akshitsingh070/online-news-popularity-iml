"""Build a compact model bundle for the live demo.

The full pipeline saves 300-tree, fully-grown forests (~520 MB) which is right for the report but
absurd to ship or to load in front of an audience. This script trains a *lighter twin* with the same
feature pipeline and the same training rows, reports how much accuracy it gives up, and writes
``models/demo_bundle.joblib`` (a few MB, committed nowhere - regenerated on demand).

    python tools/make_demo_bundle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data, preprocess, regression  # noqa: E402


def main() -> None:
    df = data.load_raw()
    ds = preprocess.prepare(df)
    y_train_bin, y_test_bin = ds.binary_labels()

    reg = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=10, max_features=0.5,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    ).fit(ds.X_train, ds.y_train_log)
    clf = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=10, max_features=0.5,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    ).fit(ds.X_train, y_train_bin)

    pred_shares = regression.inverse_log(reg.predict(ds.X_test))
    mae = mean_absolute_error(ds.y_test_raw, pred_shares)
    r2 = r2_score(ds.y_test_raw, pred_shares)
    f1 = f1_score(y_test_bin, clf.predict(ds.X_test))
    print(f"demo twin   : MAE {mae:,.0f} shares | R2 (share scale) {r2:.4f} | F1 {f1:.4f}")
    print("full model  : MAE 2,317 shares | R2 (share scale) 0.0033 | F1 0.6971  (see reports/results/)")

    bundle = {
        "feature_names": ds.feature_names,
        "popularity_threshold": ds.popularity_threshold,
        "rf_regressor": reg,
        "rf_classifier": clf,
        "trained_on_rows": int(len(ds.X_train)),
        "note": "lightweight demo twin (150 trees, min_samples_leaf=10); see reports/report.md for the full models",
        "demo_metrics": {"MAE": round(float(mae), 1), "R2_share_scale": round(float(r2), 4), "F1": round(float(f1), 4)},
    }
    out = config.MODEL_DIR / "demo_bundle.joblib"
    joblib.dump(bundle, out, compress=3)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out.relative_to(ROOT)} ({size_mb:.1f} MB, compressed)")


if __name__ == "__main__":
    main()
