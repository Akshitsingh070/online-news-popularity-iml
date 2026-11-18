"""End-to-end pipeline for the P11 Online News Popularity project.

Run with:  python -m src.run_all

Writes every number and figure that the notebook and the report quote:
  reports/results/*.csv    tables (row counts, EDA, model comparisons, importance)
  reports/figures/*.png    figures
  models/model_bundle.joblib   Random Forest pair used by the live demo
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from . import classification, config, data, eda, interpret, preprocess, regression


def _save(df, name: str) -> None:
    path = config.RESULT_DIR / name
    df.to_csv(path, index=False)
    print(f"  -> {path.relative_to(config.ROOT)}")


def main() -> dict:
    t_start = time.perf_counter()
    print("=" * 78)
    print("P11 Online News Popularity - full pipeline")
    print("=" * 78)

    # ---------------- load ------------------------------------------------- #
    df = data.load_raw()
    print(f"\n[load] {len(df)} rows x {df.shape[1]} columns from {config.DATA_RAW.name}")
    row_report = data.row_count_report(df)
    print(row_report.to_string(index=False))
    _save(row_report, "00_row_count.csv")

    sanity = data.sanity_check(df)
    _save(sanity, "01_sanity.csv")

    # ---------------- Exp. 1: EDA ------------------------------------------ #
    print("\n[Exp. 1] EDA")
    shares_summary = eda.shares_summary(df)
    print(shares_summary.to_string(index=False))
    _save(shares_summary, "exp1_shares_summary.csv")

    corr = eda.correlation_table(df, top=20)
    _save(corr, "exp1_correlations.csv")
    channels = eda.channel_table(df)
    print("\n" + channels.to_string(index=False))
    _save(channels, "exp1_channels.csv")

    f1 = eda.fig_shares_distribution(df)
    f2 = eda.fig_top_correlations(corr)
    f3 = eda.fig_channel_box(df)
    print(f"  figures: {f1}, {f2}, {f3}")

    # ---------------- Exp. 2: preprocessing -------------------------------- #
    print("\n[Exp. 2] Preprocessing: drop url + timedelta, model log(shares)")
    ds = preprocess.prepare(df)
    split_info = {
        "rows": int(len(df)),
        "features": len(ds.feature_names),
        "train_rows": int(len(ds.X_train)),
        "test_rows": int(len(ds.X_test)),
        "popularity_threshold_shares": ds.popularity_threshold,
        "target": "log(shares) for regression; shares >= training median for classification",
        "random_state": config.RANDOM_STATE,
    }
    print(json.dumps(split_info, indent=2))
    (config.RESULT_DIR / "exp2_split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    # ---------------- Exp. 3-7: regression -------------------------------- #
    print("\n[Exp. 3-7] Regression on log(shares), scored on the share scale")
    knn_k, knn_tuning = regression.tune_knn_k(ds)
    print(f"  KNN k chosen on training-only validation: {knn_k}")
    _save(knn_tuning, "exp7_knn_k_tuning.csv")

    reg_table, reg_models = regression.run_all(ds, knn_k=knn_k)
    reg_table = pd.concat(
        [pd.DataFrame([regression.median_baseline(ds)]), reg_table], ignore_index=True
    )
    print(reg_table.to_string(index=False))
    _save(reg_table, "exp3_7_regression.csv")

    log_vs_raw = regression.log_vs_raw_comparison(ds)
    print("\n" + log_vs_raw.to_string(index=False))
    _save(log_vs_raw, "exp2_log_vs_raw_target.csv")

    quintiles = regression.quintile_calibration(reg_models["Random Forest"], ds)
    print("\nRanking calibration (Random Forest, test set):")
    print(quintiles.to_string(index=False))
    _save(quintiles, "exp3_quintile_calibration.csv")

    f4 = regression.fig_predicted_vs_actual(reg_models, ds)
    print(f"  figures: {f4}")

    # ---------------- Exp. 4-8: classification ---------------------------- #
    print("\n[Exp. 4-8] Classification: popular vs not (training median)")
    clf_table, clf_results = classification.run_all(ds, knn_k=knn_k, svm_cap=None)
    print(clf_table.to_string(index=False))
    _save(clf_table, "exp4_8_classification.csv")

    for name in ("Random Forest", "Logistic Regression"):
        cm = classification.confusion(clf_results, ds, name)
        print(f"\n  confusion matrix - {name}\n{cm.to_string()}")
        cm.to_csv(config.RESULT_DIR / f"exp8_confusion_{name.split()[0].lower()}.csv")

    # ---------------- Exp. 8: interpretation ------------------------------ #
    print("\n[Exp. 8] Interpretation")
    lin_coefs = interpret.linear_coefficients(clf_results["Logistic Regression"].model, ds)
    print("\nLogistic Regression coefficients (top 15 by |coef|):")
    print(lin_coefs.to_string(index=False))
    _save(lin_coefs, "exp8_logistic_coefficients.csv")

    rf_clf = clf_results["Random Forest"].model
    rf_reg = reg_models["Random Forest"]
    imp_clf = interpret.forest_importances(rf_clf, ds)
    imp_reg = interpret.forest_importances(rf_reg, ds)
    print("\nRandom Forest importances (regression model, top 15):")
    print(imp_reg.to_string(index=False))
    _save(imp_clf, "exp8_rf_importance_classification.csv")
    _save(imp_reg, "exp8_rf_importance_regression.csv")

    perm = interpret.permutation_table(rf_reg, ds)
    print("\nPermutation importance on test R2 (regression RF, top 15):")
    print(perm.to_string(index=False))
    _save(perm, "exp8_permutation_importance.csv")

    grouped = interpret.group_feature_importance(rf_clf, ds)
    print("\nImportance by feature group (classification RF):")
    print(grouped.to_string(index=False))
    _save(grouped, "exp8_group_importance.csv")

    bucket = interpret.history_vs_content_split(grouped)
    print("\n" + bucket.to_string(index=False))
    _save(bucket, "exp8_history_vs_content.csv")

    f5 = interpret.fig_importance(perm)
    f6 = interpret.fig_group_importance(grouped)
    print(f"  figures: {f5}, {f6}")

    # ---------------- model bundle for the demo --------------------------- #
    bundle = {
        "feature_names": ds.feature_names,
        "popularity_threshold": ds.popularity_threshold,
        "rf_regressor": rf_reg,
        "rf_classifier": rf_clf,
        "trained_on_rows": int(len(ds.X_train)),
        "regression_metrics": reg_table.set_index("model").loc["Random Forest"].to_dict(),
        "classification_metrics": clf_table.set_index("model").loc["Random Forest"].to_dict(),
    }
    joblib.dump(bundle, config.MODEL_DIR / "model_bundle.joblib")
    print(f"\n[bundle] models/model_bundle.joblib")

    # ---------------- summary json --------------------------------------- #
    summary = {
        "rows_loaded": int(len(df)),
        "rows_claimed_by_uci_page": config.UCI_PAGE_ROWS_CLAIMED,
        "features": len(ds.feature_names),
        "train_rows": int(len(ds.X_train)),
        "test_rows": int(len(ds.X_test)),
        "popularity_threshold_shares": ds.popularity_threshold,
        "median_shares": float(df[config.TARGET].median()),
        "mean_shares": round(float(df[config.TARGET].mean()), 1),
        "max_shares": int(df[config.TARGET].max()),
        "skew_shares": round(float(df[config.TARGET].skew()), 2),
        "skew_log_shares": round(float(np.log(df[config.TARGET]).skew()), 2),
        "top_correlations": corr.head(5).to_dict(orient="records"),
        "best_regression": reg_table.sort_values("R2 (share scale)", ascending=False).iloc[0].to_dict(),
        "best_classification": clf_table.sort_values("F1 (popular)", ascending=False).iloc[0].to_dict(),
        "quintile_lift_q5_over_q1": round(
            float(quintiles["actual_median"].iloc[-1] / quintiles["actual_median"].iloc[0]), 2
        ),
        "knn_k": knn_k,
        "history_share_of_importance": float(bucket.iloc[0, 1]),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (config.RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {summary['runtime_seconds']} s total")
    print(json.dumps(summary, indent=2)[:1200])
    return summary


if __name__ == "__main__":
    main()
