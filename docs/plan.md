# P11 — Online News Popularity: project plan

Goal: predict how widely a Mashable article will be shared *before* publication, using only
content / link / keyword / publishing-day features. Deliverables: notebook, 6–8 page report,
GitHub repository, 10-minute presentation with live demo.

## Deliverable checklist

| # | Deliverable | Where |
|---|-------------|-------|
| 1 | Executed notebook (.ipynb) | `notebooks/P11_online_news_popularity.ipynb` |
| 2 | Report, 6–8 pages | `reports/report.md` (+ `reports/report.pdf` if a converter is available) |
| 3 | GitHub repository | local repo committed; `README.md` = repo front page |
| 4 | Presentation + live demo | `docs/slides.md`, `demo/predict_demo.py` |
| 5 | Contribution table + AI-use declaration | `docs/contribution_table.md`, report §9–§10 |

## Experiment map (matches the lab sheet's "Exp. 1–8")

| Exp. | Question | Code |
|------|----------|------|
| 1 | EDA: shape and skew of `shares`; which features relate to `log(shares)` | `src/eda.py` |
| 2 | Preprocessing: drop `url` + `timedelta`, model `log(shares)`, scale, split | `src/preprocess.py` |
| 3 | Baseline: Linear Regression; MAE + R² back on the share scale | `src/regression.py` |
| 4 | Classification: popular = shares ≥ training median; Logistic Regression, F1 | `src/classification.py` |
| 5 | Decision Tree, regression + classification | `src/regression.py`, `src/classification.py` |
| 6 | Random Forest, regression + classification | same |
| 7 | KNN (regression + classification)  | same |
| 8 | SVM classifier; comparison table; feature importance | `src/classification.py`, `src/interpret.py` |

## Method decisions

1. **Why log(shares).** `shares` is heavy-tailed (minimum 1, median ≈ 1400, maximum ≈ 843 300,
   skew ≈ 11). Squared-error models are dominated by a handful of viral outliers, and a linear
   model then predicts the mean of a right-skewed target. Fitting `log(shares)` makes the target
   roughly symmetric, stabilises the variance, and turns "x % more shares" into an additive effect.
   Metrics are then reported **back on the original share scale** (`expm1`, clipped at 1) so MAE is
   interpretable in shares; R² on that scale is the honest number and is expected to be low.
2. **Leak-free split.** Single 80/20 train/test split, `random_state=42`. All scaling is fitted on
   the training part only. The popularity threshold is the **training** median and is applied
   unchanged to the test part — otherwise the test labels would leak test-set information.
3. **Scaling only where needed.** Trees are scale-invariant; Linear Regression, KNN and SVM get
   `StandardScaler`. One-hot channel/weekday columns are kept as-is (they are already 0/1).
4. **Hyper-parameters fixed, not tuned.** The brief compares model families, so each family gets one
   sensible setting (documented in the report). The only thing searched is KNN's k, on a validation
   slice carved out of *training* data — the test set is never touched for model selection.
5. **SVM.** RBF SVM is O(n²); it is trained on the full scaled training set only if it fits the
   time budget, otherwise on a stratified 10 000-row subsample with `LinearSVC` as a full-data
   cross-check. Either way the choice is reported, not hidden.
6. **Honesty about the ceiling.** Expect R² ≈ 0.02–0.10 and F1 ≈ 0.6. Section 5 of the report says
   why: article features cannot see the audience, the referrer, the news cycle, the competing
   stories of the day, or the site's social distribution — and Mashable's own historical behaviour
   (`kw_*`, `self_reference_*`) is the only proxy for those.

## Work order

1. Data acquisition + row-count check (`data/raw/OnlineNewsPopularity.csv`).
2. `src/` library: data → preprocess → eda → regression → classification → interpret.
3. `src/run_all.py` end-to-end run: writes `reports/results/*.csv`, `reports/figures/*.png`,
   `models/model_bundle.joblib`.
4. Notebook assembled by `tools/build_notebook.py` and executed head-less with nbconvert so the
   submitted notebook already contains outputs.
5. Report from the generated numbers (never hand-copied guesses).
6. Slides + live demo; git init and commit.
