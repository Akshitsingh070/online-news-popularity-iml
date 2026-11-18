# P11 — Online News Popularity: Predicting Article Sharing Before Publication

**Course:** Introduction to Machine Learning (IML) — Project Lab  
**Group:** G11  
**Dataset:** UCI Machine Learning Repository — [Online News Popularity (Mashable)](https://archive.ics.uci.edu/dataset/332/online+news+popularity)

---

## 📌 Executive Summary

Can we predict how widely a Mashable article will be shared *prior to publication*, using only properties available at authoring time (article text statistics, media assets, topics, publishing timing, and historical keyword performance)?

We evaluate both:
1. **Regression:** Predicting exact `shares` (evaluated via MAE and $R^2$ on both raw and $\log$ scales).
2. **Classification:** Predicting binary popularity ($shares \ge 1,400$, the training median) evaluated via Accuracy, Precision, Recall, F1, and ROC-AUC.

### 🔑 Key Findings & Academic Honesty
* **Log-Transform Justification:** Target `shares` has extreme right-skew ($\text{skewness} = 33.96$, $\text{kurtosis} = 1,832.7$, range $1 \to 843,300$). Fitting on $\log(shares)$ cuts MAE by 651 shares (22%) and median absolute error from 1,682 to 733 shares compared to fitting raw shares.
* **The Reality of Raw-Scale $R^2$:** Because social virality is largely driven by off-platform amplification, homepage placement, and external news cycles, pre-publication content features explain **$\sim 0\%$ of raw-scale share variance** ($R^2 = +0.003$ for Random Forest). Any model claiming high raw $R^2$ on this dataset is either overfitting to viral outliers or leaking post-publication data.
* **Effective Quintile Ranking:** Although predicting exact share counts is intractable, models excel at **relative ranking**: held-out test articles in predicted quintile Q5 achieve an actual median of 2,700 shares vs. 957 shares in Q1 (**2.8× monotonic lift**).
* **Classification Performance:** Random Forest achieves **F1 = 0.697** and **ROC-AUC = 0.726** on the balanced median-split benchmark, significantly outperforming individual decision trees.
* **Feature Drivers:** Permutation importance reveals that **31.9% of total model signal comes from Mashable's past keyword and self-reference distribution** (`kw_avg_avg`, `kw_max_avg`, `self_reference_avg_sharess`), proving that audience familiarity dominates pure text length or sentiment.

---

## 🗂️ Project Repository Structure

```
IML Project lab/
├── .gitignore                                # Excludes venv, pycache, checkpoints
├── requirements.txt                          # Pinned dependency manifest
├── README.md                                 # Project documentation & run guide
│
├── data/
│   └── raw/
│       └── OnlineNewsPopularity.csv          # UCI dataset (39,644 rows × 61 cols)
│
├── notebooks/
│   └── P11_online_news_popularity.ipynb      # Complete, fully-executed Jupyter notebook
│
├── src/                                      # Modular production ML codebase
│   ├── __init__.py
│   ├── config.py                             # Paths, random seed (42), column groups
│   ├── data.py                               # Loader, schema sanitization, sanity checks
│   ├── preprocess.py                         # Target log-transform, leak-free train/test split
│   ├── eda.py                                # Summary stats, skewness, correlation analysis
│   ├── regression.py                         # Linear, Tree, RF, KNN regressors
│   ├── classification.py                     # Logistic, Tree, RF, KNN, SVM classifiers
│   ├── interpret.py                          # Permutation importance, feature groupings
│   └── run_all.py                            # End-to-end headless pipeline orchestrator
│
├── demo/
│   └── predict_demo.py                       # CLI interactive live scoring demonstration
│
├── models/
│   └── model_bundle.joblib                   # Serialized Random Forest regressors & classifiers
│
├── reports/
│   ├── report.md                             # 6–8 page comprehensive scientific report
│   ├── report.html                           # Standalone formatted report with CSS
│   ├── figures/                              # Generated high-resolution plots (.png)
│   └── results/                              # Exported metrics, confusion matrices, tables (.csv)
│
├── docs/
│   ├── plan.md                               # Project execution blueprint
│   ├── slides.md                             # 10-minute presentation slides with speaker notes
│   ├── slides.html                           # Standalone interactive browser slide deck
│   ├── demo_guide.md                         # Live demo step-by-step walkthrough & Q&A prep
│   └── contribution_table.md                 # Team contribution breakdown & AI statement
│
└── tools/
    ├── build_notebook.py                     # Assembles and executes the notebook
    ├── launch_notebook.bat                   # Batch helper to launch Jupyter Lab
    └── run_notebook.bat                      # Headless notebook execution runner
```

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup

Create an isolated virtual environment (Python 3.10–3.14 recommended):

```bash
# Clone the repository
git clone https://github.com/Akshitsingh070/online-news-popularity-iml.git
cd "online-news-popularity-iml"

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline

To re-run all experiments (Exp. 1 through 8), re-generate all tables and figures, and serialize the production models:

```bash
python -m src.run_all
```
*Runtime: ~5–6 minutes (includes SVM training and 5-fold permutation importance).*

### 3. Launch the Executed Notebook

```bash
jupyter lab notebooks/P11_online_news_popularity.ipynb
```
The notebook is pre-executed with all figures and outputs embedded.

---

## 🎤 10-Minute Presentation & Live Demo

### Running the Live Demo
The project includes a standalone demo utility for interactive scoring:

```bash
# 1. Score random real articles + extreme viral cases
python demo/predict_demo.py

# 2. View quintile ranking calibration on held-out test data
python demo/predict_demo.py --quintiles

# 3. Interactive live mode: test your own custom article live!
python demo/predict_demo.py --interactive
```

For the exact slide-by-slide presentation script, slide timing (0:00 to 10:00), and anticipated Q&A responses, see [docs/slides.md](docs/slides.md) and [docs/demo_guide.md](docs/demo_guide.md).

---

## 📊 Summary of Experiments & Results

| Exp | Description | Key Result |
|---|---|---|
| **Exp 1** | Exploratory Data Analysis & Skewness | Loaded 39,644 rows (153 fewer than advertised). Target skewness = 33.96. Zero duplicate rows. |
| **Exp 2** | Target Transformation & Leak-free Split | Fitting $\log(shares)$ reduces MAE on test set by 651 shares (22%) vs raw target. |
| **Exp 3** | Linear Regression Baseline | Test MAE = 2,347.8 shares, raw $R^2 = -0.0029$, $\log R^2 = 0.1274$. |
| **Exp 4** | Classification: Median Split ($shares \ge 1,400$) | Balanced test set; Logistic Regression achieves F1 = 0.6786, ROC-AUC = 0.7053. |
| **Exp 5** | Decision Tree Models | Regression DT overfits/underperforms baseline (raw $R^2 = -0.0657$); Classification DT F1 = 0.6687. |
| **Exp 6** | Random Forest (Best Overall) | **Regression:** MAE = 2,316.5 shares, raw $R^2 = +0.0033$, $\log R^2 = 0.1697$. <br>**Classification:** Accuracy = 66.6%, F1 = 0.6971, ROC-AUC = 0.7264. |
| **Exp 7** | K-Nearest Neighbors ($k=75$) | Lowest median absolute error (703.8 shares); classification F1 = 0.6572. |
| **Exp 8** | SVM & Feature Importance | RBF SVM F1 = 0.6846. Top predictor is `kw_avg_avg` ($\Delta R^2 = 0.0565$); past performance features account for 31.9% of total signal. |

---

## 👥 Contributors & Academic Declaration

Please refer to [reports/report.md](reports/report.md#8-contribution-table) and [docs/contribution_table.md](docs/contribution_table.md) for individual team roles and generative AI usage declarations.
