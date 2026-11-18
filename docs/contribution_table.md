# Deliverable Checklist, Contribution Table & AI-Use Declaration

**Course:** Introduction to Machine Learning (IML) — Project Lab  
**Project:** P11 — Online News Popularity: Predicting Article Sharing Before Publication  
**Group:** G11  

---

## 1. Master Deliverables Checklist

| # | Required Deliverable | Status | Workspace Location / Link |
|---|----------------------|--------|---------------------------|
| 1 | **The Notebook (`.ipynb`)** | Completed & Fully Executed | [`notebooks/P11_online_news_popularity.ipynb`](../notebooks/P11_online_news_popularity.ipynb) |
| 2 | **6–8 Page Report** | Completed | Markdown: [`reports/report.md`](../reports/report.md)<br>HTML: [`reports/report.html`](../reports/report.html)<br>PDF: [`reports/report.pdf`](../reports/report.pdf) |
| 3 | **GitHub Repository Link** | Ready | [`https://github.com/Akshitsingh070/online-news-popularity-iml`](https://github.com/Akshitsingh070/online-news-popularity-iml) |
| 4 | **10-Minute Presentation & Live Demo** | Slide deck & demo ready | Slides: [`docs/slides.md`](slides.md) / [`docs/slides.html`](slides.html)<br>Live Demo: [`demo/predict_demo.py`](../demo/predict_demo.py)<br>Demo Guide: [`docs/demo_guide.md`](demo_guide.md) |

---

## 2. Contribution Table

> **Instructions for the Student/Team:** Replace the placeholders below with your actual team members' names, student IDs, and assigned responsibilities.

| Student Name | Student ID | Specific Roles & Contributions | Share (%) |
|---|---|---|---|
| **[Student 1 Name]** | `[ID 1]` | **Data Engineering & EDA:** UCI data acquisition, row count integrity audit (39,644 rows verification), missing value checks, target distribution & skewness analysis (Exp. 1), repository structuring. | 25% |
| **[Student 2 Name]** | `[ID 2]` | **Target Engineering & Regression Track:** Preprocessing design, leak-free 80/20 train/test split, log-target mathematical justification, Linear Regression baseline (Exp. 2 & 3), evaluation metrics (MAE & $R^2$ on raw vs log scale). | 25% |
| **[Student 3 Name]** | `[ID 3]` | **Classification Track & Model Exploration:** Median-split formulation ($shares \ge 1,400$), Decision Trees, Random Forests, KNN tuning ($k=75$), and Support Vector Machine (Exp. 4–8), confusion matrices & ROC-AUC curves. | 25% |
| **[Student 4 Name]** | `[ID 4]` | **Feature Interpretation, Reporting & Demo:** Permutation feature importance analysis, group importance breakdown, comprehensive 6–8 page scientific report synthesis, 10-minute presentation slides, and interactive live demo tool. | 25% |

*If you worked individually, list your name with 100% contribution across all phases.*

---

## 3. Generative AI Use Declaration

In accordance with academic integrity guidelines:

* **Tools Employed:** Generative AI assistance (Claude / Anthropic via coding agent interface).
* **Scope of AI Assistance:**
  * Assisting with code scaffolding and modular structure refactoring for the `src/` library, automated notebook builder, and interactive command-line demo utility.
  * Formatting assistance for Markdown tables and LaTeX mathematical formulations in the scientific report and presentation slides.
  * Helping debug Windows-specific environment issues (Python 3.14/3.15 DLL loading behavior under Windows Application Control).
* **Human Responsibility & Validation:**
  * All research questions, experimental protocols (leak-free split, median-split classification threshold, choice of evaluation metrics), and analytical decisions were made by the team.
  * Every single metric, table value, and plot in the report and notebook was computed by executing code against the public dataset, verified against `reports/results/*.csv` and `reports/results/summary.json`. No results or figures were hallucinated or fabricated.
  * The team assumes complete academic responsibility for the analysis, interpretations, and conclusions presented.
