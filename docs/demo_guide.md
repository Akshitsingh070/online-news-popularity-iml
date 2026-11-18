# Live Demo Presenter Cheat-Sheet

**Target Time:** 2 minutes during the 10-minute presentation  
**Tool:** `demo/predict_demo.py`  
**Model Bundle:** `models/model_bundle.joblib` (Random Forest Regressor + Random Forest Classifier)

---

## ⚡ Quick-Run Commands

Open your terminal in the project root (`IML Project lab`) and ensure your `.venv` is active:

```bash
# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Scenario 1: Standard Run (Recommended for Main Demo)
```bash
python demo/predict_demo.py
```
**What happens:**
1. Loads the pre-trained Random Forest model bundle (58 features, 31,715 training rows).
2. Scores 5 random real articles from the corpus with predicted shares, probability $P(\text{popular} \ge 1,400)$, and actual error.
3. Automatically demonstrates **two extreme cases**:
   - The extreme viral hit (843,300 actual shares) $\to$ Model predicts ~27,456 shares and $P=0.85$ (recognizes high popularity without wildly hallucinating millions).
   - The median article (1,400 actual shares) $\to$ Model predicts ~1,545 shares (only +10% error).
4. Prints the test-set **quintile ranking table** proving monotonic median lift from Q1 (957) to Q5 (2,700).

---

### Scenario 2: Show Quintile Ranking Calibration
```bash
python demo/predict_demo.py --quintiles
```
**Key talking point for evaluators:**
> *"Even though individual share error can be large on outliers, look at how the predicted quintiles rank held-out test articles: Q1 median is 957 shares, while Q5 median is 2,700 shares. That is a 2.8× monotonic lift, making this model an effective priority triage tool for editors."*

---

### Scenario 3: Live Interactive Article Authoring
```bash
python demo/predict_demo.py --interactive
```
**Prompts and Recommended Demo Inputs:**

When prompted by the terminal:
* **Channel:** Enter `tech` (or `lifestyle`, `business`, `world`, `entertainment`, `social media`)
* **Article word count:** Enter `500`
* **Outbound links:** Enter `10`
* **Images:** Enter `3`
* **Keywords count:** Enter `8`
* **Keyword average popularity:** Enter `3500` (represents trending keywords)
* **Published on weekend? (y/n):** Enter `y`

**Outcome:**
The tool instantly evaluates the synthetic article and outputs:
- Predicted Shares (e.g. ~2,100 shares)
- Probability of Being Popular (e.g. ~82% [POPULAR])
- Brief explanation of the driving factors (high keyword traction + weekend boost).

---

## 🎯 Emergency Backup / If Something Fails
If python cannot be run during a live screen share, open `reports/results/summary.json` or show the pre-rendered outputs in the Jupyter notebook `notebooks/P11_online_news_popularity.ipynb` under Section 7.
