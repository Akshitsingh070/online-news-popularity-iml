# 10-Minute Presentation: Predicting Online News Popularity

**Title:** P11 — Online News Popularity: Predicting Article Sharing Before Publication  
**Target Duration:** 10 minutes (8 min presentation + 2 min interactive live demo)  
**Speaker Guidelines:** Spend roughly 1 minute per slide, maintaining clear emphasis on academic honesty, methodology, and the live demo.

---

## Slide 1: Title & Problem Formulation (0:00 – 1:00)

### Visual Content
* **Title:** Online News Popularity: Predicting Article Sharing Before Publication
* **Core Question:** Can an editor predict the social reach of a news article *at the moment of writing*, before it is published?
* **Available Features (58):** Text statistics, media assets, topics (LDA), timing, and historical keyword performance.
* **The Dual Formulation:**
  1. **Regression:** Predict exact share count (`shares`). Direct editorial question ("how many shares?"), but heavily penalized by rare viral hits.
  2. **Classification:** Predict binary popularity ($shares \ge \text{median} = 1,400$). Content-planning question ("is this worth prime scheduling?").

### 🎙️ Presenter Script (Minute 0 to 1)
> "Good morning/afternoon. Today we present our project on predicting Online News Popularity using the UCI Mashable dataset. 
> Every year, digital publishers like Mashable produce tens of thousands of articles whose social shares vary from a single share to nearly a million. 
> Our central question is: given only what an editor knows *before* hitting publish—word counts, media links, topic mixtures, and keyword history—can we reliably predict article reach?
> To answer this honestly, we investigated two distinct formulations: regression to estimate numerical shares, and balanced classification around the 1,400-share median for content scheduling."

---

## Slide 2: Data Audit & The Skewness Challenge (1:00 – 2:15)

### Visual Content
* **Row-Count Audit:** Loaded **39,644 rows × 61 columns** — exactly 153 rows fewer than advertised on UCI's webpage (verified at load time; 0 duplicates, 0 missing).
* **Extreme Target Skewness:**
  * Median: **1,400 shares** | Mean: **3,395 shares** | Maximum: **843,300 shares**
  * **Skewness:** **33.96** | **Kurtosis:** **1,832.7**
  * Top 1% of articles account for **21.8% of all platform shares**.
* *Visual:* Distribution plots comparing raw shares (infinite right tail) vs. $\log(shares)$ (bell-shaped, skewness drops to 1.02).

### 🎙️ Presenter Script (Minute 1 to 2:15)
> "Before running models, we audited data integrity. We discovered an immediate discrepancy: UCI advertises 39,797 instances, but the official archive contains exactly 39,644 rows. We verified this discrepancy programmatically—there are zero missing values and zero duplicate rows.
> The key characteristic of this dataset is extreme right-skew. While the median article gets 1,400 shares, the mean is over 3,395, and the maximum is 843,300 shares. The top 1% of articles capture over 21% of all social shares!
> On the raw scale, the skewness is 33.96 with kurtosis of 1,832. As shown in the distribution plot, raw shares have an impossible tail. By taking the natural logarithm, we reduce skewness to 1.02, producing a roughly symmetric target suitable for linear and distance-based learning."

---

## Slide 3: Methodological Rigor & Protocol (2:15 – 3:30)

### Visual Content
* **Leak-Free Hold-Out Split:**
  * 80% Train (31,715 articles), 20% Held-Out Test (7,929 articles). `random_state = 42`.
  * All scalers (`StandardScaler`) fitted **strictly on training data**.
  * Popularity threshold fixed at the **training median (1,400)**, avoiding label leakage into test data.
* **Feature Pruning:** Dropped `url` (row identifier) and `timedelta` (publication timestamp) to prevent memorizing calendar drift.
* **Why Modelling $\log(shares)$ Matters (Exp. 2 Empirical Proof):**
  * Raw Target model: MAE = **2,998.7**, Median Abs Error = **1,682.3**
  * Log Target model: MAE = **2,347.8** (-22%), Median Abs Error = **733.1** (-56%!)
  * *All metrics are mapped back to actual shares using `expm1` for reporting.*

### 🎙️ Presenter Script (Minute 2:15 to 3:30)
> "A major priority of our work was methodological rigor. We enforced a strict leak-free protocol: an 80/20 train-test split where standard scalers were fitted exclusively on training rows. The classification threshold of 1,400 was locked on training data alone.
> Crucially, we pruned `url` and `timedelta`. Keeping `timedelta` would allow a model to artificially memorize specific high-traffic weeks of a historical two-year period rather than learning true pre-publication signals.
> Furthermore, we proved empirically why modeling log(shares) is essential. Training directly on raw shares tries to chase extreme viral outliers, resulting in a median error of 1,682 shares. Training on log(shares) drops typical error to 733 shares—a 2.3-fold improvement. Crucially, we always convert predictions back to real shares before computing errors."

---

## Slide 4: Regression Track & The Reality of $R^2$ (3:30 – 5:00)

### Visual Content
* **Regression Comparison Table (Test Set, Back on Share Scale):**
  | Model | MAE (shares) | RMSE | $R^2$ (share scale) | $R^2$ (log scale) | Median Abs Error |
  |---|---|---|---|---|---|
  | **Median Baseline** | 2,395.4 | 11,152.9 | -0.0308 | -0.0578 | **600.0** |
  | **Linear Regression** | 2,347.8 | 11,000.9 | -0.0029 | +0.1274 | 733.1 |
  | **Decision Tree** | 3,019.3 | 11,340.3 | -0.0657 | -0.3903 | 975.1 |
  | **Random Forest** | **2,316.5** | **10,967.0** | **+0.0033** | **+0.1697** | 745.8 |
  | **KNN ($k=75$)** | 2,334.9 | 11,034.9 | -0.0091 | +0.1077 | 703.8 |
* **Key Takeaway:** Share-scale $R^2 \approx 0.00$. Pre-publication article attributes cannot explain raw viral variance.
* **The Silver Lining — Quintile Ranking Calibration:**
  * Articles predicted in Q1 achieve an actual median of **957 shares**.
  * Articles predicted in Q5 achieve an actual median of **2,700 shares** (**2.8× monotonic lift!**).

### 🎙️ Presenter Script (Minute 3:30 to 5:00)
> "Here are our regression results, evaluated back on the real share scale. 
> Notice the headline figure: the best model, Random Forest, achieves an $R^2$ on the original share scale of just +0.0033. It beats the trivial baseline on MAE by only 79 shares. Single decision trees perform worse than predicting the median due to high variance.
> Rather than viewing this as a failure, this is our central finding: predicting exact numerical virality before publication is fundamentally impossible from article features alone. Virality is determined by external factors like social distribution algorithms, influencer retweets, and breaking news cycles.
> However, our model succeeds powerfully at *relative ranking*. When we divide test predictions into quintiles, actual median shares climb monotonically from 957 in Q1 to 2,700 in Q5—a 2.8-fold lift. The model cannot give you the exact number, but it can reliably tell you which articles belong in your top tier."

---

## Slide 5: Classification Track: Five Model Families (5:00 – 6:15)

### Visual Content
* **Balanced Classification Benchmark ($shares \ge 1,400$):**
  | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Fit Time |
  |---|---|---|---|---|---|---|
  | **Random Forest (300 trees)** | **66.64%** | 66.49% | **73.26%** | **0.6971** | **0.7264** | 18.3 s |
  | **SVM (RBF Kernel)** | 65.59% | 65.88% | 71.24% | 0.6846 | 0.7142 | 131.0 s |
  | **Logistic Regression** | 65.23% | 65.81% | 70.04% | 0.6786 | 0.7053 | **0.3 s** |
  | **Decision Tree** | 63.99% | 64.57% | 69.34% | 0.6687 | 0.6875 | 1.3 s |
  | **KNN ($k=75$)** | 64.07% | 65.71% | 65.73% | 0.6572 | 0.6902 | 0.6 s |
* **Observations:**
  * All five families cluster tightly between F1 = 0.66 and 0.70.
  * Recall (73%) > Precision (66%): beneficial for editorial scheduling (low false negatives).
  * RBF SVM requires 400× more computation than Logistic Regression for only +0.006 F1 gain.

### 🎙️ Presenter Script (Minute 5:00 to 6:15)
> "When we convert the problem into binary classification—predicting whether an article surpasses the 1,400-share median—the models show clear practical utility.
> Random Forest leads with an F1 score of 0.6971 and an ROC-AUC of 0.7264. Notice that recall is 73% while precision is 66%. For a content recommendation system, this is an advantageous operating point because failing to feature a potential hit is costlier than giving a standard article a temporary promotional boost.
> Notice also the algorithmic comparison: all five families perform within a narrow 4% F1 window. Training an RBF SVM on all 31,000 rows took 131 seconds—over 400 times longer than Logistic Regression—for a negligible gain of 0.006 F1. This shows that the performance ceiling is bounded by feature information, not model complexity."

---

## Slide 6: Model Interpretability & Feature Drivers (6:15 – 7:30)

### Visual Content
* **Permutation Feature Importance ($\Delta R^2$ on held-out test data):**
  1. `kw_avg_avg` (+0.0565): Average past shares of article keywords.
  2. `kw_max_avg` (+0.0196): Best-performing historical keyword.
  3. `self_reference_avg_sharess` (+0.0172): Shares of linked Mashable articles.
  4. `is_weekend` (+0.0158): Weekend publication boost.
* **Importance by Feature Group:**
  * **31.9%** Past site history (`kw_*`, `self_reference_*`)
  * **33.5%** Content length & external links
  * **15.0%** LDA Topic mixtures
  * **12.9%** Sentiment & polarity
  * **6.7%** Channel & publishing day
* *Insight:* The strongest signal is historical audience engagement, not prose writing quality!

### 🎙️ Presenter Script (Minute 6:15 to 7:30)
> "To discover what actually drives predictions, we conducted permutation importance testing on held-out data.
> The single most powerful feature by far is `kw_avg_avg`—the historical average shares of the keywords assigned to the article. Permuting this single feature drops test performance more than the next ten features combined.
> When grouping features, nearly a third of total signal comes from past historical performance: previous keyword traction and self-referenced article shares. Genuinely textual features like word count, readability, and sentiment have modest impact.
> In short: the model is largely predicting the past forward. It succeeds by recognizing familiar topics that Mashable's existing audience previously shared, rather than evaluating writing quality."

---

## Slide 7: Live Demonstration (7:30 – 9:00)

### Visual Content & Action
* **Action:** Switch to terminal and run the interactive CLI tool:
  ```bash
  python demo/predict_demo.py
  ```
* **Demonstration Cases:**
  1. Real articles from test set: shows calibrated probability $P(\text{popular})$ and predicted share count.
  2. The 843,000-share viral outlier: demonstrates model conservatism (predicts ~27k shares, refusing to hallucinate extreme numbers).
  3. Test set quintile ranking display (`--quintiles`).
  4. Interactive authoring (`--interactive`): show real-time scoring of a sample tech article.

### 🎙️ Presenter Script (Minute 7:30 to 9:00)
> "Now let us demonstrate the live scoring system in action. *(Run: python demo/predict_demo.py)*
> Here we score articles in real time using our saved Random Forest model bundle.
> In our first sample, a tech article with high keyword popularity achieves an 81% probability of being popular and predicts 1,928 shares. The actual was 1,900 shares—just a 1% error.
> Next, observe how the model handles the extreme viral outlier of 843,000 shares. The model predicts 27,456 shares and marks it 85% popular. It recognizes high potential, but correctly refuses to chase millions of shares because pre-publication features cannot anticipate social media runaway loops.
> Finally, looking at our quintile breakdown, we see a clean, monotonic ladder: editors can reliably use these predictions to sort articles into promotional tiers."

---

## Slide 8: Real-World Limitations & Conclusion (9:00 – 10:00)

### Visual Content
* **Why Pre-Publication Prediction Has an Upper Bound:**
  1. **Missing Distribution Data:** Homepage slotting, push notifications, and aggregator pickups occur *after* publication.
  2. **Off-Platform Amplification:** Viral cascades depend on external influencers and community reposts.
  3. **Zero-Sum Attention:** Competing breaking news events cannot be known in advance.
  4. **Concept Drift:** Historical keyword features decay when audience interests evolve.
* **Conclusion & Submission Deliverables:**
  * ✅ Executed Jupyter Notebook (`notebooks/P11_online_news_popularity.ipynb`)
  * ✅ 6–8 Page Formal Report (`reports/report.md` / `report.pdf`)
  * ✅ GitHub Repository (`README.md`, reproducible code)
  * ✅ 10-Minute Presentation & CLI Live Demo (`demo/predict_demo.py`)

### 🎙️ Presenter Script (Minute 9:00 to 10:00)
> "To conclude, our study provides both a practical decision-support tool and an honest appraisal of predictive limits.
> An article's text, media, and keywords can identify topic relevance and relative tier, achieving 70% classification F1 and a 2.8× quintile spread. However, no model can predict exact virality before publication because viral reach is created by the environment—editorial placement, breaking news competition, and influencer amplification.
> All deliverables—including our pre-executed notebook, 6-8 page report, GitHub repository, and live demo script—are complete and reproducible. 
> Thank you, and we welcome any questions!"

---

## 💡 Anticipated Q&A (Preparation for Evaluation Panel)

* **Q: Why is your regression $R^2$ essentially zero (+0.003), while classification achieves ~70% F1?**  
  *Answer:* $R^2$ on the raw scale measures fraction of raw variance explained. The raw variance is dominated by rare viral hits (up to 843,000 shares) that pre-publication text features cannot anticipate. Classification dichotomizes the space at the median (1,400 shares), turning it into a stable ranking problem that filters out extreme tail variance.

* **Q: Why did you drop the `timedelta` column?**  
  *Answer:* `timedelta` records the publication date relative to the corpus start. In a real-world setting, a model must predict future articles whose dates lie outside the training window. Retaining `timedelta` allowed tree models to memorize specific peak-traffic calendar weeks, artificially inflating test scores through temporal leakage.

* **Q: Why didn't you spend time hyper-parameter tuning SVM?**  
  *Answer:* Our protocol aimed to compare algorithm families under equivalent regularized capacity. RBF SVM required 131 seconds on 31,000 rows for negligible gain over 0.3-second Logistic Regression. Our permutation analysis proved that performance is feature-constrained, not model-capacity-constrained.
