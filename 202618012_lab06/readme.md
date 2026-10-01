# Lab 06: Traditional ML for Images and Text

| Part | Notebook | Task |
|---|---|---|
| A: Image | `202618012_lab06_image_classification.ipynb` | Asphalt crack vs non-crack from handcrafted OpenCV/NumPy features |
| B: Text | `202618012_lab06_text_classification.ipynb` | Email spam vs non-spam from existing word-count features (no TF-IDF) |

---

## Part A: Asphalt Crack Detection

**Data:** 400 road images (200 crack, 200 non-crack, so balanced). The folder name is the label. Split 320 / 80 (stratified).

**Pipeline:** resize to 256x256, grayscale, Gaussian blur, extract 19 features (6 intensity, 4 Canny, 6 Black Hat, 3 Hough), scale, classify.

**Strategies and optimizations**
- Canny edges filtered by connected-component size, so asphalt grain is not counted as cracks.
- Black Hat pipeline: background subtraction for uneven lighting, median + Gaussian smoothing, fixed threshold, and keeping only long components.
- Scaler fitted on the training set only (no leakage).
- Five models compared, plus a feature-group ablation and Random Forest feature importances.

**Results (80 test images)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.950 | 0.929 | 0.975 | 0.951 |
| KNN | 0.950 | 0.974 | 0.925 | 0.949 |
| Logistic Regression | 0.938 | 0.927 | 0.950 | 0.938 |
| SVM (RBF) | 0.925 | 0.925 | 0.925 | 0.925 |
| Decision Tree | 0.900 | 0.881 | 0.925 | 0.902 |

Black Hat features make up about 72% of Random Forest importance. With only 80 test images, one image equals 1.25 points, so small gaps between models are not conclusive.

---

## Part B: Email Spam Classification

**Data:** 5,172 emails x 3,000 word counts, 71% non-spam / 29% spam. After removing 541 duplicates: 4,631 emails. Split 3,704 / 927 (stratified). 2,892 features remain after filtering.

**Strategies and optimizations**
- **Class imbalance:** `class_weight="balanced"` (Logistic Regression, Linear SVM) and `fit_prior=False` (Naive Bayes), compared against unweighted baselines.
- **No leakage:** duplicates removed, then the split, then rare-word filtering (min 10 emails) fitted on the training set only.
- **`log1p`** of counts for the linear models (better convergence). Naive Bayes keeps raw counts.
- **Tuning:** `GridSearchCV` with stratified 5-fold CV, optimizing spam F1.
- **Metrics for imbalance:** precision/recall/F1 of the spam class and PR-AUC, not accuracy alone (always predicting non-spam already gives 0.685).

**Results (927 test emails; spam-class metrics)**

| Model | Setting | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| Naive Bayes | Baseline | 0.956 | 0.893 | 0.976 | 0.933 | 0.930 |
| Naive Bayes | Balanced + tuned | 0.960 | 0.900 | 0.983 | 0.939 | 0.946 |
| Logistic Regression | Baseline | 0.971 | 0.952 | 0.956 | 0.954 | 0.994 |
| Logistic Regression | Balanced + tuned | 0.975 | 0.944 | 0.980 | 0.961 | 0.994 |
| Linear SVM | Baseline | 0.966 | 0.945 | 0.945 | 0.945 | 0.990 |
| Linear SVM | Balanced + tuned | 0.976 | 0.950 | 0.976 | 0.963 | 0.993 |

5-fold CV F1: Naive Bayes 0.938 ± 0.010, Logistic Regression 0.953 ± 0.005, Linear SVM 0.952 ± 0.005.

**Takeaways:** balancing and tuning raised spam recall and F1 for every model. Logistic Regression and Linear SVM are effectively tied, and Naive Bayes is the weakest but fastest (about 0.05 s to train vs 0.2 to 0.4 s).

---



