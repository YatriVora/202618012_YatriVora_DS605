# DS605 Lab 5 — ML with Scikit-learn and From Scratch

Garment Employee Productivity dataset (UCI). Linear Regression predicts `actual_productivity`; Logistic Regression predicts `MeetsTarget` (`actual_productivity >= targeted_productivity`).

## Dataset & Preprocessing
- **Missing values**: `wip` (~42% missing, structurally absent for the "finishing" department) → median imputed on train, applied to test.
- **Cleaning**: fixed `department` typo (`sweing`→`sewing`) and whitespace; dropped `date` (redundant with `quarter`/`day`); clipped `actual_productivity` to 1.0.
- **Encoding/scaling**: one-hot for `quarter`, `department`, `day`, `team`; StandardScaler on numeric features.
- **Class balance**: `MeetsTarget` is ~73% / 27% (moderate imbalance) — used stratified split, and precision/recall/F1 alongside accuracy.
- **Outliers**: `idle_time`, `idle_men`, `no_of_style_change` are mostly zero with rare spikes — checked via IQR, left as-is (genuine production-disruption events, not data errors).
- Single fixed train-test split (80/20, `random_state=42`, stratified on `MeetsTarget`) reused across Parts A–C for fair comparison.

## Part A — Scikit-learn Results

| Task | Metric | Value |
|---|---|---|
| Regression | MAE / RMSE / R² | 0.1079 / 0.1423 / 0.2924 |
| Regression | Train / Predict time | 0.0514s / 0.00023s |
| Classification | Accuracy / Precision / Recall / F1 | 0.6833 / 0.7647 / 0.8171 / 0.7901 |
| Classification | Train / Predict time | 0.0221s / 0.00028s |

*(Times measured on already-preprocessed data, with warm-up runs to remove first-call/LAPACK init overhead, averaged over 10 trials.)*

## Part B — From-Scratch (NumPy/Pandas) Results

Linear Regression solved via pseudo-inverse (normal equation); Logistic Regression via batch gradient descent (sigmoid + binary cross-entropy, `lr=0.1`, up to 2000 iterations).

| Task | Metric | Value |
|---|---|---|
| Regression | MAE / RMSE / R² | 0.1079 / 0.1423 / 0.2924 *(identical to sklearn — OLS has a unique closed-form solution)* |
| Regression | Train / Predict time | 0.0669s / 0.00006s |
| Classification | Accuracy / Precision / Recall / F1 | 0.6875 / 0.7632 / 0.8286 / 0.7945 |
| Classification | Train / Predict time | 0.2230s / 0.00031s (**910% slower** train time than sklearn) |

## Part C — Optimization

**Feature selection**: `no_of_workers` dropped — highly correlated with `smv` (r=0.915) and `over_time` (r=0.744), while having the weakest individual correlation with the target. `team` was checked (mean |coefficient| 0.54 vs. 0.40 for other features) and **kept** — it carries genuine signal, not noise.

**Regression — Ridge (λ=1.0) on reduced features.** Tested; did not improve MAE/RMSE/R² over baseline (0.1103 / 0.1441 / 0.2741 vs. baseline's 0.2924 R²). Expected: baseline R²≈0.29 with no train/test gap indicates **underfitting**, not overfitting — regularization trades bias for variance reduction that wasn't needed. An interaction/polynomial-terms variant (`incentive×targeted_productivity`, `smv²`, `over_time×no_of_style_change`) was also tried and performed worse (R²=0.2267), likely due to multicollinearity with existing features — rejected. Ridge kept as final choice for its **training-time benefit** (0.00599s — faster than both scratch baseline and sklearn) despite no accuracy gain.

**Classification — Newton's method (IRLS) replacing gradient descent.** Uses the Hessian for quadratic convergence instead of GD's linear convergence.

| Task | Metric | Scikit-learn | Scratch (baseline) | Scratch (optimized) |
|---|---|---|---|---|
| Regression | MAE | 0.1079 | 0.1079 | 0.1103 |
| Regression | RMSE | 0.1423 | 0.1423 | 0.1441 |
| Regression | R² | 0.2924 | 0.2924 | 0.2741 |
| Regression | Train time | 0.0514s | 0.0669s | **0.0060s** |
| Regression | Predict time | 0.00023s | 0.00006s | 0.00008s |
| Classification | Accuracy | 0.6833 | 0.6875 | **0.6833** |
| Classification | Precision | 0.7647 | 0.7632 | **0.7647** |
| Classification | Recall | 0.8171 | 0.8286 | **0.8171** |
| Classification | F1 | 0.7901 | 0.7945 | **0.7901** |
| Classification | Train time | 0.0221s | 0.2230s | **0.0300s** |
| Classification | Predict time | 0.00028s | 0.00031s | 0.00018s |

## Key Observations
- Linear Regression via pseudo-inverse matches sklearn **exactly** — OLS has one unique global solution, so any correct closed-form solver converges to it.
- Newton's method matches sklearn's classification metrics **exactly** and cuts the training-time gap from **910% → ~36%**, confirming GD's linear convergence was the bottleneck, not model capacity.
- Remaining ~36% classification runtime gap is implementation-language overhead (compiled LAPACK/Fortran in sklearn vs. pure Python/NumPy loop) — not closable without leaving the NumPy/Pandas constraint.
- Regression's R² ceiling (~0.29) reflects genuine unexplained variance in the dataset (likely unmeasured factors like worker skill/fatigue), not a fixable modeling gap — confirmed by testing two different remedies (regularization, feature interactions), both of which failed to improve it.

