# DS605 – Lab Assignment 3
## Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:Yatri Vora** 
**ID:202618012** 
**Dataset:** [Kaggle Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (`hotel_bookings.csv`)

---

## Preprocessing Choices

- **Missing values:**
  - Numerical columns → `KNNImputer(n_neighbors=5)`
  - Categorical columns → `SimpleImputer(strategy="most_frequent")`
- **Scaling (two pipelines compared):**
  - **Pipeline A** → `StandardScaler` on numerical features
  - **Pipeline B** → `MinMaxScaler` on numerical features
- **Categorical encoding:** `OneHotEncoder(handle_unknown="ignore")`
- **Train/test split:** 80/20, stratified on the target (`is_canceled`), `random_state=42`
- Preprocessing was assembled with `ColumnTransformer` + `Pipeline`, combined with the model in a single end-to-end `sklearn.Pipeline` for each run.

## Models

- `LogisticRegression(max_iter=1000)`
- `DecisionTreeClassifier(random_state=42)`

Each model was trained under both Pipeline A and Pipeline B (4 combinations total).

## Final Comparison Table

| Model + Pipeline | Train Accuracy | Test Accuracy | Precision | Recall | F1-score | Train-Test Gap |
|---|---|---|---|---|---|---|
| DecisionTree + Pipeline B (MinMaxScaler) | 0.9962 | 0.8652 | 0.8151 | 0.8201 | 0.8176 | 0.1310 |
| DecisionTree + Pipeline A (StandardScaler) | 0.9962 | 0.8650 | 0.8152 | 0.8193 | 0.8173 | 0.1312 |
| LogisticRegression + Pipeline A (StandardScaler) | 0.8334 | 0.8301 | 0.8107 | 0.7033 | 0.7532 | 0.0033 |
| LogisticRegression + Pipeline B (MinMaxScaler) | 0.8285 | 0.8250 | 0.8047 | 0.6933 | 0.7449 | 0.0035 |

(Full table also saved as `model_comparison_table.csv`.)

## Final Observations

1. **Best overall combination:** Decision Tree + Pipeline B (MinMaxScaler) gives the best test accuracy (0.8652) and F1-score (0.8176), narrowly ahead of Decision Tree + Pipeline A. Both Decision Tree combinations beat both Logistic Regression combinations on every metric except train-test gap.
2. **Scaler effect on Logistic Regression:** StandardScaler (0.8301 test accuracy) edges out MinMaxScaler (0.8250) by about half a point — consistent with LR's gradient-based optimizer benefiting from zero-centred, unit-variance features.
3. **Scaler effect on Decision Tree:** Scaling barely matters (0.8650 vs 0.8652) since trees split on per-feature thresholds and both scalers are monotonic transforms.
4. **Overfitting:** The Decision Tree shows a large train-test gap (~0.131), a classic sign of an unconstrained tree (no `max_depth`) memorising the training data. Logistic Regression shows almost no gap (~0.003–0.004).
5. **Precision vs. recall:** The best Decision Tree has balanced precision/recall (0.8151 / 0.8201) on the cancelled class, while Logistic Regression is precision-heavy (0.8107 / 0.7033) — it misses more true cancellations. For a hotel acting on cancellation risk, the Tree's better recall is a practically meaningful advantage.

## Repository Contents

- `202618012_DS605_Lab3.ipynb` — full runnable notebook (data cleaning, both pipelines, both models, evaluation)
- `hotel_bookings.csv` — raw dataset
- `hotel_bookings_cleaned.csv` — cleaned base dataset used for modeling
- `model_comparison_table.csv` — final comparison table
- `confusion_matrices.png` — confusion matrices for the best Logistic Regression and best Decision Tree models
- `README.md` — this file





