# DS605 — Lab 4: Airbnb NYC Price Prediction

End-to-end machine learning project that predicts the nightly price of an
Airbnb listing in New York City, built for DS605 (Fundamentals of Machine
Learning).

## Overview

The project follows the full ML workflow: data cleaning and feature
engineering, comparing multiple regression models, hyperparameter tuning,
and deploying the final model as an interactive Streamlit web app.

**Dataset:** [NYC Airbnb Open Data 2019](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (`AB_NYC_2019.csv`) — 48,895 listings, 16 original columns.

## Repository Structure

```
├── DS605_Lab4_Task1.ipynb     # Data cleaning, feature engineering, feature selection (SHAP)
├── DS605_Lab4_Task2.ipynb     # Model comparison, hyperparameter tuning, evaluation
├── app.py                     # Streamlit app for live price prediction
├── requirements.txt           # Python dependencies
├── AB_NYC_2019.csv            # Raw dataset
├── final_model_pipeline.pkl   # Saved final model + preprocessing artifacts
├── selected_features.pkl      # Features selected via SHAP importance
└── README.md
```

## Task 1 — Data Analysis & Preparation

- **Cleaning:** removed invalid `price == 0` listings and exact duplicates.
- **Missing values:** `reviews_per_month` and `last_review` are missing for
  listings that have never been reviewed (~20% of the data) — handled with
  an explicit `has_reviews` flag rather than guessed imputations.
- **Outlier treatment:** winsorized `price` and `minimum_nights` at the
  1st/99th percentiles instead of deleting rows; log-transformed `price`
  for modeling (`log1p`).
- **Feature engineering:**
  - Review-recency: `has_reviews`, `days_since_last_review` (with a
    sentinel value for never-reviewed listings), `review_density`
  - Host-based: `is_multi_listing_host`, `host_listings_bucket`
  - Availability-based: `occupancy_proxy`, `availability_bucket`
  - Location: `distance_to_center` (Times Square) and `distance_to_jfk`,
    computed via the Haversine formula
  - `minimum_nights_bucket`, `room_type × neighbourhood_group` interaction,
    `name_length`
- **Encoding:** one-hot encoding for low-cardinality categoricals; smoothed
  mean target-encoding for the high-cardinality `neighbourhood` column
  (221 unique values), fit only on the training split to avoid leakage.
- **Feature selection:** ranked all engineered features using **SHAP**
  values from a probe XGBoost model; kept features above 1% of the top
  feature's importance.

## Task 2 — Model Training & Evaluation

Compared several regression models with 5-fold cross-validation:

| Model             | CV RMSE (log-price) | CV MAE | CV R² |
|--------------------|:---:|:---:|:---:|
| Linear Regression  | _fill in_ | _fill in_ | _fill in_ |
| Ridge               | _fill in_ | _fill in_ | _fill in_ |
| Lasso               | _fill in_ | _fill in_ | _fill in_ |
| Random Forest       | _fill in_ | _fill in_ | _fill in_ |
| XGBoost             | _fill in_ | _fill in_ | _fill in_ |

**Selected model:** XGBoost — chosen despite not having the best untuned CV
score, because tree-based models have far more tuning headroom than linear
baselines (Lasso/Ridge have essentially one meaningful knob, `alpha`), and
the engineered feature set includes non-linear structure (location
distances, target-encoded neighbourhoods, interaction terms) that linear
models can't fully exploit.

Hyperparameters were tuned via `RandomizedSearchCV` (3-fold CV, 15
iterations) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
`colsample_bytree`, and `min_child_weight`.

**Final test-set performance (real price scale):**

| Metric | Value |
|---|---|
| RMSE | $_fill in_ |
| MAE  | $_fill in_ |
| R²   | _fill in_ |

Train vs. validation RMSE were compared via a learning curve to check for
overfitting/underfitting; see `DS605_Lab4_Task2.ipynb` for the plot.

## Task 3 — Streamlit Application

`app.py` loads the saved pipeline (`final_model_pipeline.pkl`) and exposes
a form for entering listing details (room type, borough, neighbourhood,
coordinates, minimum nights, review history, host info, availability). It
reproduces the exact same feature engineering used in training (distance
calculations, bucketing, neighbourhood target-encoding) before predicting
a nightly price.

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Live app:** _fill in deployed Streamlit Cloud link once available_

### Screenshot

_add a screenshot of the app here, e.g.:_
`![App Screenshot](app_screenshot.png)`

## Task 4 — Summary & Limitations

**Key findings:**
- Location (borough, neighbourhood, distance to Times Square) and room
  type are the strongest predictors of price.
- Review recency and availability provide meaningful but secondary signal.
- _fill in your own observations from the SHAP plots and model comparison_

**Limitations:**
- The model is trained on 2019 data — NYC short-term rental prices have
  shifted since (e.g., Local Law 18 restrictions), so predictions reflect
  historical, not current, market conditions.
- `neighbourhood` target-encoding can be less reliable for very small or
  unseen neighbourhoods, where it falls back toward the global mean.
- No text/image features from listing descriptions or photos are used,
  which likely explains a meaningful share of unexplained price variance.
- The app assumes reasonable/valid user inputs; it does not validate that
  a given lat/long actually falls within NYC.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Streamlit

