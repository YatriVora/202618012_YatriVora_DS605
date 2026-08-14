# DS605 - Web Scraping and Data Analysis

## Student Details

- **Name:** Yatri Vora
- **Student ID:** 202618012
- **Course:** DS605
- **Assignment:** Book Scraping, Data Preprocessing, Feature Engineering and Exploratory Data Analysis

---

## Project Overview

This project involves scraping book data from the *Books to Scrape* website using Python. The collected data was cleaned, preprocessed, and enhanced through feature engineering. Exploratory Data Analysis (EDA) was then performed to identify patterns in book prices, ratings, categories, stock availability, and value scores using statistical summaries and visualizations.

---

## Project Files

- `book_pipeline.ipynb` – Complete implementation of web scraping, preprocessing, feature engineering, and EDA.
- `books_cleaned.csv` – Cleaned dataset generated after preprocessing.
- `*.png` – Visualizations generated during exploratory data analysis.

---

## Summary of Results

- The analysis shows that book prices have little influence on ratings, as price distributions overlap across all rating levels.
- 5-star books have a comparatively lower median price, indicating that highly rated books are not necessarily expensive.
- A few categories contain the majority of books, while average prices differ across categories, showing variations in pricing among genres.
- The engineered **Value Score (Rating ÷ Price)** identifies books that provide the best value by combining high ratings with lower prices.
- Descriptive statistics show that most books are moderately priced, with only a few high-priced outliers, while stock levels remain fairly consistent across the dataset.

---

## Conclusion

The project successfully demonstrates the complete data analysis pipeline, including web scraping, data cleaning, feature engineering, visualization, and interpretation of results. Although the dataset is limited to books available on the *Books to Scrape* website, it provides meaningful insights into pricing patterns, ratings, category distribution, and value for money.
