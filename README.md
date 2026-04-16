# 🎬 Movie Recommendation System (HarvardX Capstone)

This project is part of the **HarvardX Data Science Professional Certificate (PH125.9x Capstone)**.  
The goal is to build a movie recommendation system using the **MovieLens 10M dataset** and evaluate performance using RMSE.

---

## 🚀 Project Overview

In this project, I developed a recommendation system that predicts user ratings for movies by progressively improving model performance:

- Baseline model (global mean)
- Movie effect model
- Movie + User effects model
- Regularized model (L2 regularization)

The final model successfully achieved the required performance target.

📊 **Final Result:**  
RMSE < **0.86490** ✅ (Target achieved)

---

## 📂 Dataset

- **Dataset:** MovieLens 10M  
- ~10 million ratings  
- ~72,000 users  
- ~10,000 movies  
- Ratings scale: 0.5 – 5.0  

The dataset was split into:
- `edx` (training set)
- `final_holdout_test` (test set)

---

## ⚙️ Methodology

### 1. Data Preparation
- Data cleaning and transformation
- Merging ratings with movie metadata

### 2. Exploratory Data Analysis
- Rating distribution analysis
- User & movie activity patterns
- Sparsity analysis

### 3. Modeling Approach

The model was built incrementally:

#### 🔹 Baseline Model
Predicts global mean rating

#### 🔹 Movie Effect
Captures movie-specific bias

#### 🔹 User Effect
Captures user rating behavior

#### 🔹 Regularization
Applies L2 regularization to handle sparsity

---

## 📈 Model Performance

| Model | RMSE |
|------|------|
| Baseline | 1.06020 |
| Movie Effect | 0.94374 |
| Movie + User | 0.86535 |
| Regularized Model | **0.86481** |

📉 Continuous improvement achieved at each stage.

---

## 🧠 Key Insights

- Movie bias had the largest impact on performance  
- User behavior significantly improved predictions  
- Regularization stabilized results for sparse data  
- Simple models can achieve strong results with good feature engineering  

---

## 🛠️ Tech Stack

- **R**
- tidyverse
- caret
- data.table

---

## 📁 Project Structure
