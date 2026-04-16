# Movie Recommendation System (HarvardX Capstone)

This project was completed as part of the HarvardX Data Science Professional Certificate (PH125.9x Capstone). The goal is to build a movie recommendation system using the MovieLens 10M dataset and evaluate its performance using Root Mean Squared Error (RMSE).

## Project Overview

The objective of this project is to predict the rating a user would give to a movie. The model was developed step by step, starting from a simple baseline and gradually improving performance by incorporating additional factors such as movie effects, user effects, and regularization.

The final model achieved an RMSE below 0.86490 on the held-out test set, meeting the course requirement.

## Dataset

The project uses the MovieLens 10M dataset, which contains:
- Approximately 10 million ratings
- Around 72,000 users
- Around 10,000 movies
- Ratings ranging from 0.5 to 5.0

The dataset was split into:
- edx: used for training and validation
- final_holdout_test: used only once for final evaluation

## Methodology

### Data Preparation
The raw data was processed and merged into a single dataset containing user IDs, movie IDs, ratings, and timestamps. Care was taken to ensure consistency between training and test sets.

### Exploratory Data Analysis
Initial analysis showed:
- Ratings are skewed toward higher values (mostly 3 to 5)
- Some movies have many ratings while most have very few
- User activity varies significantly

These observations guided the modeling decisions.

### Modeling Approach

The model was built in stages:

1. Baseline Model  
Predicts the global average rating for all user-movie pairs.

2. Movie Effect Model  
Adds a movie-specific bias to account for differences in movie quality.

3. Movie and User Effects Model  
Adds a user-specific bias to capture individual rating behavior.

4. Regularized Model  
Applies L2 regularization to improve stability, especially for movies and users with few ratings.

## Model Performance

The performance improved at each step:

- Baseline: 1.06020  
- Movie effect: 0.94374  
- Movie + user effects: 0.86535  
- Regularized model: 0.86481  

Final result on test set: RMSE below 0.86490

## Key Insights

- Movie-specific effects contributed the most to improving accuracy
- User behavior patterns further refined predictions
- Regularization helped reduce overfitting caused by sparse data
- A simple and well-structured model can achieve strong results

## Tools and Technologies

- R
- tidyverse
- caret
- data.table

## Project Structure

- 0-movielens_islam_gamal.Rmd  
- 1-movielens_islam_gamal.R  
- 2-movielens_report_islam_gamal.pdf  

Full report is included in the repository with detailed analysis and results.

## Conclusion

This project shows that combining exploratory data analysis with a structured modeling approach can lead to effective and reliable recommendation systems. The final model is simple, interpretable, and performs well on unseen data.

## Contact

Islam Gamal  
LinkedIn: https://www.linkedin.com/in/islamgamalig

Open to opportunities in data science and machine learning.
