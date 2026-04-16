##############################################################################
# MovieLens Recommendation System
# Author: Islam Gamal
# Course: HarvardX Data Science Capstone (edX)
# Dataset: MovieLens 10M
##############################################################################

# ============================================================
# SECTION 0: Install and load required packages
# ============================================================

# I like to make sure all necessary packages are available before starting.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))     install.packages("caret",     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# ============================================================
# SECTION 1: Data Preparation (provided by course)
# ============================================================

# Download the MovieLens 10M dataset
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
movies_file  <- "ml-10M100K/movies.dat"

if(!file.exists(ratings_file))
  unzip(dl, files = "ml-10M100K/ratings.dat")
if(!file.exists(movies_file))
  unzip(dl, files = "ml-10M100K/movies.dat")

# Parse ratings
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c('userId', 'movieId', 'rating', 'timestamp')
ratings <- ratings %>%
  mutate(userId    = as.integer(userId),
         movieId   = as.integer(movieId),
         rating    = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Parse movies
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c('movieId', 'title', 'genres')
movies <- movies %>% mutate(movieId = as.integer(movieId))

# Join ratings and movies
movielens <- left_join(ratings, movies, by = "movieId")

# Create the final holdout test set (10% of movielens data)
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx              <- movielens[-test_index, ]
temp             <- movielens[ test_index, ]

# Make sure userId and movieId in final holdout also appear in edx
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final_holdout_test back into edx
removed <- anti_join(temp, final_holdout_test)
edx     <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, removed)


# ============================================================
# SECTION 2: Exploratory Data Analysis (EDA)
# ============================================================

cat("\n--- Basic Dataset Info ---\n")
cat("edx rows:", nrow(edx), "\n")
cat("edx cols:", ncol(edx), "\n")
cat("Distinct users:", n_distinct(edx$userId), "\n")
cat("Distinct movies:", n_distinct(edx$movieId), "\n")

# Rating distribution
rating_dist <- edx %>%
  group_by(rating) %>%
  summarise(count = n()) %>%
  arrange(rating)
print(rating_dist)

# Visualization 1: Distribution of ratings
p1 <- edx %>%
  ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Movie Ratings",
       x = "Rating", y = "Count") +
  theme_minimal()
print(p1)

# Visualization 2: Number of ratings per movie (log scale)
# Some movies get rated a LOT more than others — this will matter for modeling
p2 <- edx %>%
  count(movieId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 50, fill = "tomato", color = "white") +
  scale_x_log10() +
  labs(title = "Number of Ratings per Movie (log scale)",
       x = "Number of Ratings", y = "Count of Movies") +
  theme_minimal()
print(p2)

# Visualization 3: Number of ratings per user
p3 <- edx %>%
  count(userId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  scale_x_log10() +
  labs(title = "Number of Ratings per User (log scale)",
       x = "Number of Ratings", y = "Count of Users") +
  theme_minimal()
print(p3)

# Quick look: top 10 most-rated movies
edx %>%
  group_by(movieId, title) %>%
  summarise(n = n(), avg_rating = mean(rating)) %>%
  arrange(desc(n)) %>%
  head(10) %>%
  print()


# ============================================================
# SECTION 3: Define RMSE function
# ============================================================

RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# ============================================================
# SECTION 4: Create a validation split from edx
# (I will NOT touch final_holdout_test until the very end)
# ============================================================

set.seed(1, sample.kind = "Rounding")
val_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-val_index, ]
val_temp  <- edx[ val_index, ]

# Ensure validation set only has users/movies seen in training
val_set <- val_temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed_val <- anti_join(val_temp, val_set)
train_set   <- rbind(train_set, removed_val)
rm(val_index, val_temp, removed_val)

cat("train_set rows:", nrow(train_set), "\n")
cat("val_set rows:",   nrow(val_set),   "\n")


# ============================================================
# SECTION 5: Model 1 — Baseline (Global Mean)
# ============================================================
# The simplest possible model: just predict the overall mean rating for everyone.
# Not great, but it gives us a starting point to beat.

mu_hat <- mean(train_set$rating)
cat("\nGlobal mean rating:", mu_hat, "\n")

rmse_baseline <- RMSE(val_set$rating, mu_hat)
cat("Model 1 (Baseline) RMSE:", rmse_baseline, "\n")

# Store results in a table for easy comparison later
results_table <- tibble(
  Model = "1. Baseline (global mean)",
  RMSE  = rmse_baseline
)


# ============================================================
# SECTION 6: Model 2 — Movie Effect
# ============================================================
# I noticed in the EDA that some movies consistently get higher or lower ratings.
# Adding a "movie bias" (b_i) should help a lot.
#
# Model: Y = mu + b_i + epsilon
# b_i = mean(rating - mu) for each movie

mu <- mean(train_set$rating)

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

# Predicted ratings on validation set
predicted_movie <- val_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

rmse_movie <- RMSE(val_set$rating, predicted_movie)
cat("Model 2 (Movie Effect) RMSE:", rmse_movie, "\n")

results_table <- bind_rows(results_table,
  tibble(Model = "2. Movie Effect", RMSE = rmse_movie))


# ============================================================
# SECTION 7: Model 3 — Movie + User Effect
# ============================================================
# Even after accounting for movie bias, some users just rate everything high (or low).
# So I added a user bias (b_u) on top.
#
# Model: Y = mu + b_i + b_u + epsilon
# b_u = mean(rating - mu - b_i) for each user

user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

predicted_user <- val_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs,  by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_user <- RMSE(val_set$rating, predicted_user)
cat("Model 3 (Movie + User Effect) RMSE:", rmse_user, "\n")

results_table <- bind_rows(results_table,
  tibble(Model = "3. Movie + User Effect", RMSE = rmse_user))


# ============================================================
# SECTION 8: Model 4 — Regularized Movie + User Effect
# ============================================================
# The problem with b_i and b_u: if a movie only has 1 or 2 ratings,
# its bias estimate can be huge and unreliable. Regularization shrinks
# these estimates toward zero using a penalty lambda.
#
# Regularized b_i = sum(ratings - mu) / (n_i + lambda)
# Regularized b_u = sum(ratings - mu - b_i) / (n_u + lambda)

# --- Tune lambda using validation set ---
# I'll search over a range of lambda values and pick the one that gives lowest RMSE.

lambdas <- seq(0, 10, 0.25)

rmse_by_lambda <- sapply(lambdas, function(l) {
  
  # Regularized movie bias
  b_i_reg <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n() + l))
  
  # Regularized user bias
  b_u_reg <- train_set %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n() + l))
  
  # Predict on validation set
  preds <- val_set %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  RMSE(val_set$rating, preds)
})

# Find best lambda
best_lambda <- lambdas[which.min(rmse_by_lambda)]
cat("\nBest lambda:", best_lambda, "\n")
cat("RMSE at best lambda:", min(rmse_by_lambda), "\n")

# Visualize lambda tuning
p4 <- tibble(lambda = lambdas, rmse = rmse_by_lambda) %>%
  ggplot(aes(x = lambda, y = rmse)) +
  geom_line(color = "steelblue") +
  geom_vline(xintercept = best_lambda, linetype = "dashed", color = "red") +
  labs(title = "Lambda Tuning: RMSE vs Regularization Parameter",
       x = "Lambda", y = "RMSE on Validation Set") +
  theme_minimal()
print(p4)

rmse_regularized_val <- min(rmse_by_lambda)

results_table <- bind_rows(results_table,
  tibble(Model = paste0("4. Regularized (lambda=", best_lambda, ") [Validation]"),
         RMSE  = rmse_regularized_val))

# Print intermediate results
cat("\n--- Model Comparison (on Validation Set) ---\n")
print(results_table)


# ============================================================
# SECTION 9: Final Model — Train on Full edx, Evaluate on final_holdout_test
# ============================================================
# Now that I've selected lambda, I retrain on the FULL edx dataset
# and only NOW touch final_holdout_test for the very first and last time.

mu_final <- mean(edx$rating)

b_i_final <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu_final) / (n() + best_lambda))

b_u_final <- edx %>%
  left_join(b_i_final, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu_final - b_i) / (n() + best_lambda))

# Final predictions on holdout test set
final_predictions <- final_holdout_test %>%
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  mutate(pred = mu_final + b_i + b_u) %>%
  pull(pred)

rmse_final <- RMSE(final_holdout_test$rating, final_predictions)
cat("\n==============================================\n")
cat("FINAL RMSE on final_holdout_test:", rmse_final, "\n")
cat("==============================================\n")

# Add final result to comparison table
results_table <- bind_rows(results_table,
  tibble(Model = paste0("5. FINAL — Full edx + Holdout Test (lambda=", best_lambda, ")"),
         RMSE  = rmse_final))

cat("\n--- Final Results Summary ---\n")
print(results_table)

# ============================================================
# END OF SCRIPT
# ============================================================
