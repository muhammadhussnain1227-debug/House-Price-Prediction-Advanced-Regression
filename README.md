# House Price Prediction using XGBoost
## Project Overview

This project focuses on predicting house sale prices using machine learning with XGBoost Regression.
The dataset consists of ~1200 rows and 81 original features, representing various structural, location-based, and quality-related attributes of houses.

Through feature engineering, preprocessing, and model tuning, a strong baseline regression model was achieved.

## Dataset Description

Rows: ~1200

Original Features: 81

Target Variable: SalePrice

Data Type: Mixed (numerical + categorical)

The dataset required extensive preprocessing to make it suitable for machine learning.

## Feature Engineering & Preprocessing

The following steps were applied:

Handling missing values

Label Encoding / One-Hot Encoding for categorical variables

Feature selection and cleanup

Removing data leakage

Ensuring train–test consistency

Scaling not required due to tree-based model

After feature engineering, the final dataset was fully numerical and ready for XGBoost.


## MODEL : XGBoost Regressor

### Why XGBoost?

Handles non-linearity well

Robust to feature interactions

Performs strongly on tabular data

Less sensitive to feature scaling

## Key Parameters (Simplified)

Objective: reg:squarederror

Boosting: Gradient Boosting Trees

Tuned to reduce overfitting while maintaining performance

## Model Performance
Dataset	RMSE
Training Set	~4,900
Test Set	~25,000

## Interpretation

Low training RMSE shows the model learned patterns effectively

Higher test RMSE indicates expected generalization gap due to:

Limited dataset size

High feature complexity

Real-world noise in housing prices

This performance provides a solid baseline for further improvements.

## Evaluation Metric

Root Mean Squared Error (RMSE)
RMSE was chosen because:

It penalizes large errors

Commonly used in regression and Kaggle-style competitions

Directly interpretable in price units

## 🗂️ Project Structure
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│
├── README.md

## Future Improvements

Cross-validation (K-Fold)

Hyperparameter tuning (GridSearch / Optuna)

Log-transform of target variable

Feature importance pruning

Ensemble models (XGBoost + RandomForest)

MLOps pipeline for deployment

## Key Learnings

Feature engineering matters more than model choice

XGBoost performs exceptionally well on structured data

Preventing data leakage is critical

Train–test RMSE gap highlights real-world challenges

## Tech Stack

Python

Pandas & NumPy

Scikit-learn

XGBoost

Jupyter Notebook

## Acknowledgment

This project was built as part of hands-on learning in Machine Learning and Applied Regression, focusing on real-world model behavior rather than just theory.
