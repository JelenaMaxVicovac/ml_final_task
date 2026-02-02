
# Product Category Prediction Model
## Overview

This project implements a machine learning model that predicts the product category based on the product title.
The model is designed to work with short and noisy product names, such as those found in online marketplaces.

The solution uses a scikit-learn Pipeline to ensure consistent preprocessing and prediction.

## Dataset

The model was trained on a cleaned product dataset containing product titles and their corresponding category labels.

## Basic preprocessing steps included:

 - removing missing values

- standardizing column names and category labels

- merging similar category names into unified labels

- Feature Engineering

**In addition to the raw product title, several features were created:**

TF-IDF vectors from:

product_title

title_content_check (information about numbers and special characters)

Numerical features:

length of the product title

maximum word length in the title
(scaled using MinMaxScaler)

All feature engineering and preprocessing steps are encapsulated inside a ColumnTransformer.

 # Model

## Algorithm: RandomForestClassifier

### Architecture:
Preprocessing (ColumnTransformer) → Classifier (RandomForestClassifier)

The entire workflow is saved as a single trained pipeline.

### Usage

A trained model pipeline is stored in:

model/train_the_best_model.pkl


An interactive Python script allows users to enter a product title and receive a predicted category in real time.

### Example:

Enter product title: Samsung Galaxy S21 128GB
Predicted category: mobile phone

Technologies Used

Python

pandas

scikit-learn

joblib

# Notes

The model expects input data in the same format used during training.
All preprocessing steps are handled automatically by the saved pipeline, ensuring reliable and consistent predictions.