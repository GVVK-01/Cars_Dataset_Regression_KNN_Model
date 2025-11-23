# Cars_Dataset_Regression_KNN_Model
A complete CRISP-ML based regression project using KNN to predict vehicle MPG with end-to-end data preprocessing, modeling, and evaluation.

This project applies the CRISP-ML (Cross-Industry Standard Process for Machine Learning) framework to build a regression model that predicts Miles Per Gallon (MPG) based on vehicle engine and design features from the Auto MPG dataset.

The goal is to demonstrate a complete, structured, and industry-standard ML workflow from business understanding to model evaluation.
Using CRISP-ML ensures the project is reproducible, explainable, well-documented, and aligned with real-world ML pipeline standards.
The model chosen for this project is K-Nearest Neighbors (KNN) Regression, which performs distance-based prediction and requires careful preprocessing such as outlier treatment and feature scaling.

🎯 Objectives

Understand the business problem and define success criteria
Explore and analyze the Auto MPG dataset
Detect and treat outliers using IQR-based winsorization
Prepare data through feature selection, splitting, and scaling
Build a KNN Regression model

Evaluate model performance using standard regression metrics

🏗️ CRISP-ML Steps Covered

Business Understanding
Define project goals and deliverables
Identify target and features
Data Acquisition
Load dataset and inspect schema

Data Understanding (EDA)

Statistical summaries, correlations, skew analysis
Visualizations: pairplots, heatmaps
Data Preparation
Handle missing values
Outlier capping using IQR (Winsorization)
Log transforms for skewed features (if needed)

Train-test split
Standardization using StandardScaler
Modeling (KNN Regression)
Train baseline KNN model
Predict MPG values on test data

Evaluation

Metrics: MAE, MSE, RMSE, R²
Predicted vs Actual scatter plot
Residual distribution

📈 Key Features

Clean preprocessing pipeline suitable for distance-based models
Outlier handling using IQR Winsorization
Scalable and reproducible ML workflow
High-quality code structured for deployment and extension
Ready to push directly to GitHub

🛠️ Tech Stack

Python
Pandas, NumPy
Scikit-Learn
Matplotlib, Seaborn

Jupyter Notebook
