# 🌧️ Rainfall Prediction with Machine Learning

## Overview

This project aims to predict the occurrence of rain using meteorological data. The dataset contains various weather-related features, and the goal is to develop and evaluate predictive models using K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees. The project involves data preprocessing, feature engineering, and performance evaluation of different machine learning algorithms.

## Dataset

The dataset spans 10 years of daily meteorological data from various locations across the country, featuring:
- **14,560 samples**
- **22 weather-related attributes**

It includes features related to weather conditions and target variables indicating whether it rained on a particular day and whether it will rain the next day.

For detailed descriptions and additional information, please refer to the [**report.pdf**](report.pdf) file.

## Key Steps

1. **Data Preprocessing** 🔧
   - Load and clean the dataset.
   - Handle missing values with mean, median, or most frequent values.
   - Encode categorical variables using label encoding and one-hot encoding.
   - Generate additional features like 'Temperature Difference'.

2. **Feature Engineering** 🛠️
   - Visualize and explore the dataset.
   - Analyze correlations between features.
   - Apply dimensionality reduction techniques like PCA and SVD.

3. **Model Training and Evaluation** 📊
   - Split the dataset into training and testing sets.
   - Apply Random Under-Sampling to balance the classes.
   - Train and evaluate K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees classifiers.
   - Optimize models by tuning hyperparameters such as the number of components for PCA, the number of neighbors for KNN, and tree depth for Decision Trees.

4. **Results** 
   - Evaluate model performance using accuracy, confusion matrix, and classification reports.
   - Visualize results with scatter plots, PCA-transformed data plots, and confusion matrices.

## Files

- `knn_svm.ipynb and decision_tree.ipynb`: Jupyter notebook containing the complete workflow, including data preprocessing, feature engineering, model training, and evaluation.
- `Dataset.csv`: The raw dataset used for analysis and modeling.
- `report.pdf`: Detailed project report with comprehensive descriptions and analyses.

## Installation

To run this project, ensure you have the following Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

##

This project was part of the Data Mining Course at Amirkabir University of Technology.

Coded by Parsa Khadem and Sarvin Baghi.
