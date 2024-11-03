# Retail Sales Projection Engine

## Problem
Retailers often face challenges in accurately forecasting sales, which can lead to issues such as overstocking or stockouts. Predicting future sales is crucial for effective inventory management, supply chain optimization, and maximizing revenue.

## Solution
This Sales Prediction System employs a LightGBM regression model to predict monthly sales for various products across different shops. By analyzing historical sales data and extracting relevant features, the system provides accurate forecasts that enable retailers to make informed inventory and demand planning decisions.

## Dataset
The dataset used in this project is sourced from Kaggle's "Predict Future Sales" competition. It includes historical sales data, item details, shop information, and other relevant features necessary for building the prediction model.

- **Dataset URL:** [Kaggle - Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)

## Model
The model implemented in this project is LightGBM with the following hyperparameters:

- **n_estimators:** 1000
- **learning_rate:** 0.01
- **max_depth:** 8
- **random_state:** 42
- **subsample:** 0.8
- **colsample_bytree:** 0.8

These hyperparameters were chosen to balance model complexity and training time, resulting in a robust performance.

## Evaluation Score
The model's performance is evaluated using the Root Mean Squared Error (RMSE) metric. The validation RMSE score achieved during testing is indicative of the model's accuracy in predicting sales.

## Citation
For further reference, please cite the original dataset creators:
Alexander Guschin, Dmitry Ulyanov, inversion, Mikhail Trofimov, utility, and Μαριος Μιχαηλιδης KazAnova. Predict Future Sales. [Kaggle Competition Link](https://kaggle.com/competitions/competitive-data-science-predict-future-sales), 2018. Kaggle.
