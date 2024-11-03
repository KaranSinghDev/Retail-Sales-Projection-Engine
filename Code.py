# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load the primary datasets
sales_train = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\sales_train.csv")
test = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\test.csv")
sample_submission = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\sample_submission.csv")

# Load supplementary datasets
items = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\items.csv")
item_categories = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\item_categories.csv")
shops = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\shops.csv")

# Display the first few rows of each dataset for verification
print("Sales Train Data")
print(sales_train.head())

print("\nTest Data")
print(test.head())

print("\nSample Submission")
print(sample_submission.head())

print("\nItems Data")
print(items.head())

print("\nItem Categories")
print(item_categories.head())

print("\nShops Data")
print(shops.head())

# Check for missing values in each dataset
print("Missing values in Sales Train Data:", sales_train.isnull().sum())
print("Missing values in Test Data:", test.isnull().sum())
print("Missing values in Items Data:", items.isnull().sum())
print("Missing values in Item Categories:", item_categories.isnull().sum())
print("Missing values in Shops Data:", shops.isnull().sum())

# Convert date to datetime format and extract year and month
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train['year'] = sales_train['date'].dt.year
sales_train['month'] = sales_train['date'].dt.month

# Merge supplemental data into sales_train and test
sales_train = sales_train.merge(items, on='item_id', how='left')
sales_train = sales_train.merge(item_categories, on='item_category_id', how='left')
sales_train = sales_train.merge(shops, on='shop_id', how='left')

# For the test set, merge similarly on 'item_id' and 'shop_id'
test = test.merge(items, on='item_id', how='left')
test = test.merge(item_categories, on='item_category_id', how='left')
test = test.merge(shops, on='shop_id', how='left')

# Aggregate daily data to monthly data for each shop-item combination
monthly_sales = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
    'item_cnt_day': 'sum'
}).rename(columns={'item_cnt_day': 'item_cnt_month'}).reset_index()

# Merge item_category_id into monthly_sales from items data
monthly_sales = monthly_sales.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')

# Step 6.1: Clip item counts in monthly data to [0, 20] as required by the competition
monthly_sales['item_cnt_month'] = monthly_sales['item_cnt_month'].clip(0, 20)

# Step 6.2: Handle negative sales counts by setting them to 0 (assuming returns or adjustments)
sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(lower=0)

# Step 7.1: Add lag features for previous month sales
# Shift monthly sales to create lag features (e.g., previous month’s sales)
for lag in [1, 2, 3]:
    monthly_sales[f'item_cnt_month_lag_{lag}'] = monthly_sales.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)

# Step 7.2: Mean Encoding - calculate mean sales for each item, shop, and category
# This will give an idea of the average monthly sales for each category, shop, and item
item_mean_sales = monthly_sales.groupby('item_id')['item_cnt_month'].mean().rename('item_avg_sales').reset_index()
shop_mean_sales = monthly_sales.groupby('shop_id')['item_cnt_month'].mean().rename('shop_avg_sales').reset_index()
category_mean_sales = monthly_sales.groupby('item_category_id')['item_cnt_month'].mean().rename('category_avg_sales').reset_index()

# Merge these features with the monthly sales data
monthly_sales = monthly_sales.merge(item_mean_sales, on='item_id', how='left')
monthly_sales = monthly_sales.merge(shop_mean_sales, on='shop_id', how='left')
monthly_sales = monthly_sales.merge(category_mean_sales, on='item_category_id', how='left')

# Display the modified data with new features for verification
print(monthly_sales.head())

# Step 8: Prepare the data for modeling by splitting into training and validation sets
# Define the last date_block_num for training data, using the final month as validation
X_train = monthly_sales[monthly_sales['date_block_num'] < 33]
X_val = monthly_sales[monthly_sales['date_block_num'] == 33]

# Target variable
y_train = X_train['item_cnt_month']
y_val = X_val['item_cnt_month']

# Drop columns not needed for modeling
X_train = X_train.drop(columns=['item_cnt_month', 'date_block_num'])
X_val = X_val.drop(columns=['item_cnt_month', 'date_block_num'])

# Display the shapes of training and validation sets for confirmation
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)



# Initialize the LightGBM model with basic parameters
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train the model on the training data without 'verbose' parameter
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Predict on the validation set
y_val_pred = model.predict(X_val)

# Clip predictions to the range [0, 20] as required by the competition
y_val_pred = y_val_pred.clip(0, 20)

# Calculate RMSE on the validation set
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Validation RMSE: {rmse}")



# Step 1: Prepare the test set to have the same features as the training data
# Add lag features to the test data and fill with 0 since future lags aren't available
test['item_cnt_month_lag_1'] = 0
test['item_cnt_month_lag_2'] = 0
test['item_cnt_month_lag_3'] = 0

# Step 1: Ensure the test set has matching columns with the training data
test = test.merge(item_mean_sales, on='item_id', how='left')
test = test.merge(shop_mean_sales, on='shop_id', how='left')
test = test.merge(category_mean_sales, on='item_category_id', how='left')

# Fill any NaN values in the merging process
test.fillna(0, inplace=True)

# Ensure the test set has the same columns as the training set
# Get the columns used in training
train_columns = X_train.columns.tolist()

# Prepare the test set features
test_features = test[train_columns].copy()

# Step 2: Make predictions on the test set
test_predictions = model.predict(test_features)

# Clip predictions to the range [0, 20]
test_predictions = test_predictions.clip(0, 20)

# Step 4: Prepare the submission file
submission = test[['ID']].copy()
submission['item_cnt_month'] = test_predictions

# Display the first few rows of the submission file
print(submission.head())

# Save the submission file
submission.to_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\sample_submission.csv", index=False)
print("Submission file created successfully.")
