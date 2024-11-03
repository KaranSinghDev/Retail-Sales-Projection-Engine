# Import necessary libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load the primary datasets
sales_train = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\sales_train.csv")
test = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\test.csv")
items = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\items.csv")
item_categories = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\item_categories.csv")
shops = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\shops.csv")

# Convert date to datetime format and extract year and month
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train['year'] = sales_train['date'].dt.year
sales_train['month'] = sales_train['date'].dt.month

# Aggregate daily data to monthly data for each shop-item combination
monthly_sales = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
    'item_cnt_day': 'sum'
}).rename(columns={'item_cnt_day': 'item_cnt_month'}).reset_index()

# Merge item_category_id into monthly_sales from items data
monthly_sales = monthly_sales.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')

# Clip item counts in monthly data to [0, 20]
monthly_sales['item_cnt_month'] = monthly_sales['item_cnt_month'].clip(0, 20)

# Reduced Lag Features (using fewer months)
for lag in range(1, 4):  # Only 3 months to reduce memory load
    monthly_sales[f'item_cnt_month_lag_{lag}'] = monthly_sales.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)

# Mean Encoding
item_mean_sales = monthly_sales.groupby('item_id')['item_cnt_month'].mean().rename('item_avg_sales').reset_index()
shop_mean_sales = monthly_sales.groupby('shop_id')['item_cnt_month'].mean().rename('shop_avg_sales').reset_index()
category_mean_sales = monthly_sales.groupby('item_category_id')['item_cnt_month'].mean().rename('category_avg_sales').reset_index()

# Merge mean sales with monthly sales data
monthly_sales = monthly_sales.merge(item_mean_sales, on='item_id', how='left')
monthly_sales = monthly_sales.merge(shop_mean_sales, on='shop_id', how='left')
monthly_sales = monthly_sales.merge(category_mean_sales, on='item_category_id', how='left')

# Prepare the data for modeling
X = monthly_sales.drop(columns=['item_cnt_month', 'date_block_num'])
y = monthly_sales['item_cnt_month']

# Initialize KFold cross-validation with feedback on fold completion
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store final test predictions
predictions = np.zeros(test.shape[0])  

for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    print(f"Training fold {fold}...")

    # Convert to LightGBM Dataset format
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Initialize and train the LightGBM model with early stopping
    model = lgb.train(
        {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        },
        train_set=train_data,
        num_boost_round=200,  # Reduced for faster execution
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)]  # Early stopping and logging
    )

    # Predict on the validation set
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_val_pred = np.clip(y_val_pred, 0, 20)  # Clip predictions

    # Calculate RMSE on the validation set
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f'Fold {fold} RMSE: {rmse}')

    # Prepare test features with mean encodings
    required_columns = X.columns.tolist()  # Ensure test_features has same columns as X
    test_features = test[['ID', 'item_id', 'shop_id']].copy()
    test_features = test_features.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')

    # Perform mean encoding merges
    test_features = test_features.merge(item_mean_sales, on='item_id', how='left')
    test_features = test_features.merge(shop_mean_sales, on='shop_id', how='left')
    test_features = test_features.merge(category_mean_sales, on='item_category_id', how='left')

    # Fill NaN values in the mean encodings
    test_features.fillna(0, inplace=True)

    # Ensure test_features has all columns in required_columns
    for column in required_columns:
        if column not in test_features.columns:
            test_features[column] = 0  # Add any missing columns with default 0 values

    # Drop unnecessary columns that are not in the model
    test_features = test_features[required_columns]

    # Make predictions on the test set
    predictions += model.predict(test_features, num_iteration=model.best_iteration) / kf.n_splits  # Average predictions from each fold

# Clip predictions for the submission
predictions = np.clip(predictions, 0, 20)

# Prepare the submission file
submission = test[['ID']].copy()
submission['item_cnt_month'] = predictions
submission.to_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Sales Prediction\sample_submission.csv", index=False)
print("Submission file created successfully.")
