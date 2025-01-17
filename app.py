import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
# Load the cleaned datasets

products = pd.read_csv("cleaned_product_data.csv")
customers = pd.read_csv("cleaned_customer_data.csv")
transactions = pd.read_csv("cleaned_transaction_data.csv")

# -------------------
# Data Analysis
# -------------------

# 1. Key Performance Indicators (KPIs)
total_sales = transactions["total_amount"].sum()
total_customers = customers["customer_id"].nunique()
top_product = transactions.groupby("product_id")["total_amount"].sum().idxmax()
top_product_name = products[products["product_id"] == top_product]["product_name"].values[0]

# 2. Seasonal Trends
transactions["transaction_month"] = pd.to_datetime(transactions["transaction_date"]).dt.to_period("M").astype(str)
monthly_sales = transactions.groupby("transaction_month")["total_amount"].sum().reset_index()

# 3. Customer Segmentation
customer_spending = transactions.groupby("customer_id")["total_amount"].sum().reset_index()
customer_spending["spending_category"] = pd.qcut(customer_spending["total_amount"], 4, labels=["Low", "Medium", "High", "Very High"])

# 4. Top Categories
category_sales = transactions.merge(products, on="product_id").groupby("category")["total_amount"].sum().reset_index()

# 5. Top Customers
top_customers = transactions.groupby("customer_id")["total_amount"].sum().nlargest(10).reset_index()
top_customers = top_customers.merge(customers, on="customer_id")

# -------------------
# Predictive Analysis
# -------------------

# Prepare data for predictive analysis
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
transactions["day_of_week"] = transactions["transaction_date"].dt.dayofweek
transactions["month"] = transactions["transaction_date"].dt.month

# Aggregate data for modeling
model_data = transactions.groupby(["transaction_date", "day_of_week", "month"])["total_amount"].sum().reset_index()

# Train-test split
X = model_data[["day_of_week", "month"]]
y = model_data["total_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
predictions = model.predict(X_test)
model_data["predicted_sales"] = model.predict(model_data[["day_of_week", "month"]])

# Future prediction for the next 30 days
future_dates = pd.date_range(start=model_data["transaction_date"].max() + timedelta(days=1), periods=30)
future_data = pd.DataFrame({
    "transaction_date": future_dates,
    "day_of_week": future_dates.dayofweek,
    "month": future_dates.month
})
future_data["predicted_sales"] = model.predict(future_data[["day_of_week", "month"]])

# -------------------
# Streamlit Dashboard
# -------------------

st.title("E-Commerce Sales Analysis Dashboard")

# KPI Section
st.sidebar.header("Key Performance Indicators")
st.sidebar.metric("Total Sales", f"${total_sales:,.2f}")
st.sidebar.metric("Total Customers", total_customers)
st.sidebar.metric("Top Product", top_product_name)

# Seasonal Trends
st.header("Seasonal Trends")
chart_type = st.radio("Choose a chart type for seasonal trends:", ["Line Chart", "Bar Chart"])
if chart_type == "Line Chart":
    st.line_chart(data=monthly_sales, x="transaction_month", y="total_amount", use_container_width=True)
elif chart_type == "Bar Chart":
    st.bar_chart(data=monthly_sales, x="transaction_month", y="total_amount", use_container_width=True)

# Customer Segmentation
st.header("Customer Segmentation")
fig, ax = plt.subplots()
sns.countplot(data=customer_spending, x="spending_category", ax=ax, palette="viridis")
ax.set_title("Customer Spending Categories")
st.pyplot(fig)

# Top Categories
st.header("Top Categories by Sales")
fig, ax = plt.subplots()
sns.barplot(data=category_sales, x="category", y="total_amount", ax=ax, palette="coolwarm")
ax.set_title("Category-wise Sales")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig)

# Top Products
st.header("Top Products by Sales")
top_products = transactions.merge(products, on="product_id").groupby("product_name")["total_amount"].sum().nlargest(10).reset_index()
fig, ax = plt.subplots()
sns.barplot(data=top_products, x="total_amount", y="product_name", ax=ax, palette="magma")
ax.set_title("Top 10 Products by Sales")
st.pyplot(fig)

# Top Customers
st.header("Top Customers by Sales")
fig, ax = plt.subplots()
sns.barplot(data=top_customers, x="total_amount", y="name", ax=ax, palette="cubehelix")
ax.set_title("Top 10 Customers by Spending")
st.pyplot(fig)

# Predictive Analysis
st.header("Predictive Sales Analysis")
st.subheader("Historical and Predicted Sales")
prediction_chart = model_data[["transaction_date", "total_amount", "predicted_sales"]]
st.line_chart(data=prediction_chart, x="transaction_date", y=["total_amount", "predicted_sales"], use_container_width=True)

st.subheader("Future Sales Prediction")
st.line_chart(data=future_data, x="transaction_date", y="predicted_sales", use_container_width=True)

# A/B Testing
st.header("A/B Testing")
st.write("Simulate A/B test results based on customer groups.")
# Simulate A/B test data
a_group = customer_spending.sample(frac=0.5, random_state=42)
b_group = customer_spending.drop(a_group.index)
a_conversion_rate = a_group[a_group["spending_category"] == "Very High"].shape[0] / a_group.shape[0]
b_conversion_rate = b_group[b_group["spending_category"] == "Very High"].shape[0] / b_group.shape[0]

st.write(f"Conversion Rate for Group A: {a_conversion_rate:.2%}")
st.write(f"Conversion Rate for Group B: {b_conversion_rate:.2%}")


from datetime import timedelta

# -------------------
# Load Data
# -------------------
products = pd.read_csv("cleaned_product_data.csv")
customers = pd.read_csv("cleaned_customer_data.csv")
transactions = pd.read_csv("cleaned_transaction_data.csv")

# Ensure correct data types
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

# -------------------
# KPI Section
# -------------------
total_sales = transactions["total_amount"].sum()
total_customers = customers["customer_id"].nunique()
top_product = transactions.groupby("product_id")["total_amount"].sum().idxmax()
top_product_name = products[products["product_id"] == top_product]["product_name"].values[0]

# -------------------
# ARIMA Model
# -------------------
st.header("ARIMA Time-Series Forecasting")

# Aggregate sales data for ARIMA
time_series_data = transactions.groupby("transaction_date")["total_amount"].sum().asfreq("D").fillna(0)

# Fit ARIMA Model
arima_model = SARIMAX(time_series_data, order=(0, 1, 0), seasonal_order=(1, 1, 1, 12))
arima_fit = arima_model.fit(disp=False)

# Forecast next 30 days
forecast_days = 30
forecast = arima_fit.get_forecast(steps=forecast_days)
forecast_index = pd.date_range(start=time_series_data.index.max() + timedelta(days=1), periods=forecast_days)
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)

# Plot ARIMA Results
fig, ax = plt.subplots(figsize=(10, 5))
time_series_data.plot(ax=ax, label="Historical Data")
forecast_series.plot(ax=ax, label="ARIMA Forecast", color="red")
ax.legend()
st.pyplot(fig)

# -------------------
# XGBoost Model
# -------------------
st.header("XGBoost Regression for Sales Prediction")

from datetime import timedelta

# Ensure the transaction_date column is in datetime format
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

# Feature Engineering: Adding Date-related Features
transactions["day_of_week"] = transactions["transaction_date"].dt.dayofweek
transactions["month"] = transactions["transaction_date"].dt.month

# Feature Engineering: Adding Lag Features
transactions["lag_1"] = transactions["total_amount"].shift(1)
transactions["lag_2"] = transactions["total_amount"].shift(2)
transactions["lag_3"] = transactions["total_amount"].shift(3)
transactions.dropna(inplace=True)  # Drop rows with missing values from lag features

# Prepare features and target
X = transactions[["day_of_week", "month", "lag_1", "lag_2", "lag_3"]]
y = transactions["total_amount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare the data in the DMatrix format required by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Cross-validation to find the best hyperparameters using xgb.cv
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    early_stopping_rounds=10,
    nfold=5,
    verbose_eval=10
)

# Best number of boosting rounds
best_num_rounds = cv_results['test-mae-mean'].idxmin()

# Train the XGBoost model with the best number of rounds
final_model = xgb.train(params, dtrain, num_boost_round=best_num_rounds)

# Make predictions
y_pred = final_model.predict(dtest)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output evaluation metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Future Predictions using XGBoost
future_dates = pd.date_range(start=transactions["transaction_date"].max() + timedelta(days=1), periods=30)
future_data_xgb = pd.DataFrame({
    "day_of_week": future_dates.dayofweek,
    "month": future_dates.month,
    "lag_1": [transactions["total_amount"].iloc[-1]] * 30,  # Previous day's sales
    "lag_2": [transactions["total_amount"].iloc[-2]] * 30,  # Two days ago sales
    "lag_3": [transactions["total_amount"].iloc[-3]] * 30   # Three days ago sales
})

# Convert to DMatrix format for future data prediction
dfuture = xgb.DMatrix(future_data_xgb)

# Make future predictions
future_predictions = final_model.predict(dfuture)

# Display future predictions
future_data_xgb["predicted_sales"] = future_predictions
print(future_data_xgb[['day_of_week', 'month', 'predicted_sales']])
# -------------------
# Streamlit Dashboard with Dynamic Filtering
# -------------------

# Time Range Filter
start_date, end_date = st.sidebar.date_input(
    "Select Date Range", 
    value=[transactions["transaction_date"].min(), transactions["transaction_date"].max()],
    min_value=transactions["transaction_date"].min(),
    max_value=transactions["transaction_date"].max()
)

# Convert start_date and end_date to datetime64
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Product Categories Filter
categories = transactions.merge(products, on="product_id")["category"].unique()
selected_categories = st.sidebar.multiselect("Select Product Categories", categories, default=categories)

# Customer Segments Filter
spending_categories = customer_spending["spending_category"].unique()
selected_spending_categories = st.sidebar.multiselect(
    "Select Spending Categories", 
    spending_categories, 
    default=spending_categories
)

# Filter Data Based on User Input
filtered_transactions = transactions[
    (transactions["transaction_date"] >= start_date) &
    (transactions["transaction_date"] <= end_date)
]
filtered_transactions = filtered_transactions[filtered_transactions["product_id"].isin(products[products["category"].isin(selected_categories)]["product_id"])]
filtered_customer_spending = customer_spending[customer_spending["spending_category"].isin(selected_spending_categories)]
