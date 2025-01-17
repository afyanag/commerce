import pandas as pd

# File paths
product_path = "product_data.csv"
customer_path = "customer_data.csv"
transaction_path = "transaction_data.csv"

# Load the datasets
products = pd.read_csv(product_path)
customers = pd.read_csv(customer_path)
transactions = pd.read_csv(transaction_path)

# -------------------
# Data Cleaning Steps
# -------------------

# 1. Check for duplicates
products.drop_duplicates(inplace=True)
customers.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)

# 2. Handle missing values
# Assuming no missing values should exist; otherwise, we would impute or drop as appropriate
assert not products.isnull().any().any(), "Products table contains missing values"
assert not customers.isnull().any().any(), "Customers table contains missing values"
assert not transactions.isnull().any().any(), "Transactions table contains missing values"

# 3. Ensure data types are correct
products["price"] = products["price"].astype(float)
products["stock"] = products["stock"].astype(int)

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
transactions["quantity"] = transactions["quantity"].astype(int)
transactions["total_amount"] = transactions["total_amount"].astype(float)

# 4. Validate relationships between tables
# Ensure all foreign keys in transactions exist in their respective tables
valid_customers = set(customers["customer_id"])
valid_products = set(products["product_id"])

transactions = transactions[transactions["customer_id"].isin(valid_customers)]
transactions = transactions[transactions["product_id"].isin(valid_products)]

# 5. Reset indices for cleaned data
products.reset_index(drop=True, inplace=True)
customers.reset_index(drop=True, inplace=True)
transactions.reset_index(drop=True, inplace=True)

# Save cleaned data
products.to_csv("cleaned_product_data.csv", index=False)
customers.to_csv("cleaned_customer_data.csv", index=False)
transactions.to_csv("cleaned_transaction_data.csv", index=False)

print("Data cleaning completed and saved to new files.")