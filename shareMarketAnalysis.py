import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#Loading the CSV file
nifty = pd.read_csv("NIFTY.csv")
print(nifty.head())

nifty['Date'] = pd.to_datetime(nifty['Date'], format="%d-%b-%y", dayfirst=True)
nifty.set_index('Date', inplace=True)
nifty['Open'] = pd.to_numeric(nifty['Open'], errors='coerce')
nifty['High'] = pd.to_numeric(nifty['High'], errors='coerce')
nifty['Low'] = pd.to_numeric(nifty['Low'], errors='coerce')
nifty = nifty.sort_index(ascending=True)
nifty.dropna(inplace=True)
nifty.head()

return_periods = [1, 2, 3, 4, 7, 10, 14, 15, 16, 30, 90, 180, 365]

# Create new DataFrame to store returns
returns_df = nifty.copy()
for period in return_periods:
 returns_df[f"{period}D_return"] = nifty["Close"].pct_change(periods=period) * 100
returns_df.head()
returns_df.dropna().head()

# Create target variable (return after 1 month in future)
returns_df["Target_30D_return"] = nifty["Close"].pct_change(periods=-
15) * 100
returns_df.tail(30)

# Drop NaN values
data_clean = returns_df.dropna()
data_clean.head()

#TRAIN-TEST SPLIT
# Define features and target
X = data_clean[[f"{period}D_return" for period in return_periods]]
y = data_clean["Target_30D_return"]
X.head()
y.head()

# Split last 1 year as test data
split_date = X.index.max() - pd.DateOffset(years=1)
X_train = X[X.index < split_date]
X_test = X[X.index >= split_date]
y_train = y[y.index < split_date]
y_test = y[y.index >= split_date]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_train preview:\n", X_train.head())
print("y_train preview:\n", y_train.head())

print("Missing values in X_train:\n", X_train.isnull().sum())

# Drop NaN values (if any)
X_train = X_train.dropna()
y_train = y_train.dropna()

model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Convert to BINARY CLASSIFICATION DATASET (Positive: 1, Negative: 0)
y_test_binary = (y_test > 0).astype(int)
y_pred_binary = (y_pred > 0).astype(int)
y_test_binary

# Create DataFrame for actual and predicted values
y_results_df = pd.DataFrame({
 "Actual": y_test,
 "Predicted": y_pred,
 "Actual_Class": y_test_binary,
 "Predicted_Class": y_pred_binary
},index=y_test.index)
y_results_df.tail(25)

#EVALUATION METRIC
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")

# Compute average actual return grouped by predicted class
average_actual_by_class = y_results_df.groupby("Predicted_Class")
["Actual"].mean()
print("Average Actual Return by Predicted Class:")
print(average_actual_by_class)

# Select last 30 days for plotting
last_30_days = y_test.index[-350:]
y_test_last_30 = y_test.loc[last_30_days]
y_pred_last_30 = pd.Series(y_pred, 
index=y_test.index).loc[last_30_days]

# Plot Index vs Target for both Actual and Prediction (Last 30 Days)
plt.figure(figsize=(10,6))
plt.plot(y_test_last_30.index, y_test_last_30, label="Actual", 
marker='o')
plt.plot(y_pred_last_30.index, y_pred_last_30, label="Predicted", 
marker='x')
plt.xlabel("Index")
plt.ylabel("Target 30D Return")
plt.title("Actual vs Predicted 30D Return (Last 30 Days)")
plt.legend()
plt.show()