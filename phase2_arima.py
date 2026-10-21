import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 1. Load and prepare data (resampled to daily mean for baseline stability)
df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# Resample to daily mean load to capture macro-trends and weekly/seasonal patterns cleanly
df_daily = df["AEP_MW"].resample("D").mean().dropna()

# 2. Train-Test Split (e.g., Use 85% for training, 15% for testing/evaluation)
train_size = int(len(df_daily) * 0.85)
train_data = df_daily.iloc[:train_size]
test_data = df_daily.iloc[train_size:]

print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

# 3. Fit an ARIMA Model 
# (p, d, q) parameters: autoregressive order, degree of differencing, moving average order
# We use a standard baseline configuration like ARIMA(1, 1, 1)
print("Fitting ARIMA model...")
model = ARIMA(train_data, order=(1, 1, 1))
model_fitted = model.fit()

# Print model summary (coefficients, AIC, BIC)
print(model_fitted.summary())

# 4. Forecast future values matching the test set length
forecast_steps = len(test_data)
predictions = model_fitted.forecast(steps=forecast_steps)
predictions = pd.Series(predictions, index=test_data.index)

# 5. Evaluate Performance (RMSE and MAE)
rmse = root_mean_squared_error(test_data, predictions)
mae = mean_absolute_error(test_data, predictions)

print(f"ARIMA Test RMSE: {rmse:.2f}")
print(f"ARIMA Test MAE: {mae:.2f}")

# 6. Plot Results
plt.figure(figsize=(12, 5))
plt.plot(train_data.index[-100:], train_data.iloc[-100:], label="Recent Training Data")
plt.plot(test_data.index, test_data, label="Actual Test Data", color="green")
plt.plot(test_data.index, predictions, label="ARIMA Predictions", color="red", linestyle="--")
plt.title("ARIMA Baseline Forecast vs Actual Daily Energy Demand")
plt.xlabel("Date")
plt.ylabel("Megawatts (MW)")
plt.legend()
plt.tight_layout()
plt.show()