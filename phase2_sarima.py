import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. Load and prepare data (resampled to daily mean for stability)
df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# Resample to daily mean load
df_daily = df["AEP_MW"].resample("D").mean().dropna()

# 2. Train-Test Split (85% train, 15% test)
train_size = int(len(df_daily) * 0.85)
train_data = df_daily.iloc[:train_size]
test_data = df_daily.iloc[train_size:]

print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

# 3. Fit a SARIMA Model
# Order: (p, d, q) -> non-seasonal parameters
# Seasonal Order: (P, D, Q, s) -> seasonal parameters with a 7-day weekly period
print("Fitting SARIMA model (this may take a minute)...")
model = SARIMAX(
    train_data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
model_fitted = model.fit(disp=False)

# Print summary
print(model_fitted.summary())

# 4. Forecast future values
forecast_steps = len(test_data)
predictions = model_fitted.forecast(steps=forecast_steps)
predictions = pd.Series(predictions, index=test_data.index)

# 5. Evaluate Performance
rmse = root_mean_squared_error(test_data, predictions)
mae = mean_absolute_error(test_data, predictions)

print(f"SARIMA Test RMSE: {rmse:.2f}")
print(f"SARIMA Test MAE: {mae:.2f}")

# 6. Plot Results
plt.figure(figsize=(12, 5))
plt.plot(
    train_data.index[-100:], train_data.iloc[-100:], label="Recent Training Data"
)
plt.plot(test_data.index, test_data, label="Actual Test Data", color="green")
plt.plot(
    test_data.index,
    predictions,
    label="SARIMA Predictions",
    color="purple",
    linestyle="--",
)
plt.title("SARIMA Forecast vs Actual Daily Energy Demand")
plt.xlabel("Date")
plt.ylabel("Megawatts (MW)")
plt.legend()
plt.tight_layout()
plt.show()