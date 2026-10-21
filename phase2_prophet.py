import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 1. Load and prepare data for Prophet
# Prophet requires specific column names: 'ds' for timestamp and 'y' for the target value
df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# Resample to daily mean
df_daily = df["AEP_MW"].resample("D").mean().reset_index()
df_daily.columns = ["ds", "y"]  # Rename for Prophet requirements

# 2. Train-Test Split (85% train, 15% test)
train_size = int(len(df_daily) * 0.85)
train_data = df_daily.iloc[:train_size]
test_data = df_daily.iloc[train_size:]

print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

# 3. Initialize and fit Prophet Model
print("Fitting Prophet model...")
model = Prophet(
    daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True
)
model.fit(train_data)

# 4. Make future predictions matching the test set
future = model.make_future_dataframe(periods=len(test_data), freq="D")
forecast = model.predict(future)

# Extract predictions corresponding to the test set dates
predictions = forecast.iloc[-len(test_data) :]["yhat"].values
actuals = test_data["y"].values

# 5. Evaluate Performance
rmse = root_mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print(f"Prophet Test RMSE: {rmse:.2f}")
print(f"Prophet Test MAE: {mae:.2f}")

# 6. Plot Results
fig = model.plot(forecast)
plt.title("Prophet Energy Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Megawatts (MW)")
plt.show()