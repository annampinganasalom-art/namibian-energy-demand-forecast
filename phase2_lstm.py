import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

# 1. Load and prepare data
df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()
df_daily = df["AEP_MW"].resample("D").mean().dropna().values.reshape(-1, 1)

# 2. Normalize data (Crucial for Neural Networks)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_daily)


# 3. Create sliding window sequences for LSTM
def create_sequences(dataset, window_size):
  X, y = [], []
  for i in range(window_size, len(dataset)):
    X.append(dataset[i - window_size : i, 0])
    y.append(dataset[i, 0])
  return np.array(X), np.array(y)


WINDOW_SIZE = 30  # Use past 30 days to predict the next day
X, y = create_sequences(scaled_data, WINDOW_SIZE)

# Reshape X to fit LSTM expectations [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Train-Test Split (85% train, 15% test)
train_size = int(len(X) * 0.85)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"LSTM Training shapes: X_train {X_train.shape}, y_train {y_train.shape}")

# 5. Build LSTM Model Architecture
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1),
])

model.compile(optimizer="adam", loss="mean_squared_error")

# 6. Fit the Model (using fewer epochs for quick training)
print("Training LSTM model...")
model.fit(
    X_train, y_train, batch_size=32, epochs=5, validation_split=0.1, verbose=1
)

# 7. Predict and Inverse Transform
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8. Evaluate Performance
rmse = root_mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print(f"LSTM Test RMSE: {rmse:.2f}")
print(f"LSTM Test MAE: {mae:.2f}")

# 9. Plot Results
plt.figure(figsize=(12, 5))
plt.plot(actuals, label="Actual Test Data", color="green")
plt.plot(
    predictions, label="LSTM Predictions", color="orange", linestyle="--"
)
plt.title("LSTM Neural Network Energy Demand Forecast")
plt.xlabel("Days")
plt.ylabel("Megawatts (MW)")
plt.legend()
plt.tight_layout()
plt.show()