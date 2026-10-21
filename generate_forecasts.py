import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

print("Loading data...")
df = pd.read_csv('AEP_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime').sort_index()
df_daily = df['AEP_MW'].resample('D').mean().dropna()

# Split data
train_size = int(len(df_daily) * 0.85)
train_data = df_daily.iloc[:train_size]
test_data = df_daily.iloc[train_size:]
test_dates = test_data.index

results_df = pd.DataFrame({'Actual': test_data.values}, index=test_dates)

# 1. ARIMA Forecast
print("Fitting ARIMA...")
arima_model = ARIMA(train_data, order=(1,1,1)).fit()
results_df['ARIMA'] = arima_model.forecast(steps=len(test_data))

# 2. SARIMA Forecast
print("Fitting SARIMA...")
sarima_model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,7), 
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
results_df['SARIMA'] = sarima_model.forecast(steps=len(test_data))

# 3. Prophet Forecast
print("Fitting Prophet...")
prophet_train = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
m.fit(prophet_train)
future = m.make_future_dataframe(periods=len(test_data), freq='D')
forecast = m.predict(future)
results_df['Prophet'] = forecast.iloc[-len(test_data):]['yhat'].values

# 4. LSTM Forecast
print("Fitting LSTM...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_daily.values.reshape(-1, 1))

def create_sequences(dataset, window_size):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])
    return np.array(X)

WINDOW_SIZE = 30
X_all = create_sequences(scaled_data, WINDOW_SIZE)
X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1))
X_test_lstm = X_all[train_size - WINDOW_SIZE:]

model = Sequential([
    Input(shape=(WINDOW_SIZE, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_all[:train_size - WINDOW_SIZE], scaled_data[WINDOW_SIZE:train_size], batch_size=32, epochs=3, verbose=0)

lstm_preds_scaled = model.predict(X_test_lstm, verbose=0)
results_df['LSTM'] = scaler.inverse_transform(lstm_preds_scaled).flatten()

# Save outputs
results_df.to_csv('model_comparison_results.csv')
print("Forecasts successfully generated and saved to 'model_comparison_results.csv'!")