# ⚡ Namibian demand forecasting and time series modelling

A comprehensive time-series machine learning project comparing classical statistical models, modern additive frameworks, and deep learning architectures for hourly and daily energy consumption forecasting.

## 📌 Project Overview
Accurate energy demand forecasting is critical for grid stability, load balancing, and minimizing power generation waste. This project implements and evaluates four distinct forecasting tiers using 14 years of historical hourly power consumption data:
1. **ARIMA** (Classical Statistical Baseline)
2. **SARIMA** (Seasonal Autoregressive Integrated Moving Average)
3. **Prophet** (Meta's Modern Additive Model)
4. **LSTM** (Long Short-Term Memory Deep Learning Neural Network)

---

## 📊 Model Performance Comparison
| Model Architecture | Test RMSE (MW) | Test MAE (MW) | Key Advantage |
| :--- | :--- | :--- | :--- |
| **ARIMA** | 2,117.00 | 1,788.04 | Simple statistical baseline |
| **SARIMA** | 1,856.95 | 1,558.49 | Captures recurring weekly/seasonal cycles |
| **Prophet** | 1,652.30 | 1,269.07 | Robust multi-seasonality handling |
| **LSTM** | **1,174.43** | **926.38** | Superior non-linear sequential learning |

---

## 🚀 Live Interactive Dashboard
The project features an interactive **Streamlit Dashboard** allowing users to:
* Filter custom date range horizons.
* Toggle between ARIMA, SARIMA, Prophet, and LSTM forecasts on the fly.
* Inspect unified hover tooltips and real-time evaluation metrics.

### Local Installation & Quickstart
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME