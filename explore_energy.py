import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load and sort data
df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# 2. Plot a single month (e.g., January 2005) to see daily cycles (morning/evening peaks)
plt.figure(figsize=(12, 4))
df["AEP_MW"].loc["2005-01-01":"2005-01-31"].plot(title="January Daily Consumption Cycles")
plt.ylabel("Megawatts (MW)")
plt.xlabel("Datetime")
plt.tight_layout()
plt.show()

# 3. Resample to Daily Mean to examine macro trends and yearly seasonality
df_daily = df["AEP_MW"].resample("D").mean()

plt.figure(figsize=(12, 4))
df_daily.plot(title="Daily Resampled AEP Energy Trend (2004-2018)")
plt.ylabel("Average Daily Load (MW)")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# 4. Seasonal Decomposition (using a 365-day yearly window)
decomposition = seasonal_decompose(df_daily, model="additive", period=365)
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

decomposition.observed.plot(ax=axes[0], title="Observed Daily Load")
decomposition.trend.plot(
    ax=axes[1], title="Trend Component (Long-term progression)"
)
decomposition.seasonal.plot(
    ax=axes[2], title="Seasonal Component (Yearly recurring pattern)"
)
decomposition.resid.plot(
    ax=axes[3], title="Residuals / Anomalies (Noise & Shocks)"
)

plt.tight_layout()
plt.show()