import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Namibian Energy Demand AI Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Namibian Grid Energy Demand Forecasting Engine")
st.markdown(
    "Interactive dashboard comparing **ARIMA, SARIMA, Prophet, and LSTM**"
    " machine learning architectures on historical power load data."
)


# 2. Load Cached Results
@st.cache_data
def load_results():
  return pd.read_csv(
      "model_comparison_results.csv", index_col="Datetime", parse_dates=True
  )


try:
  results = load_results()
except FileNotFoundError:
  st.error(
      "Please run `generate_forecasts.py` first to generate model result files!"
  )
  st.stop()

# 3. Sidebar Controls
st.sidebar.header("Dashboard Controls")

# Model Multi-selector
selected_models = st.sidebar.multiselect(
    "Choose Models to Compare",
    ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    default=["LSTM", "Prophet", "SARIMA"],
)

# Date Filter Slider
min_date = results.index.min().date()
max_date = results.index.max().date()

start_date, end_date = st.sidebar.date_input(
    "Select Date Horizon View",
    value=[min_date, min_date + pd.Timedelta(days=60)],  # Default 60 days view
    min_value=min_date,
    max_value=max_date,
)

# Filter dataset to selected range
filtered_results = results.loc[
    (results.index.date >= start_date) & (results.index.date <= end_date)
]

# 4. Main Plotly Interactive Visual
st.subheader("Interactive Model Demand Forecast Comparison")

fig = go.Figure()

# Add Ground Truth (Actual Load)
fig.add_trace(
    go.Scatter(
        x=filtered_results.index,
        y=filtered_results["Actual"],
        mode="lines",
        name="Actual Energy Load",
        line=dict(color="black", width=2.5),
    )
)

# Color Mapping for Models
model_colors = {
    "ARIMA": "#EF553B",  # Red
    "SARIMA": "#AB63FA",  # Purple
    "Prophet": "#FFA15A",  # Orange
    "LSTM": "#00CC96",  # Green
}

# Add Selected Model Forecasts
for model in selected_models:
  if model in filtered_results.columns:
    fig.add_trace(
        go.Scatter(
            x=filtered_results.index,
            y=filtered_results[model],
            mode="lines",
            name=f"{model} Forecast",
            line=dict(color=model_colors.get(model, "blue"), dash="dash"),
        )
    )

# Figure Styling
fig.update_layout(
    title=dict(text="Hover over lines to inspect exact MW values", font=dict(size=14)),
    xaxis_title="Date",
    yaxis_title="Megawatts (MW)",
    hovermode="x unified",
    template="plotly_white",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, width="stretch")

# 5. Metrics Evaluation Table
st.subheader("📊 Evaluation Metrics Summary")


def calc_metrics(actual, pred):
  rmse = ((actual - pred) ** 2).mean() ** 0.5
  mae = (abs(actual - pred)).mean()
  return rmse, mae


metrics_data = []
for model in ["ARIMA", "SARIMA", "Prophet", "LSTM"]:
  if model in results.columns:
    rmse, mae = calc_metrics(results["Actual"], results[model])
    metrics_data.append(
        {"Model": model, "RMSE (MW)": round(rmse, 2), "MAE (MW)": round(mae, 2)}
    )

metrics_df = pd.DataFrame(metrics_data).set_index("Model")

col1, col2 = st.columns([2, 1])

with col1:
  st.table(metrics_df)

with col2:
  st.markdown("### 💡 Model Rankings")
  st.markdown("1. **LSTM** (Best non-linear accuracy)")
  st.markdown("2. **Prophet** (Strong seasonal handling)")
  st.markdown("3. **SARIMA** (Captures weekly cycles)")
  st.markdown("4. **ARIMA** (Baseline comparison)")