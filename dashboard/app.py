import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1d4ed8;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1d4ed8; }
    .metric-label { font-size: 13px; color: #666; margin-top: 2px; }
    h1 { color: #111827 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load / generate data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
    except FileNotFoundError:
        from data.generate_data import generate_demand_data
        df = generate_demand_data()
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/sales_data.csv", index=False)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def run_arima_forecast(train_demand, n_steps, order=(5,1,2)):
    model  = ARIMA(train_demand, order=order)
    fitted = model.fit()
    return fitted.forecast(steps=n_steps)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/combo-chart--v2.png", width=60)
    st.title("⚙️ Controls")

    st.subheader("📅 Forecast Settings")
    forecast_days = st.slider("Forecast horizon (days)", 30, 120, 90, step=15)
    arima_p = st.selectbox("ARIMA — p (AR order)", [3, 5, 7], index=1)
    arima_q = st.selectbox("ARIMA — q (MA order)", [1, 2, 3], index=1)

    st.subheader("📊 Display")
    show_decomp   = st.checkbox("Show decomposition", value=True)
    show_inventory= st.checkbox("Show inventory insights", value=True)

    st.markdown("---")
    st.caption("AI-Driven Demand Forecasting System · Harshit Srivastava")


# ── Load data ──────────────────────────────────────────────────────────────
df = load_data()
split_idx  = len(df) - forecast_days
train_df   = df.iloc[:split_idx]
test_df    = df.iloc[split_idx:]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📈 AI-Driven Demand Forecasting & Inventory Optimization")
st.markdown(f"**Dataset:** {df['date'].min().date()} → {df['date'].max().date()} &nbsp;|&nbsp; "
            f"**Total records:** {len(df):,} days &nbsp;|&nbsp; "
            f"**Forecast window:** {forecast_days} days")
st.markdown("---")

# ── KPI Cards ──────────────────────────────────────────────────────────────
avg_demand  = int(df["demand"].mean())
max_demand  = int(df["demand"].max())
total_units = int(df["demand"].sum())
promo_lift  = int(df[df["promotion"]==1]["demand"].mean() - df[df["promotion"]==0]["demand"].mean())

c1, c2, c3, c4 = st.columns(4)
for col, val, label, icon in zip(
    [c1, c2, c3, c4],
    [f"{avg_demand} units", f"{max_demand} units", f"{total_units:,}", f"+{promo_lift} units"],
    ["Avg Daily Demand", "Peak Demand", "Total Units Sold", "Promo Lift"],
    ["📦", "🚀", "🏭", "🎯"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {val}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Demand Trend ───────────────────────────────────────────────────────────
st.subheader("📉 Historical Demand Trend")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["date"], df["demand"], color="#1d4ed8", linewidth=0.8, alpha=0.85, label="Daily Demand")
# 30-day rolling mean
rolling = df["demand"].rolling(30).mean()
ax.plot(df["date"], rolling, color="#ef4444", linewidth=2, label="30-day Rolling Mean")
ax.set_xlabel("Date"); ax.set_ylabel("Units")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30); ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
st.pyplot(fig)

# ── ARIMA Forecast ─────────────────────────────────────────────────────────
st.subheader(f"🔮 ARIMA Forecast — Next {forecast_days} Days")

with st.spinner("Running ARIMA model..."):
    train_series = train_df.set_index("date")["demand"]
    forecast     = run_arima_forecast(train_series, forecast_days, order=(arima_p, 1, arima_q))
    forecast     = np.clip(forecast, 0, None).astype(int)
    actual       = test_df["demand"].values

mae  = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
mape = np.mean(np.abs((actual - forecast) / actual)) * 100
acc  = round(100 - mape, 2)

# Metric row
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE",      f"{mae:.1f} units")
m2.metric("RMSE",     f"{rmse:.1f} units")
m3.metric("MAPE",     f"{mape:.1f}%")
m4.metric("Accuracy", f"{acc}%", delta=f"{acc-85:.1f}% vs 85% baseline")

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(train_df["date"].values[-90:], train_df["demand"].values[-90:],
         color="#94a3b8", linewidth=1, label="Historical (last 90d)")
ax2.plot(test_df["date"].values,  actual,   color="#111111", linewidth=1.2, label="Actual")
ax2.plot(test_df["date"].values,  forecast, color="#ef4444", linewidth=1.8,
         linestyle="--", label=f"ARIMA Forecast (Acc: {acc}%)")
ax2.fill_between(test_df["date"].values,
                 forecast * 0.92, forecast * 1.08,
                 alpha=0.12, color="#ef4444", label="±8% confidence band")
ax2.axvline(test_df["date"].values[0], color="#888", linestyle=":", linewidth=1.2, label="Forecast start")
ax2.set_xlabel("Date"); ax2.set_ylabel("Demand (Units)")
ax2.legend(loc="upper left"); ax2.grid(alpha=0.25)
plt.tight_layout()
st.pyplot(fig2)

# ── Seasonality ─────────────────────────────────────────────────────────────
st.subheader("📅 Seasonality Analysis")
col_a, col_b = st.columns(2)

with col_a:
    monthly = df.groupby("month")["demand"].mean()
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    colors = ["#ef4444" if m in [11,12] else "#3b82f6" for m in monthly.index]
    ax3.bar(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            monthly.values, color=colors, edgecolor="white", linewidth=0.5)
    ax3.set_title("Avg Demand by Month", fontweight="bold")
    ax3.set_ylabel("Units"); plt.xticks(rotation=30); ax3.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig3)

with col_b:
    dow = df.groupby("day_of_week")["demand"].mean()
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    colors2 = ["#f87171" if i >= 5 else "#60a5fa" for i in range(7)]
    ax4.bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            dow.values, color=colors2, edgecolor="white", linewidth=0.5)
    ax4.set_title("Avg Demand by Day of Week", fontweight="bold")
    ax4.set_ylabel("Units"); ax4.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig4)

# ── Decomposition ─────────────────────────────────────────────────────────
if show_decomp:
    st.subheader("🔍 Time Series Decomposition")
    ts     = df.set_index("date")["demand"]
    result = seasonal_decompose(ts, model="additive", period=30)
    fig5, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, title, comp, color in zip(
        axes,
        ["Observed", "Trend", "Seasonal", "Residual"],
        [result.observed, result.trend, result.seasonal, result.resid],
        ["#1d4ed8", "#10b981", "#f59e0b", "#ef4444"]
    ):
        ax.plot(comp, linewidth=0.9, color=color)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Value"); ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig5)

# ── Inventory Insights ─────────────────────────────────────────────────────
if show_inventory:
    st.subheader("📦 Inventory Optimization Insights")
    avg_forecast  = int(np.mean(forecast))
    safety_stock  = int(np.std(forecast) * 1.65)   # 95% service level
    reorder_point = avg_forecast * 7 + safety_stock  # 7-day lead time
    max_forecast  = int(np.max(forecast))

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Avg Forecasted Demand", f"{avg_forecast} units/day")
    i2.metric("Safety Stock (95% SL)", f"{safety_stock} units")
    i3.metric("Reorder Point",         f"{reorder_point} units")
    i4.metric("Peak Forecast",         f"{max_forecast} units/day")

    st.info("💡 **Recommendation:** Maintain safety stock of "
            f"**{safety_stock} units** and trigger reorder when inventory "
            f"falls below **{reorder_point} units** (assumes 7-day lead time).")

# ── Raw data ───────────────────────────────────────────────────────────────
with st.expander("🗂️ View Raw Dataset"):
    st.dataframe(df.tail(30), use_container_width=True)

st.markdown("---")
st.caption("Built by Harshit Srivastava · AI-Driven Demand Forecasting & Inventory Optimization System")
