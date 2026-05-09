import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load_data(train_path="data/train.csv", store_path="data/store.csv", store_id=1):
    """
    Load and merge real Rossmann store sales data.
    Filters to a single store for time-series forecasting.
    """
    print(f"📂 Loading Rossmann dataset...")
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path)

    df = train.merge(store, on="Store", how="left")
    df = df[df["Open"] == 1].copy()
    df = df[df["Sales"] > 0].copy()
    df = df[df["Store"] == store_id].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df = df.rename(columns={"Date": "date", "Sales": "demand"})

    df["day_of_week"]       = df["date"].dt.dayofweek
    df["month"]             = df["date"].dt.month
    df["year"]              = df["date"].dt.year
    df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
    df["promotion"]         = df["Promo"].astype(int)

    print(f"✅ Loaded {len(df)} records for Store {store_id}")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def eda_summary(df):
    print("\n" + "=" * 50)
    print("📊 DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape         : {df.shape}")
    print(f"Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nSales Statistics:")
    print(df["demand"].describe().round(2))


def plot_demand_over_time(df):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["date"], df["demand"], color="#1d4ed8", linewidth=0.8, alpha=0.8)
    rolling = df["demand"].rolling(30).mean()
    ax.plot(df["date"], rolling, color="#ef4444", linewidth=2, label="30-day Rolling Mean")
    ax.set_title("Rossmann Store — Daily Sales Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (€)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("data/plot_demand_trend.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_demand_trend.png")


def plot_seasonality(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    monthly = df.groupby("month")["demand"].mean()
    axes[0].bar(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        monthly.values, color="#3b82f6", edgecolor="white"
    )
    axes[0].set_title("Avg Sales by Month", fontweight="bold")
    axes[0].set_xlabel("Month"); axes[0].set_ylabel("Avg Sales (€)")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

    dow = df.groupby("day_of_week")["demand"].mean()
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_labels = [day_names[i] for i in dow.index]
    colors = ["#f87171" if i >= 5 else "#60a5fa" for i in dow.index]
    axes[1].bar(dow_labels,
                dow.values, color=colors, edgecolor="white")
    axes[1].set_title("Avg Sales by Day of Week", fontweight="bold")
    axes[1].set_xlabel("Day"); axes[1].set_ylabel("Avg Sales (€)")

    plt.tight_layout()
    plt.savefig("data/plot_seasonality.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_seasonality.png")


def plot_promo_impact(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    promo_avg    = df[df["promotion"] == 1]["demand"].mean()
    no_promo_avg = df[df["promotion"] == 0]["demand"].mean()
    ax.bar(["No Promotion", "With Promotion"],
           [no_promo_avg, promo_avg],
           color=["#94a3b8", "#10b981"], edgecolor="white", width=0.4)
    ax.set_title("Promotion Impact on Sales", fontweight="bold")
    ax.set_ylabel("Avg Sales (€)")
    for i, v in enumerate([no_promo_avg, promo_avg]):
        ax.text(i, v + 50, f"€{v:,.0f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/plot_promo_impact.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_promo_impact.png")


def plot_decomposition(df):
    ts = df.set_index("date")["demand"]
    ts = ts.resample("D").mean().interpolate()
    result = seasonal_decompose(ts, model="additive", period=30)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, title, comp, color in zip(
        axes,
        ["Observed", "Trend", "Seasonal", "Residual"],
        [result.observed, result.trend, result.seasonal, result.resid],
        ["#1d4ed8", "#10b981", "#f59e0b", "#ef4444"]
    ):
        ax.plot(comp, linewidth=0.8, color=color)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Value")
    plt.suptitle("Time Series Decomposition — Rossmann Store Sales",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("data/plot_decomposition.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_decomposition.png")


def preprocess(df, test_days=90):
    feature_cols = ["day_of_week", "month", "year",
                    "is_weekend", "is_holiday_season", "promotion"]

    split_idx = len(df) - test_days
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["demand"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["demand"].values

    print(f"✅ Train: {len(train_df)} days | Test: {len(test_df)} days")
    return train_df, test_df, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    eda_summary(df)
    plot_demand_over_time(df)
    plot_seasonality(df)
    plot_promo_impact(df)
    plot_decomposition(df)
    preprocess(df)
