import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

def load_data(path="data/sales_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def eda_summary(df):
    print("=" * 50)
    print("📊 DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape         : {df.shape}")
    print(f"Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nDemand Statistics:")
    print(df["demand"].describe().round(2))


def plot_demand_over_time(df):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["date"], df["demand"], color="#1d4ed8", linewidth=0.8, alpha=0.8)
    ax.set_title("Daily Demand Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (Units)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("data/plot_demand_trend.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_demand_trend.png")


def plot_seasonality(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Monthly average
    monthly = df.groupby("month")["demand"].mean()
    axes[0].bar(monthly.index, monthly.values, color="#3b82f6", edgecolor="white")
    axes[0].set_title("Average Demand by Month", fontweight="bold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Avg Demand")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=30)

    # Day of week
    dow = df.groupby("day_of_week")["demand"].mean()
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    colors = ["#f87171" if i >= 5 else "#60a5fa" for i in range(7)]
    axes[1].bar(days, dow.values, color=colors, edgecolor="white")
    axes[1].set_title("Average Demand by Day of Week", fontweight="bold")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Avg Demand")

    plt.tight_layout()
    plt.savefig("data/plot_seasonality.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_seasonality.png")


def plot_decomposition(df):
    ts = df.set_index("date")["demand"]
    result = seasonal_decompose(ts, model="additive", period=30)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    titles = ["Observed", "Trend", "Seasonal", "Residual"]
    components = [result.observed, result.trend, result.seasonal, result.resid]

    for ax, title, comp in zip(axes, titles, components):
        ax.plot(comp, linewidth=0.8, color="#1d4ed8")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Value")

    plt.suptitle("Time Series Decomposition", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("data/plot_decomposition.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_decomposition.png")


def preprocess(df, test_days=90):
    """
    Split into train/test and create feature matrix.
    Returns:
        train_df, test_df  — full rows
        X_train, X_test, y_train, y_test — for ML models
    """
    feature_cols = ["day_of_week", "month", "year",
                    "is_weekend", "is_holiday_season", "promotion"]

    split_idx = len(df) - test_days
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

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
    plot_decomposition(df)
    train_df, test_df, X_train, X_test, y_train, y_test = preprocess(df)
