import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings, joblib, os
warnings.filterwarnings("ignore")

def run_arima(train_df, test_df, order=(5, 1, 2)):
    """
    Fit ARIMA model on training demand and forecast test period.
    order: (p, d, q) — AR lags, differencing, MA lags
    """
    print(f"\n🔄 Fitting ARIMA{order}...")

    train_series = train_df.set_index("date")["demand"]
    test_series  = test_df.set_index("date")["demand"]

    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    # Forecast
    forecast = fitted.forecast(steps=len(test_series))
    forecast = np.clip(forecast, 0, None).astype(int)

    # Metrics
    mae  = mean_absolute_error(test_series, forecast)
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    mape = np.mean(np.abs((test_series.values - forecast) / test_series.values)) * 100

    print(f"✅ ARIMA Results:")
    print(f"   MAE  : {mae:.2f} units")
    print(f"   RMSE : {rmse:.2f} units")
    print(f"   MAPE : {mape:.2f}%")
    print(f"   Accuracy : {100 - mape:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    fitted.save("models/arima_model.pkl")
    print("✅ Model saved: models/arima_model.pkl")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_series[-90:].index, train_series[-90:].values,
            label="Train (last 90 days)", color="#3b82f6", linewidth=1)
    ax.plot(test_series.index, test_series.values,
            label="Actual", color="#111111", linewidth=1.2)
    ax.plot(test_series.index, forecast,
            label=f"ARIMA Forecast (MAPE: {mape:.1f}%)", color="#ef4444",
            linewidth=1.5, linestyle="--")
    ax.fill_between(test_series.index,
                    forecast * 0.93, forecast * 1.07,
                    alpha=0.15, color="#ef4444", label="±7% band")
    ax.set_title("ARIMA Demand Forecast vs Actual", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Demand (Units)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/plot_arima_forecast.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_arima_forecast.png")

    return forecast, {"MAE": mae, "RMSE": rmse, "MAPE": mape,
                      "Accuracy": round(100 - mape, 2)}


if __name__ == "__main__":
    import sys; sys.path.append(".")
    from data.eda import load_data, preprocess
    df = load_data()
    train_df, test_df, *_ = preprocess(df)
    run_arima(train_df, test_df)
