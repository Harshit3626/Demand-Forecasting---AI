import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os, joblib
import warnings
warnings.filterwarnings("ignore")

# ── Sequence builder ───────────────────────────────────────────────────────
def create_sequences(data, look_back=30):
    """Convert time series array into (X, y) sequences for LSTM."""
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def run_lstm(train_df, test_df, look_back=30, epochs=30, batch_size=32):
    """
    Build, train, and evaluate an LSTM model for demand forecasting.
    Uses TensorFlow/Keras.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        print(f"✅ TensorFlow {tf.__version__} detected")
    except ImportError:
        print("❌ TensorFlow not installed. Run: pip install tensorflow")
        return None, None

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_vals = train_df["demand"].values.reshape(-1, 1)
    test_vals  = test_df["demand"].values.reshape(-1, 1)

    train_scaled = scaler.fit_transform(train_vals)
    # For test, use tail of train + test to preserve look_back window
    full_scaled = scaler.transform(
        np.concatenate([train_vals[-look_back:], test_vals])
    )

    X_train, y_train = create_sequences(train_scaled, look_back)
    X_test,  y_test  = create_sequences(full_scaled,  look_back)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],   X_test.shape[1],  1)

    # ── Build model ────────────────────────────────────────────────────────
    print("\n🔄 Building LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n🔄 Training for up to {epochs} epochs...")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )

    # ── Predict ────────────────────────────────────────────────────────────
    pred_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(pred_scaled).flatten()
    predictions = np.clip(predictions, 0, None).astype(int)
    actual      = test_df["demand"].values[:len(predictions)]

    # ── Metrics ────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    print(f"\n✅ LSTM Results:")
    print(f"   MAE      : {mae:.2f} units")
    print(f"   RMSE     : {rmse:.2f} units")
    print(f"   MAPE     : {mape:.2f}%")
    print(f"   Accuracy : {100 - mape:.2f}%")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.h5")
    joblib.dump(scaler, "models/lstm_scaler.pkl")
    print("✅ Model saved: models/lstm_model.h5")

    # ── Plots ──────────────────────────────────────────────────────────────
    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train Loss", color="#3b82f6")
    ax.plot(history.history["val_loss"], label="Val Loss",   color="#ef4444")
    ax.set_title("LSTM Training Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/plot_lstm_loss.png", dpi=150)
    plt.show()

    # Forecast vs Actual
    dates = test_df["date"].values[:len(predictions)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, actual,      label="Actual",  color="#111111", linewidth=1.2)
    ax.plot(dates, predictions, label=f"LSTM Forecast (MAPE: {mape:.1f}%)",
            color="#10b981", linewidth=1.5, linestyle="--")
    ax.fill_between(dates,
                    predictions * 0.93, predictions * 1.07,
                    alpha=0.15, color="#10b981", label="±7% band")
    ax.set_title("LSTM Demand Forecast vs Actual", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Demand (Units)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/plot_lstm_forecast.png", dpi=150)
    plt.show()
    print("✅ Saved: data/plot_lstm_forecast.png")

    return predictions, {"MAE": mae, "RMSE": rmse, "MAPE": mape,
                         "Accuracy": round(100 - mape, 2)}


if __name__ == "__main__":
    import sys; sys.path.append(".")
    from data.eda import load_data, preprocess
    df = load_data()
    train_df, test_df, *_ = preprocess(df)
    run_lstm(train_df, test_df)
