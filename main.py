"""
AI-Driven Demand Forecasting & Inventory Optimization System
============================================================
Run this to: load real data → EDA → train ARIMA → train LSTM
Then run the Streamlit dashboard separately.
"""

import os, sys
sys.path.append(os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("  AI-Driven Demand Forecasting System (Rossmann Dataset)")
    print("=" * 60)

    # Step 1: Load real data
    print("\n📊 STEP 1: Loading Rossmann real-world dataset...")
    from data.eda import load_data, eda_summary, preprocess
    df = load_data(train_path="data/train.csv", store_path="data/store.csv", store_id=1)
    eda_summary(df)
    train_df, test_df, X_train, X_test, y_train, y_test = preprocess(df)

    # Step 2: ARIMA
    print("\n📊 STEP 2: Training ARIMA model...")
    from models.arima_model import run_arima
    arima_preds, arima_metrics = run_arima(train_df, test_df)

    # Step 3: LSTM (optional — needs TensorFlow)
    print("\n📊 STEP 3: Training LSTM model...")
    try:
        import tensorflow
        from models.lstm_model import run_lstm
        lstm_preds, lstm_metrics = run_lstm(train_df, test_df)

        print("\n" + "=" * 60)
        print("  MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<10} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Accuracy':>10}")
        print("-" * 50)
        print(f"{'ARIMA':<10} {arima_metrics['MAE']:>8.2f} {arima_metrics['RMSE']:>8.2f} "
              f"{arima_metrics['MAPE']:>7.2f}% {arima_metrics['Accuracy']:>9.2f}%")
        if lstm_metrics:
            print(f"{'LSTM':<10} {lstm_metrics['MAE']:>8.2f} {lstm_metrics['RMSE']:>8.2f} "
                  f"{lstm_metrics['MAPE']:>7.2f}% {lstm_metrics['Accuracy']:>9.2f}%")
    except ImportError:
        print("   ⚠️  TensorFlow not found. Skipping LSTM.")
        print("   Install with: pip install tensorflow")

    print("\n" + "=" * 60)
    print("✅ All steps complete!")
    print("\n🚀 Launch the dashboard:")
    print("   python -m streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
