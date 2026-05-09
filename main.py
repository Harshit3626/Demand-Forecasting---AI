"""
AI-Driven Demand Forecasting & Inventory Optimization System
============================================================
Run this to: generate data → EDA → train ARIMA → train LSTM
Then run the Streamlit dashboard separately.
"""

import os, sys
sys.path.append(os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("  AI-Driven Demand Forecasting System")
    print("=" * 60)

    # Step 1: Generate data
    print("\n📊 STEP 1: Generating synthetic dataset...")
    os.makedirs("data", exist_ok=True)
    from data.generate_data import generate_demand_data
    df = generate_demand_data()
    df.to_csv("data/sales_data.csv", index=False)
    print(f"   ✅ {len(df)} records saved to data/sales_data.csv")

    # Step 2: EDA
    print("\n📊 STEP 2: Running EDA...")
    from data.eda import load_data, eda_summary, preprocess
    df = load_data()
    eda_summary(df)
    train_df, test_df, X_train, X_test, y_train, y_test = preprocess(df)

    # Step 3: ARIMA
    print("\n📊 STEP 3: Training ARIMA model...")
    from models.arima_model import run_arima
    arima_preds, arima_metrics = run_arima(train_df, test_df)

    # Step 4: LSTM (optional — needs TensorFlow)
    print("\n📊 STEP 4: Training LSTM model...")
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
    print("   streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
