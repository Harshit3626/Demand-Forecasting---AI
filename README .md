# 📈 AI-Driven Demand Forecasting & Inventory Optimization System

A machine learning project that forecasts real-world retail demand using **LSTM** and **ARIMA** models on the **Rossmann Store Sales** dataset, with an interactive **Streamlit dashboard** for visualization and inventory insights.

---

## 🚀 Features
- Real-world retail sales data (Rossmann Store Sales — 1,000+ days)
- Exploratory Data Analysis (EDA) with trend, seasonality & promotion impact plots
- Time series decomposition (trend, seasonal, residual components)
- **ARIMA** model for statistical time-series forecasting
- **LSTM** (Deep Learning) model for sequence-based forecasting
- Interactive **Streamlit dashboard** with KPIs, forecasts & inventory recommendations

---

## 📦 Dataset
This project uses the **Rossmann Store Sales** dataset from Kaggle.

👉 Download from: https://www.kaggle.com/competitions/rossmann-store-sales/data

Download and place these files in the `data/` folder:
- `train.csv` — historical sales data
- `store.csv` — store metadata

> Note: Dataset files are not included in this repo due to size.

---

## 🗂️ Project Structure
```
demand_forecasting/
├── data/
│   └── eda.py               # Data loading, EDA & preprocessing
├── models/
│   ├── arima_model.py       # ARIMA forecasting
│   └── lstm_model.py        # LSTM deep learning model
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── main.py                  # Run full pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/Harshit3626/Demand-Forecasting---AI.git
cd Demand-Forecasting---AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset files to data/ folder
# Download train.csv and store.csv from Kaggle (link above)

# 4. Run full pipeline (EDA → ARIMA → LSTM)
python main.py

# 5. Launch Streamlit dashboard
python -m streamlit run dashboard/app.py
```

---

## 📊 Results

| Model | MAE | MAPE | Accuracy |
|-------|-----|------|----------|
| ARIMA | ~500 € | ~6% | ~94% |
| LSTM  | ~400 € | ~5% | ~95% |

---

## 🛠️ Tech Stack
- **Python**, **Pandas**, **NumPy** — data processing
- **Statsmodels** — ARIMA
- **TensorFlow / Keras** — LSTM
- **Scikit-learn** — metrics & scaling
- **Matplotlib / Seaborn** — visualization
- **Streamlit** — interactive dashboard

---

## 👤 Author
**Harshit Srivastava** — AI/ML Engineer  
🔗 [LinkedIn](https://www.linkedin.com/in/harshit-srivastava1008/) | [GitHub](https://github.com/Harshit3626)
