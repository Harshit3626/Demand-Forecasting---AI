# 📈 AI-Driven Demand Forecasting & Inventory Optimization System

A machine learning project that forecasts product demand using **LSTM** and **ARIMA** models, with an interactive **Streamlit dashboard** for visualization and inventory insights.

---

## 🚀 Features
- Synthetic dataset generation with realistic trend, seasonality & promotions
- Exploratory Data Analysis (EDA) with decomposition plots
- **ARIMA** model for statistical time-series forecasting
- **LSTM** (Deep Learning) model for sequence-based forecasting
- Interactive **Streamlit dashboard** with KPIs, forecasts & inventory recommendations

---

## 🗂️ Project Structure
```
demand_forecasting/
├── data/
│   ├── generate_data.py     # Synthetic dataset generator
│   └── eda.py               # EDA & preprocessing
├── models/
│   ├── arima_model.py       # ARIMA forecasting
│   └── lstm_model.py        # LSTM deep learning model
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── main.py                  # Run everything
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / download the project
cd demand_forecasting

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline (data → EDA → ARIMA → LSTM)
python main.py

# 5. Launch Streamlit dashboard
streamlit run dashboard/app.py
```

---

## 📊 Results

| Model | MAE | MAPE | Accuracy |
|-------|-----|------|----------|
| ARIMA | ~18 units | ~6% | ~94% |
| LSTM  | ~15 units | ~5% | ~95% |

---

## 🛠️ Tech Stack
- **Python**, **Pandas**, **NumPy** — data processing
- **Statsmodels** — ARIMA
- **TensorFlow / Keras** — LSTM
- **Scikit-learn** — metrics & scaling
- **Matplotlib / Seaborn** — visualization
- **Streamlit** — dashboard

---

## 👤 Author
**Harshit Srivastava** — AI/ML Engineer
