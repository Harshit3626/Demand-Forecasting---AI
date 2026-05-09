import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_demand_data(start_date="2021-01-01", days=1095, seed=42):
    """
    Generate realistic synthetic sales/demand data with:
    - Trend (gradual growth)
    - Seasonality (weekly + yearly)
    - Promotions/events
    - Noise
    """
    np.random.seed(seed)
    dates = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i) for i in range(days)]

    # Base trend (slowly growing business)
    trend = np.linspace(200, 350, days)

    # Weekly seasonality (weekends sell more)
    weekly = np.array([30 if datetime.weekday(d) >= 5 else 0 for d in dates])

    # Yearly seasonality (Nov-Dec peak for holidays)
    yearly = np.array([
        80 if d.month in [11, 12] else
        40 if d.month in [6, 7] else
        -20 if d.month in [1, 2] else 0
        for d in dates
    ])

    # Random promotions (10% of days have a promo boost)
    promo = np.where(np.random.rand(days) < 0.10, np.random.randint(50, 150, days), 0)

    # Gaussian noise
    noise = np.random.normal(0, 20, days)

    demand = trend + weekly + yearly + promo + noise
    demand = np.clip(demand, 50, None).astype(int)  # no negative demand

    df = pd.DataFrame({
        "date": dates,
        "demand": demand,
        "day_of_week": [d.weekday() for d in dates],
        "month": [d.month for d in dates],
        "year": [d.year for d in dates],
        "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
        "is_holiday_season": [1 if d.month in [11, 12] else 0 for d in dates],
        "promotion": (promo > 0).astype(int),
    })

    return df


if __name__ == "__main__":
    df = generate_demand_data()
    df.to_csv("data/sales_data.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} rows")
    print(df.head())
    print(f"\nDemand stats:\n{df['demand'].describe()}")
