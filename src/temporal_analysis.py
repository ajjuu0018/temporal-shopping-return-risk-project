from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "orders_demo.csv"
REPORTS = ROOT / "reports"

def main():
    REPORTS.mkdir(exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError("Run python src/generate_demo_data.py first.")
    df = pd.read_csv(DATA, parse_dates=["order_date"])
    daily = df.groupby(df.order_date.dt.date).agg(
        orders=("order_id","count"), returned_orders=("returned","sum"),
        return_rate=("returned","mean"), revenue_proxy=("product_price","sum")
    ).reset_index(names="date")
    daily.to_csv(REPORTS/"temporal_daily_orders.csv", index=False)
    monthly = df.groupby(df.order_date.dt.to_period("M")).agg(
        orders=("order_id","count"), returned_orders=("returned","sum"),
        return_rate=("returned","mean")).reset_index()
    monthly["month"] = monthly.order_date.astype(str)
    monthly.to_csv(REPORTS/"temporal_monthly_orders.csv", index=False)
    plt.figure(figsize=(10,4)); plt.plot(pd.to_datetime(daily.date), daily.orders)
    plt.title("Daily order volume (synthetic demo)"); plt.xlabel("Date"); plt.ylabel("Orders")
    plt.tight_layout(); plt.savefig(REPORTS/"daily_order_volume.png", dpi=150); plt.close()
    plt.figure(figsize=(10,4)); plt.plot(monthly.month, monthly.return_rate, marker="o")
    plt.title("Monthly return rate (synthetic demo)"); plt.xlabel("Month"); plt.ylabel("Return rate")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(REPORTS/"monthly_return_rate.png", dpi=150); plt.close()
    print("Temporal reports saved in", REPORTS)

if __name__ == "__main__": main()
