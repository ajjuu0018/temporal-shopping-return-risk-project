from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "orders_demo.csv"

def generate_data(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    cats = np.array(["Clothing", "Footwear", "Electronics", "Home", "Beauty", "Accessories"])
    category = rng.choice(cats, n, p=[.28, .18, .16, .15, .10, .13])
    price = np.round(np.exp(rng.normal(4.1, .75, n)), 2).clip(5, 1500)
    quantity = rng.choice([1, 2, 3, 4], n, p=[.72, .19, .07, .02])
    discount = np.round(rng.beta(2, 8, n) * 70, 1)
    desc = np.round(rng.beta(7, 2, n), 2)
    size_sensitive = np.isin(category, ["Clothing", "Footwear"])
    size_chart = np.where(size_sensitive, rng.binomial(1, .68, n), 1)
    size_mismatch = np.where(size_sensitive, rng.binomial(1, .13, n), 0)
    quality = rng.binomial(1, .055, n)
    delay = np.maximum(0, rng.normal(1.6, 2.1, n)).round().astype(int)
    delivery_issue = ((delay >= 5) & (rng.random(n) < .35)).astype(int)
    service_contact = ((delivery_issue == 1) | (rng.random(n) < .035)).astype(int)
    # Synthetic relationship solely for demonstrating a complete ML workflow.
    logit = (-2.45 + .90*(category=="Clothing") + .70*(category=="Footwear")
             + 1.15*size_mismatch + .95*quality + .12*delay + .75*delivery_issue
             + .012*discount + .80*(1-desc) - .55*size_chart + .00035*price)
    p = 1/(1+np.exp(-logit))
    returned = rng.binomial(1, p)
    reasons = []
    for r, sm, q, di, d in zip(returned, size_mismatch, quality, delivery_issue, delay):
        if not r: reasons.append("Not returned")
        elif sm: reasons.append("Size/fit")
        elif q: reasons.append("Quality")
        elif di or d >= 5: reasons.append("Delivery")
        else: reasons.append(rng.choice(["Changed mind", "Not as described", "Other"]))
    return pd.DataFrame({
        "order_id": [f"ORD{i:06d}" for i in range(1, n+1)], "order_date": dates,
        "product_category": category, "product_price": price, "quantity": quantity,
        "discount_pct": discount, "description_completeness": desc,
        "size_chart_available": size_chart, "size_mismatch_reported": size_mismatch,
        "quality_issue_flag": quality, "shipping_delay_days": delay,
        "delivery_issue_flag": delivery_issue, "customer_service_contact": service_contact,
        "return_reason": reasons, "returned": returned
    })

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True, parents=True)
    df = generate_data()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} synthetic orders to {OUT}; return rate={df.returned.mean():.1%}")
    print("Synthetic demo only: do not present these results as real-world evidence.")
