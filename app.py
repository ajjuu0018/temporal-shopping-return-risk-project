from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from generate_demo_data import generate_data
from recommendations import build_recommendations

st.set_page_config(page_title="Shopping Return Risk", page_icon="📦", layout="wide")
st.title("Temporal Shopping Analytics & Return-Risk Prediction")
st.caption("Student project | Synthetic data is for demonstration only—not real-world evidence.")

data_path = ROOT / "data" / "orders_demo.csv"
if not data_path.exists():
    data_path.parent.mkdir(exist_ok=True, parents=True)
    generate_data().to_csv(data_path, index=False)
df = pd.read_csv(data_path, parse_dates=["order_date"])

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Temporal patterns", "Risk scoring", "Industry actions"])
with tab1:
    c1,c2,c3 = st.columns(3)
    c1.metric("Orders", f"{len(df):,}")
    c2.metric("Return rate", f"{df.returned.mean():.1%}")
    c3.metric("Non-return rate", f"{1-df.returned.mean():.1%}")
    st.warning("The included data is synthetic. Do not cite its metrics as evidence of actual customer behaviour.")
    st.subheader("Why accuracy alone can mislead")
    st.write("If 90% of orders are not returned, a classifier that predicts no return for every order gets 90% accuracy but detects zero returns. Review return-class recall, precision, F1, PR-AUC and the confusion matrix.")
    st.dataframe(df.head(10), use_container_width=True)
with tab2:
    daily = df.groupby(df.order_date.dt.date).size().rename("orders").reset_index()
    daily.columns = ["date","orders"]
    st.subheader("Daily order volume")
    st.line_chart(daily.set_index("date"))
    monthly = df.assign(month=df.order_date.dt.to_period("M").astype(str)).groupby("month").agg(
        orders=("order_id","count"), return_rate=("returned","mean"))
    st.subheader("Monthly return rate")
    st.line_chart(monthly[["return_rate"]])
    st.dataframe(monthly, use_container_width=True)
with tab3:
    model_path = ROOT / "models" / "return_risk_model.joblib"
    if not model_path.exists():
        st.info("Train the model first: python src/train_models.py")
    else:
        pack = joblib.load(model_path)
        model, features = pack["model"], pack["features"]
        st.write("Enter values for an illustrative order. This demo model was trained on synthetic data.")
        with st.form("risk_form"):
            category = st.selectbox("Product category", ["Clothing","Footwear","Electronics","Home","Beauty","Accessories"])
            price = st.number_input("Product price", min_value=1.0, value=60.0)
            quantity = st.number_input("Quantity", min_value=1, max_value=20, value=1)
            discount = st.slider("Discount (%)", 0.0, 70.0, 10.0)
            description = st.slider("Description completeness (0–1)", 0.0, 1.0, 0.8)
            size_chart = st.selectbox("Size chart available?", [1,0], format_func=lambda x: "Yes" if x else "No")
            month = st.slider("Order month", 1, 12, 6)
            dow = st.slider("Day of week (Monday=0)", 0, 6, 2)
            submitted = st.form_submit_button("Estimate return risk")
        if submitted:
            row = {"product_category":category, "product_price":price, "quantity":quantity,
                   "discount_pct":discount, "description_completeness":description,
                   "size_chart_available":size_chart, "order_month":month, "order_day_of_week":dow}
            inp = pd.DataFrame([{f:row.get(f, 0) for f in features}])
            prob = float(model.predict_proba(inp)[0,1])
            st.metric("Estimated return risk", f"{prob:.1%}")
            st.write("This is a demonstration score, not a validated probability for real customers.")
            st.progress(min(max(prob,0.0),1.0))
            st.caption("Do not automatically reject orders or penalize customers based on this score.")
with tab4:
    st.subheader("Intervention hypotheses")
    recs = build_recommendations(df)
    if recs.empty:
        st.info("No recommendations met the current diagnostic rules. Review the data and thresholds.")
    else:
        st.dataframe(recs, use_container_width=True)
    st.write("Recommendations should be validated with real data and experiments. Observed associations do not prove causation.")
