# Temporal Shopping Pattern & Interpretable Return-Risk Prediction

This project combines temporal shopping analysis, return-risk classification, evaluation for imbalanced labels, interpretable model explanations, and operational recommendations.

## Important data note
The included generator creates **synthetic demonstration data** so the project runs end-to-end. It is not real-world evidence and its scores must not be presented as real-world performance. For valid conclusions, replace it with an order-level dataset containing a documented, confirmed return label. Do not automatically treat cancellations or negative quantities as confirmed returns.

## Quick start (Python 3.10+)
#if you want to just open website then click on below link
https://temporal.streamlit.app/
and if you want to try in vscode terminl then use beloww code:-
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_demo_data.py
python src/train_models.py
python src/temporal_analysis.py
streamlit run app.py
```

## Features
- Temporal daily/monthly order volume and return-rate summaries.
- Majority-class baseline, class-weighted Logistic Regression, and class-weighted Random Forest.
- Accuracy, precision, recall, F1, ROC-AUC, average precision (PR-AUC), confusion matrix.
- Permutation feature importance to explain which available features influence predictions globally.
- Streamlit dashboard with risk scoring and intervention hypotheses.

## Dataset fields
`order_id`, `order_date`, `product_category`, `product_price`, `quantity`, `discount_pct`, `description_completeness`, `size_chart_available`, `size_mismatch_reported`, `quality_issue_flag`, `shipping_delay_days`, `delivery_issue_flag`, `customer_service_contact`, `return_reason`, `returned`.

`returned` must be 1 for a confirmed return and 0 for a confirmed non-return. Resolve unknown/pending orders before training.

## Leakage warning
A prediction must only use fields available at the time it is made. The model intentionally excludes `return_reason`, `size_mismatch_reported`, `customer_service_contact`, `quality_issue_flag`, `shipping_delay_days`, and `delivery_issue_flag`, because these can be observed after purchase or delivery. Review the timing of every field when adapting to a real dataset. For operational future prediction, prefer a time-based holdout.

## Why accuracy alone can mislead
If 90% of orders are not returned, predicting “not returned” for every order gives 90% accuracy but detects no returns. Report precision, recall, F1, PR-AUC and the confusion matrix as well. The threshold should reflect business costs and customer experience; never automatically penalize customers based only on a risk score.

## Industry recommendations
The dashboard creates hypotheses around product descriptions, sizing, quality control, logistics, and customer experience based on observed signals. Associations do not prove causation. Validate changes with controlled experiments or well-designed before/after analysis and track return reasons, customer satisfaction, conversion and operational costs.

## Limitations
- Demo data is synthetic.
- Feature importance is associative, not causal.
- Results and interventions must be validated using real data.
