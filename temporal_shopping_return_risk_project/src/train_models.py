from pathlib import Path
import json, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "orders_demo.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TARGET = "returned"
# Only use features that should be available at the scoring moment.
EXCLUDE = {"order_id", "order_date", TARGET, "return_reason", "size_mismatch_reported",
           "customer_service_contact", "quality_issue_flag", "shipping_delay_days", "delivery_issue_flag"}

def metrics(name, truth, pred, score):
    return {"model": name, "accuracy": accuracy_score(truth, pred),
            "precision_return": precision_score(truth, pred, zero_division=0),
            "recall_return": recall_score(truth, pred, zero_division=0),
            "f1_return": f1_score(truth, pred, zero_division=0),
            "roc_auc": roc_auc_score(truth, score),
            "pr_auc_average_precision": average_precision_score(truth, score)}

def main():
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    if not DATA.exists(): raise FileNotFoundError("Run python src/generate_demo_data.py first.")
    df = pd.read_csv(DATA, parse_dates=["order_date"])
    if TARGET not in df: raise ValueError("Dataset needs confirmed binary returned label.")
    df = df[df[TARGET].isin([0,1])].copy()
    y = df[TARGET].astype(int)
    if y.nunique() != 2: raise ValueError("Need both returned and non-returned rows.")
    X = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore")
    X["order_month"] = df.order_date.dt.month
    X["order_day_of_week"] = df.order_date.dt.dayofweek
    cat = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    models = {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (balanced)": Pipeline([("prep", prep),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1500))]),
        "Random Forest (balanced)": Pipeline([("prep", prep),
            ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=4,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
    }
    rows, fitted = [], {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        score = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else pred.astype(float)
        # ROC-AUC and AP are defined here because both classes are present in the stratified split.
        rows.append(metrics(name, yte, pred, score)); fitted[name] = model
        print(f"\n{name}\n{classification_report(yte, pred, zero_division=0)}")
    pd.DataFrame(rows).to_csv(REPORTS/"model_metrics.csv", index=False)
    with open(REPORTS/"class_balance.json","w",encoding="utf-8") as f:
        json.dump({"rows":int(len(df)), "return_rate":float(y.mean()),
                   "not_returned_rate":float(1-y.mean()), "synthetic_demo":True}, f, indent=2)
    chosen = fitted["Random Forest (balanced)"]
    joblib.dump({"model":chosen, "features":list(X.columns), "synthetic_demo":True}, MODELS/"return_risk_model.joblib")
    score = chosen.predict_proba(Xte)[:,1]; pred = (score >= .5).astype(int)
    cm = confusion_matrix(yte, pred, labels=[0,1])
    plt.figure(figsize=(6,4.5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted no return","Predicted return"],
        yticklabels=["Actual no return","Actual return"])
    plt.title("Random Forest confusion matrix (threshold 0.50)")
    plt.tight_layout(); plt.savefig(REPORTS/"confusion_matrix.png", dpi=150); plt.close()
    pi = permutation_importance(chosen, Xte, yte, scoring="average_precision",
                                n_repeats=4, random_state=42, n_jobs=-1)
    pd.DataFrame({"feature":X.columns, "importance_mean":pi.importances_mean,
                  "importance_std":pi.importances_std}).sort_values("importance_mean",ascending=False).to_csv(
                      REPORTS/"feature_importance.csv", index=False)
    print("\nReports saved to", REPORTS)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Reminder: scores use synthetic demo data only.")

if __name__ == "__main__": main()
