import pandas as pd

def build_recommendations(df):
    rows = []
    if "returned" not in df: return pd.DataFrame(columns=["area","observed_signal","recommended_action","validation"])
    overall = df.returned.mean()
    if "product_category" in df:
        grouped = df.groupby("product_category").returned.agg(["count","mean"]).reset_index()
        for _, r in grouped.iterrows():
            if r["count"] >= 30 and r["mean"] > overall * 1.15:
                rows.append({"area":"Category investigation",
                    "observed_signal":f"{r.product_category}: return rate {r['mean']:.1%} vs overall {overall:.1%} ({int(r['count'])} orders)",
                    "recommended_action":"Audit return reasons, product content, fit/quality and fulfillment for this category.",
                    "validation":"Confirm with real return-reason data; measure results in a controlled or well-designed before/after test."})
    if "description_completeness" in df:
        low = df[df.description_completeness < .6]
        if len(low) >= 30:
            rows.append({"area":"Product descriptions",
                "observed_signal":f"{len(low)} orders have description-completeness score below 0.60.",
                "recommended_action":"Improve specifications, dimensions, materials, compatibility, photos and expectation-setting.",
                "validation":"A/B test improved listings; track returns and conversion."})
    if "size_chart_available" in df and "product_category" in df:
        fit = df[df.product_category.isin(["Clothing","Footwear"])]
        if len(fit) and fit.size_chart_available.mean() < .85:
            rows.append({"area":"Sizing information",
                "observed_signal":f"Size-chart coverage for clothing/footwear is {fit.size_chart_available.mean():.1%}.",
                "recommended_action":"Add accurate size charts, fit notes and model measurements; explain between-size choices.",
                "validation":"Track size/fit returns by category and size before and after rollout."})
    if "quality_issue_flag" in df and df.quality_issue_flag.mean() > .03:
        rows.append({"area":"Quality control","observed_signal":f"Quality issue flag in {df.quality_issue_flag.mean():.1%} of rows.",
            "recommended_action":"Inspect suppliers and batches, define incoming checks and investigate recurring defects.",
            "validation":"Measure defect and quality-related return rates by supplier/batch."})
    if "shipping_delay_days" in df and df.shipping_delay_days.mean() > 2:
        rows.append({"area":"Logistics","observed_signal":f"Mean recorded shipping delay is {df.shipping_delay_days.mean():.1f} days.",
            "recommended_action":"Review late lanes/carriers, delivery estimates and proactive delay notifications.",
            "validation":"Compare on-time delivery, contacts and returns across lanes."})
    if "customer_service_contact" in df and df.customer_service_contact.mean() > .03:
        rows.append({"area":"Customer experience","observed_signal":f"Customer-service contact flag in {df.customer_service_contact.mean():.1%} of rows.",
            "recommended_action":"Review contact reasons, improve tracking and clarify exchange/return policies.",
            "validation":"Track repeat contacts, satisfaction and return rates; contact may be a consequence rather than a cause."})
    return pd.DataFrame(rows)
