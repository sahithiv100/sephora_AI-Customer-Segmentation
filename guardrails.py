"""
Sephora AI Customer Segmentation Prototype
Module: guardrails.py
Description: Responsible AI audits, differential privacy / k-anonymity verification,
demographic fairness checks, and marketing guardrails.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

def run_governance_audit(df: pd.DataFrame, min_k_anonymity: int = 30) -> Dict[str, Any]:
    """
    Performs algorithmic governance and privacy compliance checks on customer segments.
    
    Audits:
    1. K-Anonymity: Ensures all segment cohorts have at least `min_k_anonymity` records.
    2. Demographic Disparity: Checks representation of age groups across clusters.
    3. Consent Compliance: Audits opt-in rates to prevent unsolicited marketing communications.
    4. Cannibalization Risk: Checks if high-spend segments are exposed to margin-diluting promotions.
    """
    segment_counts = df["segment_name"].value_counts().to_dict()
    total_customers = len(df)
    
    # 1. K-Anonymity & Minimum Cell Size
    privacy_flags = []
    k_anon_passed = True
    for seg, count in segment_counts.items():
        if count < min_k_anonymity:
            k_anon_passed = False
            privacy_flags.append(
                f"Cohort '{seg}' has only {count} records (threshold: {min_k_anonymity}). "
                "Aggregated reporting is recommended to prevent individual client re-identification."
            )
            
    # Check sub-cells (Segment x Age Group)
    subcell_cross = pd.crosstab(df["segment_name"], df["age_group"])
    small_cells = (subcell_cross < 5).sum().sum()
    if small_cells > 0:
        privacy_flags.append(
            f"Detected {small_cells} demographic sub-cells with <5 members. "
            "Differential noise or suppression should be applied if exporting granular demographic drilldowns."
        )
        
    # 2. Fairness & Demographic Balance
    overall_age_dist = df["age_group"].value_counts(normalize=True).to_dict()
    fairness_disparities = []
    for seg in df["segment_name"].unique():
        seg_sub = df[df["segment_name"] == seg]
        seg_age_dist = seg_sub["age_group"].value_counts(normalize=True).to_dict()
        for age_grp, pop_ratio in overall_age_dist.items():
            seg_ratio = seg_age_dist.get(age_grp, 0.0)
            if pop_ratio > 0:
                disparity = seg_ratio / pop_ratio
                if disparity < 0.25:
                    fairness_disparities.append(
                        f"Age bracket {age_grp} is under-represented in '{seg}' (Disparity ratio: {disparity:.2f}x)."
                    )
                    
    # 3. Consent Compliance Summary
    consent_metrics = {
        "email_opt_in_pct": round(float(df["consent_email"].mean()) * 100, 1),
        "sms_opt_in_pct": round(float(df["consent_sms"].mean()) * 100, 1),
        "app_push_opt_in_pct": round(float(df["consent_app_push"].mean()) * 100, 1),
        "direct_mail_opt_in_pct": round(float(df["consent_direct_mail"].mean()) * 100, 1)
    }
    
    # 4. Cannibalization Risk Check
    # High-spend Rouge customers receiving heavy discounts
    rouge_sub = df[df["loyalty_tier"] == "Rouge"]
    discounted_rouge = (rouge_sub["discount_purchase_ratio"] > 0.40).mean() * 100.0 if len(rouge_sub) > 0 else 0.0
    
    return {
        "k_anonymity_passed": k_anon_passed,
        "min_k_anonymity_threshold": min_k_anonymity,
        "smallest_cohort_size": min(segment_counts.values()) if segment_counts else 0,
        "privacy_flags": privacy_flags,
        "fairness_disparities": fairness_disparities[:5],  # top 5
        "consent_metrics": consent_metrics,
        "rouge_discount_cannibalization_pct": round(discounted_rouge, 1),
        "human_review_required": True,
        "governance_status": "PASS WITH GUARDRAILS" if k_anon_passed else "ATTENTION NEEDED"
    }
