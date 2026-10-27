"""
Sephora AI Customer Segmentation Prototype
Module: features.py
Description: Feature engineering logic computing RFM indices, category diversity
(Shannon entropy), discount sensitivity, omnichannel engagement scores, and churn risk.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

FEATURE_COLUMNS_FOR_CLUSTERING = [
    "recency_days",
    "frequency_12m",
    "monetary_total_12m",
    "avg_order_value",
    "category_entropy",
    "discount_purchase_ratio",
    "email_open_rate",
    "app_sessions_monthly",
    "return_rate"
]

def calculate_shannon_entropy(cat_shares: np.ndarray) -> np.ndarray:
    """
    Computes normalized Shannon entropy across category proportions.
    H_norm = - sum(p * ln(p)) / ln(K)
    Returns value between 0.0 (concentrated in 1 category) and 1.0 (equally spread across all 5).
    """
    # Number of categories
    k = cat_shares.shape[1]
    max_entropy = np.log(k)
    
    entropies = []
    for row in cat_shares:
        # filter out zero or negative shares to avoid log(0)
        p = row[row > 0]
        if len(p) <= 1:
            entropies.append(0.0)
        else:
            h = -np.sum(p * np.log(p))
            entropies.append(float(np.clip(h / max_entropy, 0.0, 1.0)))
            
    return np.array(entropies, dtype=float)


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the customer dataset with engineered marketing metrics.
    
    Computed metrics:
    - category_entropy: Normalized Shannon diversity index across 5 categories
    - rfm_r_score, rfm_f_score, rfm_m_score: Quintile ranks (1-5)
    - rfm_composite: Weighted composite score (1-5)
    - discount_sensitivity_score: Scale 0-100 indicating promotional dependency
    - omnichannel_engagement_score: Scale 0-100 measuring digital & marketing responsiveness
    - churn_risk_score: Estimated churn probability percentage [0-100%]
    - estimated_future_clv: 12-month forward predictive lifetime value projection
    """
    df = df.copy()
    
    # 1. Category Diversity (Shannon Entropy)
    cat_cols = ["pct_skincare", "pct_makeup", "pct_fragrance", "pct_haircare", "pct_bath_body"]
    cat_matrix = df[cat_cols].to_numpy()
    df["category_entropy"] = np.round(calculate_shannon_entropy(cat_matrix), 3)
    
    # 2. RFM Quintiles (qcut with fallback to cut for low unique values)
    # Recency: lower days is better -> invert labels
    df["rfm_r_score"] = pd.qcut(df["recency_days"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    # Frequency: higher count is better
    df["rfm_f_score"] = pd.qcut(df["frequency_12m"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary: higher spend is better
    df["rfm_m_score"] = pd.qcut(df["monetary_total_12m"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    
    df["rfm_composite"] = np.round(
        df["rfm_r_score"] * 0.25 + df["rfm_f_score"] * 0.35 + df["rfm_m_score"] * 0.40,
        2
    )
    
    # 3. Discount Sensitivity Score (0 to 100)
    # Weighted combination of discount purchase ratio and average discount depth
    df["discount_sensitivity_score"] = np.round(
        np.clip((df["discount_purchase_ratio"] * 75.0 + (df["avg_discount_depth"] / 0.25) * 25.0), 0.0, 100.0),
        1
    )
    
    # 4. Omnichannel Engagement Score (0 to 100)
    # Combines email responsiveness and app session activity
    norm_app = np.clip(df["app_sessions_monthly"] / 25.0, 0.0, 1.0)
    norm_open = np.clip(df["email_open_rate"], 0.0, 1.0)
    norm_click = np.clip(df["email_click_rate"] / 0.4, 0.0, 1.0)
    df["omnichannel_engagement_score"] = np.round(
        (norm_open * 35.0 + norm_click * 30.0 + norm_app * 35.0),
        1
    )
    
    # 5. Churn Risk Score (Heuristic probability 0 - 100%)
    # Logit-based formulation on recency, frequency, and engagement drop-off
    rec_factor = (df["recency_days"] - 45) / 60.0
    freq_factor = (df["frequency_12m"] - 6) / 4.0
    eng_factor = (df["omnichannel_engagement_score"] - 50) / 25.0
    
    # log-odds
    z = 0.8 * rec_factor - 0.7 * freq_factor - 0.6 * eng_factor
    prob = 1.0 / (1.0 + np.exp(-z))
    df["churn_risk_score"] = np.round(np.clip(prob * 100.0, 2.0, 98.0), 1)
    
    # 6. Estimated Future 12M CLV Projection
    # CLV = Current Spend * (1 - Churn Risk) * Tier Multiplier + baseline retention
    tier_mult = df["loyalty_tier"].map({"Rouge": 1.25, "VIB": 1.10, "Beauty Insider": 1.0}).fillna(1.0)
    retention_prob = 1.0 - (df["churn_risk_score"] / 100.0)
    df["estimated_future_clv"] = np.round(
        df["monetary_total_12m"] * retention_prob * tier_mult * 0.95 + 25.0,
        2
    )
    
    return df
