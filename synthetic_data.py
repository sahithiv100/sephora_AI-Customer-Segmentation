"""
Sephora AI Customer Segmentation Prototype
Module: synthetic_data.py
Description: Generates realistic, internally consistent fictional customer
and transaction datasets for Sephora marketing segmentation demonstrations.

DISCLAIMER: ALL DATA GENERATED IS 100% SYNTHETIC AND FICTIONAL.
DOES NOT CONTAIN REAL SEPHORA CUSTOMER DATA, PROPRIETARY METRICS, OR SCRAPED INFORMATION.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Fictional Sephora product categories
CATEGORIES = ["Skincare", "Makeup", "Fragrance", "Haircare", "Bath & Body"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
LOYALTY_TIERS = ["Beauty Insider", "VIB", "Rouge"]

def generate_synthetic_data(
    n_customers: int = 1500,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic Sephora customer records and simulated transaction histories.
    
    Parameters:
        n_customers (int): Number of customer records to generate (default 1500)
        random_seed (int): Seed for reproducibility
        
    Returns:
        customers_df (pd.DataFrame): Aggregated customer feature profiles
        transactions_df (pd.DataFrame): Simulated granular transaction log
    """
    rng = np.random.default_rng(random_seed)
    
    # Archetype distribution weights:
    # 0: Prestige Rouge Connoisseurs (~20%)
    # 1: Skincare Ritualists (~22%)
    # 2: Glam & Trend Makeup Seekers (~25%)
    # 3: Deal-Seeking Promo Shoppers (~20%)
    # 4: Occasional / At-Risk Shoppers (~13%)
    archetype_probs = [0.20, 0.22, 0.25, 0.20, 0.13]
    archetypes = rng.choice(5, size=n_customers, p=archetype_probs)
    
    customer_ids = [f"SEP-{100000 + i}" for i in range(n_customers)]
    records = []
    all_transactions = []
    
    for i, (cid, arch) in enumerate(zip(customer_ids, archetypes)):
        # Base attributes by archetype
        if arch == 0:  # Prestige Rouge Connoisseurs
            tier = "Rouge" if rng.random() > 0.08 else "VIB"
            frequency = int(rng.integers(8, 22))
            aov = float(rng.normal(125, 30))
            aov = max(65.0, aov)
            recency_days = int(rng.integers(3, 45))
            cat_weights = [0.35, 0.30, 0.20, 0.10, 0.05]  # Balanced luxury mix
            discount_ratio = float(np.clip(rng.beta(2, 6), 0.05, 0.40))  # Low discount reliance
            avg_discount_depth = 0.15
            email_open_rate = float(np.clip(rng.normal(0.62, 0.12), 0.25, 0.95))
            email_click_rate = float(np.clip(email_open_rate * rng.uniform(0.35, 0.65), 0.05, 0.55))
            app_sessions = int(rng.integers(12, 35))
            store_ratio = float(np.clip(rng.normal(0.45, 0.18), 0.10, 0.90))
            return_rate = float(np.clip(rng.beta(1.5, 18), 0.01, 0.12))
            sample_affinity = int(rng.integers(2, 5))
            tenure_months = int(rng.integers(24, 72))
            age_group = rng.choice(AGE_GROUPS, p=[0.10, 0.35, 0.30, 0.15, 0.10])
            consent_sms = rng.random() < 0.78
            consent_push = rng.random() < 0.88
            
        elif arch == 1:  # Skincare Ritualists
            tier = rng.choice(LOYALTY_TIERS, p=[0.20, 0.48, 0.32])
            frequency = int(rng.integers(4, 12))
            aov = float(rng.normal(105, 25))
            aov = max(50.0, aov)
            recency_days = int(rng.integers(15, 90))
            cat_weights = [0.65, 0.12, 0.08, 0.10, 0.05]  # Skincare dominant
            discount_ratio = float(np.clip(rng.beta(2.5, 4.5), 0.15, 0.55))
            avg_discount_depth = 0.15
            email_open_rate = float(np.clip(rng.normal(0.50, 0.14), 0.15, 0.85))
            email_click_rate = float(np.clip(email_open_rate * rng.uniform(0.30, 0.55), 0.03, 0.45))
            app_sessions = int(rng.integers(6, 20))
            store_ratio = float(np.clip(rng.normal(0.35, 0.20), 0.05, 0.85))
            return_rate = float(np.clip(rng.beta(1.2, 22), 0.01, 0.09))
            sample_affinity = int(rng.integers(3, 5))
            tenure_months = int(rng.integers(12, 60))
            age_group = rng.choice(AGE_GROUPS, p=[0.12, 0.40, 0.28, 0.14, 0.06])
            consent_sms = rng.random() < 0.60
            consent_push = rng.random() < 0.70
            
        elif arch == 2:  # Glam & Trend Makeup Seekers
            tier = rng.choice(LOYALTY_TIERS, p=[0.50, 0.38, 0.12])
            frequency = int(rng.integers(5, 14))
            aov = float(rng.normal(68, 18))
            aov = max(30.0, aov)
            recency_days = int(rng.integers(7, 60))
            cat_weights = [0.15, 0.62, 0.08, 0.10, 0.05]  # Makeup dominant
            discount_ratio = float(np.clip(rng.beta(3, 4), 0.20, 0.65))
            avg_discount_depth = 0.18
            email_open_rate = float(np.clip(rng.normal(0.42, 0.15), 0.10, 0.80))
            email_click_rate = float(np.clip(email_open_rate * rng.uniform(0.35, 0.60), 0.02, 0.45))
            app_sessions = int(rng.integers(10, 30))
            store_ratio = float(np.clip(rng.normal(0.55, 0.20), 0.10, 0.95))
            return_rate = float(np.clip(rng.beta(2, 14), 0.02, 0.18))
            sample_affinity = int(rng.integers(1, 4))
            tenure_months = int(rng.integers(6, 42))
            age_group = rng.choice(AGE_GROUPS, p=[0.45, 0.35, 0.12, 0.06, 0.02])
            consent_sms = rng.random() < 0.68
            consent_push = rng.random() < 0.82
            
        elif arch == 3:  # Deal-Seeking Promo Shoppers
            tier = rng.choice(LOYALTY_TIERS, p=[0.60, 0.35, 0.05])
            frequency = int(rng.integers(2, 6))
            aov = float(rng.normal(55, 15))
            aov = max(25.0, aov)
            recency_days = int(rng.integers(20, 150))
            cat_weights = [0.25, 0.35, 0.15, 0.15, 0.10]
            discount_ratio = float(np.clip(rng.beta(6, 2), 0.60, 0.98))  # High discount affinity
            avg_discount_depth = 0.20
            email_open_rate = float(np.clip(rng.normal(0.55, 0.14), 0.15, 0.90))  # Opens sale emails
            email_click_rate = float(np.clip(email_open_rate * rng.uniform(0.40, 0.70), 0.05, 0.55))
            app_sessions = int(rng.integers(2, 10))
            store_ratio = float(np.clip(rng.normal(0.30, 0.22), 0.05, 0.80))
            return_rate = float(np.clip(rng.beta(3, 12), 0.04, 0.25))  # Higher returns on impulse promos
            sample_affinity = int(rng.integers(2, 5))
            tenure_months = int(rng.integers(6, 36))
            age_group = rng.choice(AGE_GROUPS, p=[0.30, 0.35, 0.20, 0.10, 0.05])
            consent_sms = rng.random() < 0.45
            consent_push = rng.random() < 0.50
            
        else:  # Occasional / At-Risk Shoppers
            tier = "Beauty Insider" if rng.random() > 0.05 else "VIB"
            frequency = int(rng.integers(1, 3))
            aov = float(rng.normal(48, 16))
            aov = max(20.0, aov)
            recency_days = int(rng.integers(140, 360))  # Long lapse
            cat_weights = [0.20, 0.25, 0.30, 0.15, 0.10]  # Often gift fragrance or simple makeup
            discount_ratio = float(np.clip(rng.beta(2, 3), 0.10, 0.70))
            avg_discount_depth = 0.15
            email_open_rate = float(np.clip(rng.normal(0.18, 0.10), 0.02, 0.50))
            email_click_rate = float(np.clip(email_open_rate * rng.uniform(0.10, 0.30), 0.00, 0.20))
            app_sessions = int(rng.integers(0, 4))
            store_ratio = float(np.clip(rng.normal(0.60, 0.25), 0.10, 0.95))
            return_rate = float(np.clip(rng.beta(1, 20), 0.00, 0.10))
            sample_affinity = int(rng.integers(0, 2))
            tenure_months = int(rng.integers(3, 48))
            age_group = rng.choice(AGE_GROUPS, p=[0.20, 0.25, 0.25, 0.18, 0.12])
            consent_sms = rng.random() < 0.25
            consent_push = rng.random() < 0.30
            
        # Category proportions with Dirichlet noise around weights
        dirichlet_alpha = np.array(cat_weights) * 15.0
        cat_shares = rng.dirichlet(dirichlet_alpha)
        
        # Calculate annual monetary spend
        monetary_total = round(float(frequency * aov), 2)
        
        # Category spend breakdown
        spend_skincare = round(monetary_total * cat_shares[0], 2)
        spend_makeup = round(monetary_total * cat_shares[1], 2)
        spend_fragrance = round(monetary_total * cat_shares[2], 2)
        spend_haircare = round(monetary_total * cat_shares[3], 2)
        spend_bath_body = round(monetary_total * cat_shares[4], 2)
        
        # Primary category
        cat_spend_map = {
            "Skincare": spend_skincare,
            "Makeup": spend_makeup,
            "Fragrance": spend_fragrance,
            "Haircare": spend_haircare,
            "Bath & Body": spend_bath_body
        }
        primary_cat = max(cat_spend_map, key=cat_spend_map.get)
        
        # Consented channels
        consent_email = rng.random() < 0.90
        consent_mail = rng.random() < 0.32
        
        records.append({
            "customer_id": cid,
            "synthetic_archetype_id": arch,
            "loyalty_tier": tier,
            "age_group": age_group,
            "tenure_months": tenure_months,
            "recency_days": recency_days,
            "frequency_12m": frequency,
            "monetary_total_12m": monetary_total,
            "avg_order_value": round(aov, 2),
            "pct_skincare": round(cat_shares[0], 4),
            "pct_makeup": round(cat_shares[1], 4),
            "pct_fragrance": round(cat_shares[2], 4),
            "pct_haircare": round(cat_shares[3], 4),
            "pct_bath_body": round(cat_shares[4], 4),
            "spend_skincare": spend_skincare,
            "spend_makeup": spend_makeup,
            "spend_fragrance": spend_fragrance,
            "spend_haircare": spend_haircare,
            "spend_bath_body": spend_bath_body,
            "primary_category": primary_cat,
            "discount_purchase_ratio": round(discount_ratio, 3),
            "avg_discount_depth": round(avg_discount_depth, 2),
            "email_open_rate": round(email_open_rate, 3),
            "email_click_rate": round(email_click_rate, 3),
            "app_sessions_monthly": app_sessions,
            "store_purchase_ratio": round(store_ratio, 3),
            "return_rate": round(return_rate, 3),
            "sample_affinity_score": sample_affinity,
            "consent_email": consent_email,
            "consent_sms": consent_sms,
            "consent_app_push": consent_push,
            "consent_direct_mail": consent_mail,
            "is_synthetic": True
        })
        
        # Generate simulated transaction records for customer drilldown
        # Generate min(frequency, 5) sample transactions to keep memory light
        n_sample_tx = min(frequency, 6)
        tx_dates_back = np.sort(rng.integers(recency_days, 365, size=n_sample_tx))
        for tx_idx, d_back in enumerate(tx_dates_back):
            tx_cat = rng.choice(CATEGORIES, p=cat_shares)
            tx_amt = round(max(15.0, float(rng.normal(aov, 15))), 2)
            has_disc = rng.random() < discount_ratio
            all_transactions.append({
                "transaction_id": f"TX-{cid[-5:]}-{tx_idx+1}",
                "customer_id": cid,
                "days_ago": int(d_back),
                "category": tx_cat,
                "order_value": tx_amt,
                "applied_discount": has_disc,
                "channel": "Store" if rng.random() < store_ratio else ("App" if consent_push else "Web")
            })

    customers_df = pd.DataFrame(records)
    transactions_df = pd.DataFrame(all_transactions)
    
    return customers_df, transactions_df


def get_data_dictionary() -> pd.DataFrame:
    """Returns a structured metadata dictionary describing all synthetic dataset columns."""
    dict_records = [
        {"Column": "customer_id", "Type": "Categorical", "Unit": "ID string", "Description": "Unique fictional Sephora client identifier (e.g., SEP-100001)."},
        {"Column": "loyalty_tier", "Type": "Categorical", "Unit": "Tier", "Description": "Sephora Beauty Insider loyalty tier: Beauty Insider, VIB, or Rouge."},
        {"Column": "age_group", "Type": "Categorical", "Unit": "Age bracket", "Description": "Synthetic demographic age group bracket."},
        {"Column": "tenure_months", "Type": "Integer", "Unit": "Months", "Description": "Customer lifetime tenure with Sephora in months."},
        {"Column": "recency_days", "Type": "Integer", "Unit": "Days", "Description": "Days elapsed since the customer's most recent completed order."},
        {"Column": "frequency_12m", "Type": "Integer", "Unit": "Count", "Description": "Total completed purchase transactions in the trailing 12 months."},
        {"Column": "monetary_total_12m", "Type": "Float", "Unit": "USD ($)", "Description": "Total gross spend across all sales channels in trailing 12 months."},
        {"Column": "avg_order_value", "Type": "Float", "Unit": "USD ($)", "Description": "Average transaction basket value (monetary_total / frequency)."},
        {"Column": "primary_category", "Type": "Categorical", "Unit": "Category", "Description": "Product department generating highest share of customer spend."},
        {"Column": "pct_skincare", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Share of 12-month spend allocated to Skincare products."},
        {"Column": "pct_makeup", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Share of 12-month spend allocated to Makeup products."},
        {"Column": "pct_fragrance", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Share of 12-month spend allocated to Fragrance products."},
        {"Column": "pct_haircare", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Share of 12-month spend allocated to Haircare products."},
        {"Column": "pct_bath_body", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Share of 12-month spend allocated to Bath & Body products."},
        {"Column": "discount_purchase_ratio", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Proportion of orders placed utilizing a discount, promo code, or sale event."},
        {"Column": "email_open_rate", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Percentage of marketing promotional emails opened in trailing 12 months."},
        {"Column": "email_click_rate", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Percentage of marketing emails clicked through to site/app."},
        {"Column": "app_sessions_monthly", "Type": "Integer", "Unit": "Sessions/mo", "Description": "Average monthly active browsing/shopping sessions on the Sephora iOS/Android app."},
        {"Column": "store_purchase_ratio", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Share of purchases conducted in brick-and-mortar stores vs digital channels."},
        {"Column": "return_rate", "Type": "Float", "Unit": "Ratio [0-1]", "Description": "Returned merchandise value divided by gross purchase value."},
        {"Column": "sample_affinity_score", "Type": "Integer", "Unit": "Score [0-4]", "Description": "Average number of complimentary luxury samples redeemed per checkout."},
        {"Column": "consent_email", "Type": "Boolean", "Unit": "True/False", "Description": "Customer opted in to promotional email marketing communications."},
        {"Column": "consent_sms", "Type": "Boolean", "Unit": "True/False", "Description": "Customer opted in to transactional and marketing SMS alerts."},
        {"Column": "consent_app_push", "Type": "Boolean", "Unit": "True/False", "Description": "Customer enabled mobile push notifications on Sephora app."},
        {"Column": "consent_direct_mail", "Type": "Boolean", "Unit": "True/False", "Description": "Customer opted in to postal mail catalogs and Rouge gift packages."},
        {"Column": "is_synthetic", "Type": "Boolean", "Unit": "Flag", "Description": "Explicit synthetic identifier verifying data is non-production simulation."}
    ]
    return pd.DataFrame(dict_records)
