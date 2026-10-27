"""
Sephora AI Customer Segmentation Prototype
Module: scenarios.py
Description: Marketing campaign budget allocation, expected-lift scenario modeling,
ROI / ROAS forecasting, and guardrail validations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

CHANNEL_UNIT_COSTS = {
    "Email": 0.02,        # $0.02 per recipient
    "SMS": 0.08,          # $0.08 per SMS message
    "App Push": 0.01,     # $0.01 per push delivery
    "Direct Mail": 1.25   # $1.25 per catalog/postcard printed & mailed
}

def simulate_campaign_scenario(
    df: pd.DataFrame,
    target_segments: List[str],
    channels: List[str],
    allocated_budget: float,
    expected_lift_pct: float,
    discount_depth_pct: float,
    baseline_margin_pct: float = 60.0,
    reach_pct: float = 100.0
) -> Dict[str, Any]:
    """
    Computes immediate campaign economics based on target segments, chosen channels,
    budget allocation, promotional discount depth, and response lift assumptions.
    """
    # 1. Filter customers in target segments
    target_df = df[df["segment_name"].isin(target_segments)].copy()
    total_segment_customers = len(target_df)
    
    if total_segment_customers == 0:
        return {
            "error": "No customers found matching selected segments.",
            "status": "EMPTY"
        }
        
    # 2. Check channel consent availability
    # Customers must have consented to at least one selected channel
    consent_mask = pd.Series(False, index=target_df.index)
    if "Email" in channels:
        consent_mask |= target_df["consent_email"]
    if "SMS" in channels:
        consent_mask |= target_df["consent_sms"]
    if "App Push" in channels:
        consent_mask |= target_df["consent_app_push"]
    if "Direct Mail" in channels:
        consent_mask |= target_df["consent_direct_mail"]
        
    consented_df = target_df[consent_mask]
    consented_count = len(consented_df)
    consent_rate = (consented_count / total_segment_customers) * 100.0 if total_segment_customers > 0 else 0.0
    
    # 3. Apply reach scale (user slider 10-100%)
    reachable_contacts = int(np.round(consented_count * (reach_pct / 100.0)))
    
    # 4. Channel Contact Cost Calculation
    channel_cost_per_contact = sum(CHANNEL_UNIT_COSTS.get(ch, 0.0) for ch in channels)
    raw_campaign_cost = reachable_contacts * channel_cost_per_contact
    
    # Check if budget caps contact count
    if allocated_budget > 0 and raw_campaign_cost > allocated_budget:
        budget_capped = True
        actual_campaign_cost = allocated_budget
        actual_contacts = int(np.floor(allocated_budget / channel_cost_per_contact)) if channel_cost_per_contact > 0 else reachable_contacts
    else:
        budget_capped = False
        actual_campaign_cost = raw_campaign_cost
        actual_contacts = reachable_contacts
        
    # 5. Baseline vs Incremental Conversions
    # Baseline conversion rate derived from historical frequency and engagement
    avg_aov = float(consented_df["avg_order_value"].mean()) if len(consented_df) > 0 else 75.0
    avg_historical_orders = float(consented_df["frequency_12m"].mean()) if len(consented_df) > 0 else 4.0
    
    # Baseline monthly conversion probability
    baseline_conversion_rate = np.clip((avg_historical_orders / 12.0) * 0.15, 0.02, 0.12)
    
    # Lift multiplier: lift is relative increase over baseline
    lift_multiplier = expected_lift_pct / 100.0
    incremental_conversion_rate = baseline_conversion_rate * lift_multiplier
    
    baseline_conversions = int(np.round(actual_contacts * baseline_conversion_rate))
    incremental_conversions = int(np.round(actual_contacts * incremental_conversion_rate))
    total_campaign_conversions = baseline_conversions + incremental_conversions
    
    # 6. Revenue & Profit Accounting
    discount_mult = 1.0 - (discount_depth_pct / 100.0)
    effective_aov = avg_aov * discount_mult
    
    incremental_gross_revenue = incremental_conversions * effective_aov
    total_campaign_gross_revenue = total_campaign_conversions * effective_aov
    
    # Effective margin after accounting for product cost and discount
    # Product cost = AOV * (1 - baseline_margin_pct)
    # Revenue = AOV * (1 - discount_depth_pct)
    # Unit Gross Profit = effective_aov - (avg_aov * (1 - baseline_margin_pct / 100.0))
    unit_product_cost = avg_aov * (1.0 - baseline_margin_pct / 100.0)
    unit_gross_profit = max(0.0, effective_aov - unit_product_cost)
    
    incremental_gross_profit = incremental_conversions * unit_gross_profit
    net_incremental_profit = incremental_gross_profit - actual_campaign_cost
    
    # Return on Investment (ROI) & Return on Ad Spend (ROAS)
    if actual_campaign_cost > 0:
        net_roi_pct = (net_incremental_profit / actual_campaign_cost) * 100.0
        roas = incremental_gross_revenue / actual_campaign_cost
    else:
        net_roi_pct = 0.0
        roas = 0.0
        
    # Guardrail flags
    guardrails = {
        "budget_capped": budget_capped,
        "is_profitable": net_incremental_profit > 0,
        "consent_warning": consent_rate < 50.0,
        "k_anonymity_pass": actual_contacts >= 30,
        "low_margin_warning": (discount_depth_pct >= baseline_margin_pct * 0.5)
    }
    
    return {
        "status": "SUCCESS",
        "target_segments": target_segments,
        "channels": channels,
        "total_segment_customers": total_segment_customers,
        "consented_customers": consented_count,
        "consent_rate_pct": round(consent_rate, 1),
        "actual_contacts": actual_contacts,
        "channel_cost_per_contact": round(channel_cost_per_contact, 3),
        "actual_campaign_cost": round(actual_campaign_cost, 2),
        "avg_aov": round(avg_aov, 2),
        "effective_aov": round(effective_aov, 2),
        "baseline_conversion_rate_pct": round(baseline_conversion_rate * 100, 2),
        "incremental_conversions": incremental_conversions,
        "total_campaign_conversions": total_campaign_conversions,
        "incremental_gross_revenue": round(incremental_gross_revenue, 2),
        "total_campaign_gross_revenue": round(total_campaign_gross_revenue, 2),
        "incremental_gross_profit": round(incremental_gross_profit, 2),
        "net_incremental_profit": round(net_incremental_profit, 2),
        "net_roi_pct": round(net_roi_pct, 1),
        "roas": round(roas, 2),
        "guardrails": guardrails
    }


def compare_scenarios(
    df: pd.DataFrame,
    target_segments: List[str],
    channels: List[str],
    budget: float,
    discount_depth: float,
    baseline_margin: float = 60.0
) -> pd.DataFrame:
    """
    Generates a 3-tier comparative matrix: Conservative (+15% lift), Base (+35% lift), Aggressive (+65% lift).
    """
    scenarios = [
        {"Case": "Conservative Case", "Lift": 15.0},
        {"Case": "Base Scenario", "Lift": 35.0},
        {"Case": "Aggressive Stretch", "Lift": 65.0}
    ]
    
    rows = []
    for sc in scenarios:
        res = simulate_campaign_scenario(
            df=df,
            target_segments=target_segments,
            channels=channels,
            allocated_budget=budget,
            expected_lift_pct=sc["Lift"],
            discount_depth_pct=discount_depth,
            baseline_margin_pct=baseline_margin
        )
        if res.get("status") == "SUCCESS":
            rows.append({
                "Scenario": sc["Case"],
                "Assumed Lift (%)": f"+{sc['Lift']:.0f}%",
                "Targeted Contacts": f"{res['actual_contacts']:,}",
                "Campaign Spend ($)": f"${res['actual_campaign_cost']:,.2f}",
                "Incr. Orders": f"{res['incremental_conversions']:,}",
                "Incr. Revenue ($)": f"${res['incremental_gross_revenue']:,.2f}",
                "Net Profit ($)": f"${res['net_incremental_profit']:,.2f}",
                "Net ROI (%)": f"{res['net_roi_pct']:.1f}%",
                "ROAS": f"{res['roas']:.2f}x"
            })
            
    return pd.DataFrame(rows)
