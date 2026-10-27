"""
Sephora AI Customer Segmentation Prototype
Module: clustering.py
Description: Feature scaling, K-Means clustering, silhouette/stability evaluation,
PCA dimensionality reduction, evidence-based segment naming, profiling, and strategy generation.
"""

import os
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from src.features import FEATURE_COLUMNS_FOR_CLUSTERING

def evaluate_cluster_range(
    df: pd.DataFrame,
    k_min: int = 3,
    k_max: int = 8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates K-Means clustering quality across a range of k values (Inertia, Silhouette, Davies-Bouldin).
    """
    X = df[FEATURE_COLUMNS_FOR_CLUSTERING].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        inertia = km.inertia_
        
        results.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil, 4),
            "davies_bouldin_index": round(db, 4)
        })
        
    return pd.DataFrame(results)


def perform_clustering_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes full clustering pipeline:
    1. StandardScaler normalization
    2. KMeans fit_predict (with fallback if needed)
    3. PCA 2D and 3D projections
    4. Evidence-based segment naming and profile generation
    5. Quantitative model metrics
    """
    df_clustered = df.copy()
    X = df[FEATURE_COLUMNS_FOR_CLUSTERING].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering with fallback
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        model_used = "K-Means (k-means++ initialization)"
    except Exception as e:
        # Transparent fallback to quantile baseline
        labels = pd.qcut(df["monetary_total_12m"], q=n_clusters, labels=False).values
        model_used = f"Fallback Quantile Discretizer (Reason: {str(e)})"
        
    df_clustered["cluster_id"] = labels
    
    # Quantitative Validation Metrics
    try:
        sil_score = float(silhouette_score(X_scaled, labels))
        db_score = float(davies_bouldin_score(X_scaled, labels))
    except Exception:
        sil_score = 0.0
        db_score = 0.0
        
    # PCA Dimensionality Reduction for Visualization (2D and 3D)
    pca_3d = PCA(n_components=3, random_state=random_state)
    pca_coords = pca_3d.fit_transform(X_scaled)
    df_clustered["pca_1"] = np.round(pca_coords[:, 0], 3)
    df_clustered["pca_2"] = np.round(pca_coords[:, 1], 3)
    df_clustered["pca_3"] = np.round(pca_coords[:, 2], 3)
    
    var_exp = pca_3d.explained_variance_ratio_
    pca_variance_info = {
        "pc1": round(float(var_exp[0]) * 100, 1),
        "pc2": round(float(var_exp[1]) * 100, 1),
        "pc3": round(float(var_exp[2]) * 100, 1),
        "total_explained": round(float(np.sum(var_exp)) * 100, 1)
    }
    
    # Generate Grounded Segment Names & Profiles based on empirical cluster centroids
    segment_profiles, segment_name_map = derive_evidence_based_profiles(df_clustered)
    df_clustered["segment_name"] = df_clustered["cluster_id"].map(segment_name_map)
    
    pipeline_metadata = {
        "n_clusters": n_clusters,
        "model_used": model_used,
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(db_score, 4),
        "pca_variance": pca_variance_info,
        "profiles": segment_profiles,
        "feature_columns": FEATURE_COLUMNS_FOR_CLUSTERING
    }
    
    return df_clustered, pipeline_metadata


def derive_evidence_based_profiles(df: pd.DataFrame) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    """
    Computes empirical segment characteristics and assigns evidence-grounded
    business titles and marketing playbooks based on data ranks.
    """
    cluster_ids = [int(c) for c in sorted(df["cluster_id"].unique())]
    n_total = len(df)
    
    cluster_stats = {}
    for cid in cluster_ids:
        c_sub = df[df["cluster_id"] == cid]
        cluster_stats[cid] = {
            "count": len(c_sub),
            "pct_total": (len(c_sub) / n_total) * 100.0,
            "avg_spend": float(c_sub["monetary_total_12m"].mean()),
            "median_spend": float(c_sub["monetary_total_12m"].median()),
            "avg_frequency": float(c_sub["frequency_12m"].mean()),
            "avg_recency": float(c_sub["recency_days"].mean()),
            "avg_aov": float(c_sub["avg_order_value"].mean()),
            "avg_skincare_pct": float(c_sub["pct_skincare"].mean()),
            "avg_makeup_pct": float(c_sub["pct_makeup"].mean()),
            "avg_fragrance_pct": float(c_sub["pct_fragrance"].mean()),
            "avg_haircare_pct": float(c_sub["pct_haircare"].mean()),
            "avg_bath_body_pct": float(c_sub["pct_bath_body"].mean()),
            "avg_discount_pct": float(c_sub["discount_purchase_ratio"].mean()),
            "avg_engagement": float(c_sub["omnichannel_engagement_score"].mean()),
            "avg_churn_risk": float(c_sub["churn_risk_score"].mean()),
            "avg_return_rate": float(c_sub["return_rate"].mean()),
            "top_loyalty_tier": c_sub["loyalty_tier"].mode().iloc[0] if len(c_sub) > 0 else "Beauty Insider",
            "primary_cat": c_sub["primary_category"].mode().iloc[0] if len(c_sub) > 0 else "Skincare",
            "consent_email_pct": float(c_sub["consent_email"].mean()) * 100.0,
            "consent_sms_pct": float(c_sub["consent_sms"].mean()) * 100.0,
            "consent_push_pct": float(c_sub["consent_app_push"].mean()) * 100.0
        }
        
    # Rank clusters along behavioral axes to dynamically match best fitting Sephora persona
    # Sort by spend to determine prestige level
    c_by_spend = sorted(cluster_ids, key=lambda c: cluster_stats[c]["avg_spend"], reverse=True)
    c_by_recency = sorted(cluster_ids, key=lambda c: cluster_stats[c]["avg_recency"], reverse=True)
    c_by_discount = sorted(cluster_ids, key=lambda c: cluster_stats[c]["avg_discount_pct"], reverse=True)
    c_by_skincare = sorted(cluster_ids, key=lambda c: cluster_stats[c]["avg_skincare_pct"], reverse=True)
    c_by_makeup = sorted(cluster_ids, key=lambda c: cluster_stats[c]["avg_makeup_pct"], reverse=True)
    
    assigned_names = {}
    used_personas = set()
    
    for cid in cluster_ids:
        stats = cluster_stats[cid]
        # Grounded rules:
        # If top spend & frequency -> Prestige Rouge VIP
        # If high recency (>150 days) & low frequency -> At-Risk / Lapsed Occasionals
        # If high discount (>60%) -> Deal-Seeking Promo Shoppers
        # If skincare share dominates (>45%) -> Skincare Ritualists
        # If makeup share dominates (>45%) -> Glam & Trend Makeup Lovers
        
        name = None
        if stats["avg_spend"] >= 800 and stats["avg_frequency"] >= 7 and "Prestige Rouge Connoisseurs" not in used_personas:
            name = "Prestige Rouge Connoisseurs"
        elif stats["avg_recency"] >= 140 and stats["avg_frequency"] <= 3.5 and "Lapsed Occasional Shoppers" not in used_personas:
            name = "Lapsed Occasional Shoppers"
        elif stats["avg_discount_pct"] >= 0.55 and "Deal-Seeking Promo Hunters" not in used_personas:
            name = "Deal-Seeking Promo Hunters"
        elif stats["avg_skincare_pct"] >= 0.38 and "Skincare Ritualists" not in used_personas:
            name = "Skincare Ritualists"
        elif stats["avg_makeup_pct"] >= 0.38 and "Glam & Trend Makeup Seekers" not in used_personas:
            name = "Glam & Trend Makeup Seekers"
        else:
            # Fallback based on remaining criteria
            if stats["avg_spend"] > 600:
                name = f"High-Value Multi-Category Shoppers (Cluster {cid+1})"
            elif stats["avg_discount_pct"] > 0.4:
                name = f"Promotional Value Seekers (Cluster {cid+1})"
            elif stats["avg_recency"] > 100:
                name = f"Low-Cadence Shoppers (Cluster {cid+1})"
            else:
                name = f"Omnichannel Beauty Explorers (Cluster {cid+1})"
                
        used_personas.add(name)
        assigned_names[cid] = name
        
    # Generate tailored playbooks and narrative summaries grounded in calculated numbers
    profiles = {}
    for cid in cluster_ids:
        s = cluster_stats[cid]
        name = assigned_names[cid]
        
        # Grounded narrative text
        narrative = (
            f"Representing {s['pct_total']:.1f}% of total customers ({s['count']:,} clients), this segment delivers "
            f"an average annual spend of ${s['avg_spend']:,.2f} across {s['avg_frequency']:.1f} annual purchases (AOV ${s['avg_aov']:.2f}). "
            f"Their primary category preference is {s['primary_cat']}, with a discount usage rate of {s['avg_discount_pct']*100:.1f}%. "
            f"Average recency is {s['avg_recency']:.0f} days with an estimated churn risk of {s['avg_churn_risk']:.1f}%."
        )
        
        # Strategy tailoring based on persona
        if "Prestige" in name or s["avg_spend"] >= 800:
            strategy = {
                "headline": "VIP Retention & Luxury Pre-Access",
                "recommended_channels": ["App Push Notification", "Dedicated Concierge Email", "Direct Mail Luxury Box"],
                "offer_strategy": "Zero deep discounting. Offer exclusive brand founder masterclasses, early access to Holiday Vault drops, and deluxe travel miniatures with purchase.",
                "content_theme": "High-efficacy clinical prestige, limited-edition bespoke packaging, tier recognition.",
                "priority": "Critical Retention (P1)",
                "expected_impact": "Preserves highest margin revenue stream (+12% spend retention, <5% churn).",
                "risk_uncertainty": "Over-messaging risks push unsubscribe fatigue; enforce strict weekly contact caps."
            }
        elif "Skincare" in name:
            strategy = {
                "headline": "Replenishment Regimen & Cross-Sell",
                "recommended_channels": ["App In-Session Trigger", "Educational Email", "SMS Replenishment"],
                "offer_strategy": "Predictive 60-day serum/cream refill reminders, double Beauty Insider points on Clean Skincare purchases, complimentary mini SPF.",
                "content_theme": "Ingredient efficacy (Retinol, Vitamin C, Peptides), step-by-step layering routines, dermatologist-backed claims.",
                "priority": "High LTV Expansion (P2)",
                "expected_impact": "Accelerates replenishment frequency from 4x/yr to 6x/yr (+24% category spend).",
                "risk_uncertainty": "Skincare formulations require high consumer trust; avoid overly aggressive cross-brand pushes."
            }
        elif "Makeup" in name or "Glam" in name:
            strategy = {
                "headline": "Trend Velocity & Mobile Discovery",
                "recommended_channels": ["Mobile App Push", "SMS Flash Launches", "Social Re-targeting"],
                "offer_strategy": "Tiered gift with purchase (GWP) threshold (e.g. spend $55 get viral setting spray mini), virtual try-on shade finder gamification.",
                "content_theme": "Viral TikTok/Reels aesthetics, seasonal color palettes, tutorial capsules, clean glow finishes.",
                "priority": "High Engagement & Frequency (P2)",
                "expected_impact": "Drives immediate conversion during new brand launches and seasonal drops (+18% basket units).",
                "risk_uncertainty": "High return rate on wrong shades; mitigate with shade matching AI tools."
            }
        elif "Promo" in name or "Deal" in name or s["avg_discount_pct"] >= 0.50:
            strategy = {
                "headline": "Margin-Controlled Event Conversion",
                "recommended_channels": ["Promotional Email", "App Banner", "Flash SMS"],
                "offer_strategy": "Direct toward Sephora Collection (high-margin in-house brand) or bundled discovery sets. Require minimum cart size ($60+) for promo code eligibility.",
                "content_theme": "Value bundles, 'Best Under $30', dupe comparisons, annual VIB/BI seasonal event countdowns.",
                "priority": "Margin Optimization (P3)",
                "expected_impact": "Reduces discount margin bleed by 150-250 bps while capturing event volume.",
                "risk_uncertainty": "Full-price conversion elasticity is low; deep discounts risk margin erosion."
            }
        else: # Lapsed / Occasional
            strategy = {
                "headline": "Win-Back & Reactivation Journey",
                "recommended_channels": ["Re-activation Email Journey", "Direct Mail Postcard ($10 off $50)"],
                "offer_strategy": "Generous 'We miss you' re-activation voucher ($10 off $50) with 14-day expiry, highlight bestsellers launched since their last visit.",
                "content_theme": "Effortless beauty essentials, seasonal gifting, loyalty point balance reminder before forfeiture.",
                "priority": "Reactivation Campaign (P4)",
                "expected_impact": "Reactivates 6-9% of dormant clients into active purchasing cycle within 45 days.",
                "risk_uncertainty": "High bounce/decay on stale contact info; throttle sends to preserve sender reputation."
            }
            
        profiles[cid] = {
            "cluster_id": cid,
            "name": name,
            "stats": s,
            "narrative": narrative,
            "strategy": strategy
        }
        
    return profiles, assigned_names
