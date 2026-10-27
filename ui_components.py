"""
Sephora AI Customer Segmentation Prototype
Module: ui_components.py
Description: UI styling, dark luxury theme CSS, KPI cards, disclaimer banners,
and interactive Plotly visualization helpers.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Color Palette: Sephora Dark Luxury Aesthetic
THEME_COLORS = {
    "bg_main": "#0B0F19",
    "bg_card": "#131C2E",
    "bg_card_alt": "#1A253C",
    "border": "#273553",
    "accent_purple": "#A855F7",
    "accent_pink": "#EC4899",
    "accent_indigo": "#6366F1",
    "accent_cyan": "#06B6D4",
    "accent_gold": "#F59E0B",
    "accent_emerald": "#10B981",
    "accent_rose": "#F43F5E",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8"
}

PALETTE_CLUSTERS = [
    "#A855F7", # Electric Purple (Rouge / Prestige)
    "#EC4899", # Hot Glam Pink (Makeup Trend)
    "#06B6D4", # Clean Aqua (Skincare Ritual)
    "#F59E0B", # Radiant Amber (Deal / Promo)
    "#64748B", # Muted Slate (Lapsed / Occasional)
    "#10B981", # Emerald
    "#3B82F6", # Sapphire Blue
    "#F97316"  # Sunset Orange
]

def apply_custom_css():
    """Injects high-end dark luxury CSS styling for Streamlit."""
    css = f"""
    <style>
        /* Base page background & font */
        .stApp {{
            background-color: {THEME_COLORS['bg_main']};
            color: {THEME_COLORS['text_primary']};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: #0E1626;
            border-right: 1px solid {THEME_COLORS['border']};
        }}
        
        /* Metric Cards */
        .kpi-card {{
            background: linear-gradient(135deg, {THEME_COLORS['bg_card']} 0%, {THEME_COLORS['bg_card_alt']} 100%);
            border: 1px solid {THEME_COLORS['border']};
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
            transition: transform 0.2s ease, border-color 0.2s ease;
            margin-bottom: 12px;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: {THEME_COLORS['accent_purple']};
        }}
        .kpi-title {{
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {THEME_COLORS['text_secondary']};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 700;
            color: {THEME_COLORS['text_primary']};
            line-height: 1.2;
        }}
        .kpi-subtext {{
            font-size: 0.80rem;
            color: {THEME_COLORS['accent_purple']};
            margin-top: 6px;
        }}
        
        /* Synthetic Demo Banner */
        .synthetic-banner {{
            background: linear-gradient(90deg, rgba(168, 85, 247, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%);
            border: 1px solid rgba(168, 85, 247, 0.5);
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .synthetic-badge {{
            background-color: {THEME_COLORS['accent_purple']};
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            padding: 4px 8px;
            border-radius: 4px;
            text-transform: uppercase;
            margin-right: 12px;
            display: inline-block;
        }}
        .synthetic-text {{
            font-size: 0.88rem;
            color: #E2E8F0;
            font-weight: 500;
        }}
        
        /* Segment Profile Card */
        .segment-card {{
            background: {THEME_COLORS['bg_card']};
            border: 1px solid {THEME_COLORS['border']};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 600;
            color: {THEME_COLORS['text_secondary']};
            border: 1px solid transparent;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {THEME_COLORS['bg_card_alt']} !important;
            color: {THEME_COLORS['accent_purple']} !important;
            border: 1px solid {THEME_COLORS['accent_purple']} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_synthetic_banner():
    """Renders persistent disclaimer banner required by project guidelines."""
    st.markdown(
        """
        <div class="synthetic-banner">
            <div>
                <span class="synthetic-badge">SYNTHETIC DEMO DATA</span>
                <span class="synthetic-text">
                    Illustrative Prototype for Sephora Marketing Architecture &bull; 
                    <strong>100% Synthetic Fictional Data &bull; Not Real Sephora Company Records</strong>
                </span>
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; text-align: right;">
                Reproducible Seed Simulation &bull; Guardrails Active
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi_card(title: str, value: str, subtext: str = ""):
    """Renders a styled dark KPI card."""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_plotly_dark_layout(title: str = "") -> Dict[str, Any]:
    """Provides a consistent dark theme layout dict for Plotly figures."""
    return dict(
        title=dict(text=title, font=dict(color=THEME_COLORS['text_primary'], size=15, weight=600)),
        paper_bgcolor=THEME_COLORS['bg_card'],
        plot_bgcolor=THEME_COLORS['bg_card'],
        font=dict(color=THEME_COLORS['text_secondary'], family="sans-serif", size=11),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color=THEME_COLORS['text_secondary'])
        ),
        xaxis=dict(
            gridcolor=THEME_COLORS['border'],
            zerolinecolor=THEME_COLORS['border'],
            showline=True,
            linecolor=THEME_COLORS['border']
        ),
        yaxis=dict(
            gridcolor=THEME_COLORS['border'],
            zerolinecolor=THEME_COLORS['border'],
            showline=True,
            linecolor=THEME_COLORS['border']
        )
    )


def plot_pca_2d(df: pd.DataFrame) -> go.Figure:
    """Generates an interactive 2D PCA scatter plot of behavioral segments."""
    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="segment_name",
        color_discrete_sequence=PALETTE_CLUSTERS,
        hover_data={
            "customer_id": True,
            "loyalty_tier": True,
            "monetary_total_12m": ":$.2f",
            "frequency_12m": True,
            "primary_category": True,
            "discount_purchase_ratio": ":.1%",
            "pca_1": False,
            "pca_2": False,
            "segment_name": True
        },
        title="2D Behavioral Feature Map (PCA Decomposition)",
        labels={"pca_1": "Principal Component 1 (Spend & Frequency)", "pca_2": "Principal Component 2 (Category & Discount Elasticity)"}
    )
    layout = get_plotly_dark_layout("2D Behavioral Feature Map (PCA Decomposition)")
    layout["height"] = 480
    fig.update_layout(**layout)
    fig.update_traces(marker=dict(size=7, opacity=0.82, line=dict(width=0.5, color='#FFFFFF')))
    return fig


def plot_pca_3d(df: pd.DataFrame) -> go.Figure:
    """Generates an interactive 3D PCA scatter plot with camera orbit."""
    fig = px.scatter_3d(
        df,
        x="pca_1",
        y="pca_2",
        z="pca_3",
        color="segment_name",
        color_discrete_sequence=PALETTE_CLUSTERS,
        hover_data={
            "customer_id": True,
            "loyalty_tier": True,
            "monetary_total_12m": ":$.2f",
            "frequency_12m": True,
            "primary_category": True,
            "pca_1": False,
            "pca_2": False,
            "pca_3": False
        },
        title="3D Latent Cluster Space (PC1 x PC2 x PC3)",
        labels={"pca_1": "PC1: Value", "pca_2": "PC2: Category", "pca_3": "PC3: Engagement"}
    )
    fig.update_layout(
        paper_bgcolor=THEME_COLORS['bg_card'],
        font=dict(color=THEME_COLORS['text_secondary']),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
        scene=dict(
            xaxis=dict(backgroundcolor=THEME_COLORS['bg_card'], gridcolor=THEME_COLORS['border']),
            yaxis=dict(backgroundcolor=THEME_COLORS['bg_card'], gridcolor=THEME_COLORS['border']),
            zaxis=dict(backgroundcolor=THEME_COLORS['bg_card'], gridcolor=THEME_COLORS['border']),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_traces(marker=dict(size=4.5, opacity=0.85))
    return fig


def plot_segment_size_value_treemap(df: pd.DataFrame) -> go.Figure:
    """Treemap of segments: Box size = Customer count, Box Color = Total Revenue."""
    summary = df.groupby("segment_name").agg(
        customer_count=("customer_id", "count"),
        total_revenue=("monetary_total_12m", "sum"),
        avg_spend=("monetary_total_12m", "mean")
    ).reset_index()
    
    summary["formatted_rev"] = summary["total_revenue"].apply(lambda x: f"${x:,.0f}")
    summary["formatted_spend"] = summary["avg_spend"].apply(lambda x: f"${x:,.2f}")
    
    fig = px.treemap(
        summary,
        path=["segment_name"],
        values="customer_count",
        color="total_revenue",
        color_continuous_scale="Purp",
        title="Segment Population (Area) vs Total Revenue Contribution (Color)",
        hover_data={"formatted_rev": True, "formatted_spend": True, "customer_count": True}
    )
    layout = get_plotly_dark_layout("Segment Population vs Revenue Contribution")
    layout["height"] = 400
    fig.update_layout(**layout)
    return fig


def plot_pareto_revenue(df: pd.DataFrame) -> go.Figure:
    """Lorenz / Pareto curve showing customer cumulative % vs cumulative revenue %."""
    sorted_df = df.sort_values(by="monetary_total_12m", ascending=False).reset_index(drop=True)
    sorted_df["cum_spend"] = sorted_df["monetary_total_12m"].cumsum()
    total_rev = sorted_df["monetary_total_12m"].sum()
    sorted_df["cum_spend_pct"] = (sorted_df["cum_spend"] / total_rev) * 100.0
    sorted_df["cum_cust_pct"] = ((sorted_df.index + 1) / len(sorted_df)) * 100.0
    
    fig = go.Figure()
    
    # 45-degree parity line
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines",
        name="Equal Distribution (1:1)",
        line=dict(color="#475569", dash="dash")
    ))
    
    # Sephora Pareto curve
    fig.add_trace(go.Scatter(
        x=sorted_df["cum_cust_pct"],
        y=sorted_df["cum_spend_pct"],
        mode="lines",
        name="Sephora Cumulative Revenue",
        line=dict(color=THEME_COLORS['accent_purple'], width=3),
        hovertemplate="Top %{x:.1f}% of Customers generate %{y:.1f}% of Revenue<extra></extra>"
    ))
    
    # Highlight 80/20 threshold
    fig.add_vline(x=20, line_dash="dot", line_color=THEME_COLORS['accent_pink'], annotation_text="Top 20% Base", annotation_position="top left")
    
    layout = get_plotly_dark_layout("Revenue Concentration (Pareto / Lorenz Curve)")
    layout["xaxis"]["title"] = "Cumulative % of Customers (Ranked by Spend)"
    layout["yaxis"]["title"] = "Cumulative % of 12-Month Revenue"
    layout["height"] = 400
    fig.update_layout(**layout)
    return fig


def plot_segment_radar_profiles(profiles: Dict[int, Dict[str, Any]]) -> go.Figure:
    """Generates a spider / radar chart comparing normalized behavioral dimensions."""
    categories = [
        "Monetary Spend",
        "Purchase Frequency",
        "Recency (Inverse)",
        "Category Diversity",
        "Discount Reliance",
        "Omnichannel Activity"
    ]
    
    fig = go.Figure()
    
    for idx, (cid, p) in enumerate(profiles.items()):
        s = p["stats"]
        # Normalize each metric to 0-100 scale for visual comparison
        norm_spend = min(100.0, (s["avg_spend"] / 1200.0) * 100.0)
        norm_freq = min(100.0, (s["avg_frequency"] / 15.0) * 100.0)
        norm_recency = max(0.0, 100.0 - (s["avg_recency"] / 2.5))
        norm_entropy = 70.0 if "Prestige" in p["name"] else (35.0 if "Skincare" in p["name"] else 50.0)
        norm_discount = s["avg_discount_pct"] * 100.0
        norm_engagement = s["avg_engagement"]
        
        values = [norm_spend, norm_freq, norm_recency, norm_entropy, norm_discount, norm_engagement]
        values.append(values[0])  # Close the radar loop
        
        color = PALETTE_CLUSTERS[idx % len(PALETTE_CLUSTERS)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=p["name"][:24],
            line=dict(color=color, width=2),
            fill='toself',
            opacity=0.25
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=THEME_COLORS['border']),
            angularaxis=dict(gridcolor=THEME_COLORS['border'], linecolor=THEME_COLORS['border']),
            bgcolor=THEME_COLORS['bg_card']
        ),
        paper_bgcolor=THEME_COLORS['bg_card'],
        font=dict(color=THEME_COLORS['text_secondary']),
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    return fig


def plot_category_mix_by_segment(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing category share % per segment."""
    summary = df.groupby("segment_name")[
        ["pct_skincare", "pct_makeup", "pct_fragrance", "pct_haircare", "pct_bath_body"]
    ].mean() * 100.0
    
    summary.columns = ["Skincare", "Makeup", "Fragrance", "Haircare", "Bath & Body"]
    summary = summary.reset_index()
    
    fig = go.Figure()
    cat_colors = {
        "Skincare": "#06B6D4",
        "Makeup": "#EC4899",
        "Fragrance": "#A855F7",
        "Haircare": "#F59E0B",
        "Bath & Body": "#10B981"
    }
    
    for cat in ["Skincare", "Makeup", "Fragrance", "Haircare", "Bath & Body"]:
        fig.add_trace(go.Bar(
            y=summary["segment_name"],
            x=summary[cat],
            name=cat,
            orientation="h",
            marker=dict(color=cat_colors[cat])
        ))
        
    layout = get_plotly_dark_layout("Product Category Mix by Segment (% Share)")
    layout["barmode"] = "stack"
    layout["xaxis"]["title"] = "Average Category Share (%)"
    layout["height"] = 380
    fig.update_layout(**layout)
    return fig


def plot_customer_category_donut(customer_row: pd.Series) -> go.Figure:
    """Donut chart for individual customer category spend drilldown."""
    labels = ["Skincare", "Makeup", "Fragrance", "Haircare", "Bath & Body"]
    values = [
        customer_row["spend_skincare"],
        customer_row["spend_makeup"],
        customer_row["spend_fragrance"],
        customer_row["spend_haircare"],
        customer_row["spend_bath_body"]
    ]
    colors = ["#06B6D4", "#EC4899", "#A855F7", "#F59E0B", "#10B981"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        insidetextorientation="radial"
    )])
    
    layout = get_plotly_dark_layout(f"Spend Breakdown for {customer_row['customer_id']}")
    layout["height"] = 320
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def plot_cluster_evaluation_curves(eval_df: pd.DataFrame, selected_k: int) -> go.Figure:
    """Renders dual-axis evaluation curve for Elbow Inertia and Silhouette Score."""
    fig = go.Figure()
    
    # Inertia line (Elbow)
    fig.add_trace(go.Scatter(
        x=eval_df["k"],
        y=eval_df["inertia"],
        mode="lines+markers",
        name="Inertia (Within-Cluster Sum of Squares)",
        line=dict(color="#3B82F6", width=2.5),
        yaxis="y1"
    ))
    
    # Silhouette Score line
    fig.add_trace(go.Scatter(
        x=eval_df["k"],
        y=eval_df["silhouette_score"],
        mode="lines+markers",
        name="Silhouette Score (Cohesion & Separation)",
        line=dict(color=THEME_COLORS['accent_purple'], width=2.5),
        yaxis="y2"
    ))
    
    # Highlight chosen K
    fig.add_vline(x=selected_k, line_dash="dash", line_color=THEME_COLORS['accent_pink'], annotation_text=f"Selected K={selected_k}")
    
    layout = get_plotly_dark_layout("K-Means Diagnostic Evaluation Curves")
    layout["xaxis"] = dict(title="Cluster Count (k)", gridcolor=THEME_COLORS['border'], dtick=1)
    layout["yaxis"] = dict(title="Inertia (Lower is tighter)", titlefont=dict(color="#3B82F6"), tickfont=dict(color="#3B82F6"), gridcolor=THEME_COLORS['border'])
    layout["yaxis2"] = dict(
        title="Silhouette Score (Higher is better)",
        titlefont=dict(color=THEME_COLORS['accent_purple']),
        tickfont=dict(color=THEME_COLORS['accent_purple']),
        overlaying="y",
        side="right",
        showgrid=False
    )
    layout["height"] = 380
    fig.update_layout(**layout)
    return fig
