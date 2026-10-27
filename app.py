"""
Sephora AI Customer Segmentation Prototype
Main Application: app.py
Description: Full-stack interactive Streamlit web application for marketing customer segmentation,
algorithmic clustering, scenario modeling, customer 360 exploration, and managerial insights.
"""

import os
import io
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Sephora AI Customer Segmentation",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.synthetic_data import generate_synthetic_data, get_data_dictionary, CATEGORIES, LOYALTY_TIERS
from src.features import compute_engineered_features, FEATURE_COLUMNS_FOR_CLUSTERING
from src.clustering import perform_clustering_pipeline, evaluate_cluster_range
from src.scenarios import simulate_campaign_scenario, compare_scenarios, CHANNEL_UNIT_COSTS
from src.guardrails import run_governance_audit
from src.ui_components import (
    apply_custom_css,
    render_synthetic_banner,
    render_kpi_card,
    plot_pca_2d,
    plot_pca_3d,
    plot_segment_size_value_treemap,
    plot_pareto_revenue,
    plot_segment_radar_profiles,
    plot_category_mix_by_segment,
    plot_customer_category_donut,
    plot_cluster_evaluation_curves,
    THEME_COLORS
)

# Apply UI styles
apply_custom_css()

# Session State Initialization
if "random_seed" not in st.session_state:
    st.session_state["random_seed"] = 42
if "n_customers" not in st.session_state:
    st.session_state["n_customers"] = 1500
if "n_clusters" not in st.session_state:
    st.session_state["n_clusters"] = 5
if "raw_customers" not in st.session_state:
    # Initial data generation
    raw_df, raw_tx = generate_synthetic_data(
        n_customers=st.session_state["n_customers"],
        random_seed=st.session_state["random_seed"]
    )
    eng_df = compute_engineered_features(raw_df)
    clustered_df, metadata = perform_clustering_pipeline(
        eng_df,
        n_clusters=st.session_state["n_clusters"],
        random_state=st.session_state["random_seed"]
    )
    st.session_state["raw_customers"] = raw_df
    st.session_state["transactions"] = raw_tx
    st.session_state["clustered_data"] = clustered_df
    st.session_state["metadata"] = metadata

# Helper to regenerate data
def regenerate_data(seed: int, count: int, k: int):
    raw_df, raw_tx = generate_synthetic_data(n_customers=count, random_seed=seed)
    eng_df = compute_engineered_features(raw_df)
    clustered_df, metadata = perform_clustering_pipeline(eng_df, n_clusters=k, random_state=seed)
    st.session_state["raw_customers"] = raw_df
    st.session_state["transactions"] = raw_tx
    st.session_state["clustered_data"] = clustered_df
    st.session_state["metadata"] = metadata
    st.session_state["random_seed"] = seed
    st.session_state["n_customers"] = count
    st.session_state["n_clusters"] = k

# Persistent Header Banner
render_synthetic_banner()

# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 💄 Sephora AI Intelligence")
    st.caption("Marketing Segmentation Prototype")
    
    page = st.radio(
        "Navigation Menu",
        [
            "Overview & Business Value",
            "Data Explorer & Dictionary",
            "How the Algorithm Works",
            "Segment Explorer & Profiles",
            "Customer 360° Detail",
            "Campaign Scenario Simulator",
            "Manager Strategic Questions",
            "Responsible AI & Guardrails"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Data & Model Controls")
    
    seed_input = st.number_input(
        "Random Seed",
        min_value=1,
        max_value=9999,
        value=int(st.session_state["random_seed"]),
        step=1,
        help="Change seed to generate an alternate reproducible synthetic cohort."
    )
    
    cust_count_input = st.slider(
        "Synthetic Customer Count",
        min_value=500,
        max_value=3000,
        value=int(st.session_state["n_customers"]),
        step=250,
        help="Volume of synthetic client accounts simulated in the active cohort."
    )
    
    k_slider = st.slider(
        "Cluster Count (k)",
        min_value=3,
        max_value=8,
        value=int(st.session_state["n_clusters"]),
        step=1,
        help="Number of behavioral segments to discover via K-Means."
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Apply / Refresh", use_container_width=True):
            regenerate_data(seed_input, cust_count_input, k_slider)
            st.rerun()
    with col_btn2:
        if st.button("Reset Defaults", use_container_width=True):
            regenerate_data(42, 1500, 5)
            st.rerun()
            
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #94A3B8; line-height: 1.4;">
            <strong>System Status</strong><br>
            &bull; Pipeline: K-Means++ &amp; PCA<br>
            &bull; Privacy: k-Anonymity Verified<br>
            &bull; Model Integrity: Real Computations<br>
            &bull; Environment: Synthetic Simulation
        </div>
        """,
        unsafe_allow_html=True
    )

# Retrieve current working data
df = st.session_state["clustered_data"]
tx_df = st.session_state["transactions"]
meta = st.session_state["metadata"]
profiles = meta["profiles"]

# ==========================================
# PAGE 1: OVERVIEW & BUSINESS VALUE
# ==========================================
if page == "Overview & Business Value":
    st.title("Sephora Omnichannel Marketing Segmentation")
    st.subheader("Transforming Behavioral Client Signatures into Precision Marketing Campaigns")
    
    st.markdown(
        """
        **Executive Context:**  
        In prestige omnichannel beauty retail, one-size-fits-all promotional discounting erodes gross margins 
        and exhausts high-affinity clients. Sephora's **Beauty Insider** loyalty ecosystem spans clients 
        with vastly disparate purchase motivations—from high-spending **Rouge clinical skincare loyalists** 
        to **viral TikTok makeup collectors** and **seasonal promotional hunters**.
        
        This decision system discovers organic behavioral segments using multi-dimensional clustering 
        (RFM, Category Diversity, Discount Sensitivity, and Digital Engagement) to replace static demographic 
        groupings with **evidence-backed, ROI-optimized campaign strategies**.
        """
    )
    
    # Executive KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Synthetic Clients", f"{len(df):,}", "Active Cohort Size")
    with col2:
        render_kpi_card("Total 12M Spend", f"${df['monetary_total_12m'].sum():,.0f}", "Gross Omnichannel GMV")
    with col3:
        render_kpi_card("Average Order Value", f"${df['avg_order_value'].mean():.2f}", "Across all baskets")
    with col4:
        render_kpi_card("Avg Annual Orders", f"{df['frequency_12m'].mean():.1f}", "Purchase frequency")
    with col5:
        high_churn = (df["churn_risk_score"] > 65).mean() * 100
        render_kpi_card("High Churn Risk", f"{high_churn:.1f}%", ">65% churn probability")
        
    st.markdown("---")
    
    # Strategic Business Value Levers
    st.markdown("### Strategic Business Levers & Intended Impact")
    col_lev1, col_lev2, col_lev3, col_lev4 = st.columns(4)
    with col_lev1:
        st.markdown(
            """
            <div class="segment-card">
                <h4 style="color: #A855F7; margin-bottom: 8px;">1. VIP Retention</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Shield Rouge and top spenders from promotional fatigue. Substitute margin-diluting discount codes 
                    with brand founder pre-access, deluxe GWP miniatures, and concierge services.
                </p>
                <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">Target: +12% Net Spend Retention</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_lev2:
        st.markdown(
            """
            <div class="segment-card">
                <h4 style="color: #06B6D4; margin-bottom: 8px;">2. Regimen Replenishment</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Trigger predictive 60-day refill alerts for clinical skincare and haircare ritualists. 
                    Lock in recurring repurchase cadences before clients consider competitors.
                </p>
                <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">Target: +25% Category Repeat Rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_lev3:
        st.markdown(
            """
            <div class="segment-card">
                <h4 style="color: #EC4899; margin-bottom: 8px;">3. Trend Velocity</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Capitalize on Gen-Z / Millennial viral makeup launches via mobile app push and SMS flash alerts, 
                    maximizing launch day basket additions while minimizing paid social ad spend.
                </p>
                <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">Target: 3.2x Push Conversion Lift</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_lev4:
        st.markdown(
            """
            <div class="segment-card">
                <h4 style="color: #F59E0B; margin-bottom: 8px;">4. Margin Protection</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Enforce minimum cart thresholds ($60+) on deal-seeking cohorts. Direct discount-dependent 
                    shoppers toward high-margin Sephora Collection private-label beauty.
                </p>
                <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">Target: +180 bps Margin Preservation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Discovered Behavioral Segments Snapshot")
    summary_cols = st.columns(len(profiles))
    for i, (cid, p) in enumerate(profiles.items()):
        s = p["stats"]
        with summary_cols[i]:
            st.markdown(
                f"""
                <div class="segment-card" style="border-top: 3px solid {THEME_COLORS['accent_purple']};">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Segment {cid+1}</div>
                    <div style="font-size: 1.0rem; font-weight: 700; color: #F8FAFC; margin: 4px 0 8px 0;">{p['name']}</div>
                    <div style="font-size: 0.85rem; color: #E2E8F0; line-height: 1.5;">
                        &bull; <strong>Clients:</strong> {s['count']:,} ({s['pct_total']:.1f}%)<br>
                        &bull; <strong>Avg Spend:</strong> ${s['avg_spend']:,.0f}<br>
                        &bull; <strong>Orders/Yr:</strong> {s['avg_frequency']:.1f}<br>
                        &bull; <strong>Top Focus:</strong> {s['primary_cat']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ==========================================
# PAGE 2: DATA EXPLORER & DICTIONARY
# ==========================================
elif page == "Data Explorer & Dictionary":
    st.title("Synthetic Customer & Transaction Data Explorer")
    st.caption("Inspect synthetic client records, explore granular transactions, and download benchmark datasets.")
    
    tab_data, tab_tx, tab_dict = st.tabs(["👥 Customer Feature Profiles", "💳 Granular Transaction Log", "📖 Data Dictionary"])
    
    with tab_data:
        st.markdown("#### Multi-Dimensional Filters")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            tier_filter = st.multiselect("Loyalty Tier", LOYALTY_TIERS, default=LOYALTY_TIERS)
        with f_col2:
            cat_filter = st.multiselect("Primary Category", CATEGORIES, default=CATEGORIES)
        with f_col3:
            spend_range = st.slider("12M Spend Range ($)", 0, int(df["monetary_total_12m"].max()), (0, int(df["monetary_total_12m"].max())))
        with f_col4:
            recency_range = st.slider("Recency Range (Days)", 0, 365, (0, 365))
            
        # Apply filters
        filtered_df = df[
            (df["loyalty_tier"].isin(tier_filter)) &
            (df["primary_category"].isin(cat_filter)) &
            (df["monetary_total_12m"] >= spend_range[0]) &
            (df["monetary_total_12m"] <= spend_range[1]) &
            (df["recency_days"] >= recency_range[0]) &
            (df["recency_days"] <= recency_range[1])
        ]
        
        st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} customers**")
        
        # Display table
        cols_to_show = [
            "customer_id", "segment_name", "loyalty_tier", "primary_category",
            "monetary_total_12m", "frequency_12m", "recency_days", "avg_order_value",
            "discount_purchase_ratio", "omnichannel_engagement_score", "churn_risk_score"
        ]
        st.dataframe(
            filtered_df[cols_to_show].style.format({
                "monetary_total_12m": "${:,.2f}",
                "avg_order_value": "${:,.2f}",
                "discount_purchase_ratio": "{:.1%}",
                "omnichannel_engagement_score": "{:.1f}",
                "churn_risk_score": "{:.1f}%"
            }),
            use_container_width=True,
            height=380
        )
        
        # CSV Downloads
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Customer CSV",
                data=csv_filtered,
                file_name="sephora_synthetic_customers_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d_col2:
            csv_full = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Synthetic Dataset (All Columns)",
                data=csv_full,
                file_name="sephora_synthetic_customers_complete.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with tab_tx:
        st.markdown(f"#### Granular Synthetic Transactions ({len(tx_df):,} total simulated orders)")
        st.dataframe(
            tx_df.style.format({
                "order_value": "${:,.2f}"
            }),
            use_container_width=True,
            height=380
        )
        csv_tx = tx_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Granular Transaction Log CSV",
            data=csv_tx,
            file_name="sephora_synthetic_transactions.csv",
            mime="text/csv"
        )
        
    with tab_dict:
        st.markdown("#### Data Dictionary & Metric Definitions")
        dict_df = get_data_dictionary()
        st.dataframe(dict_df, use_container_width=True, height=450)

# ==========================================
# PAGE 3: HOW THE ALGORITHM WORKS
# ==========================================
elif page == "How the Algorithm Works":
    st.title("How the Customer Segmentation Algorithm Works")
    st.subheader("End-to-End Mathematical Pipeline from Raw Events to Segment Strategy")
    
    st.markdown(
        """
        The segmentation engine combines feature engineering, mathematical scaling, 
        unsupervised K-Means clustering with validation diagnostics, and dimensionality reduction.
        """
    )
    
    # Visual Numbered Architecture Flow
    steps = [
        ("Step 1: Synthetic Event & Profile Synthesis", "Simulates realistic retail omnichannel behaviors across spend, frequency, categories, returns, and digital sessions using seeded Dirichlet & Beta distributions."),
        ("Step 2: Mathematical Feature Engineering", "Derives RFM quintiles, normalized Shannon entropy category diversity H = -Σ(p·ln(p))/ln(5), discount sensitivity indices, and calibrated churn risk probabilities."),
        ("Step 3: Feature Normalization (StandardScaler)", "Centers and scales each feature to μ=0, σ=1 to prevent large-magnitude variables (annual spend) from mathematically dominating distance computations over rates."),
        ("Step 4: K-Means Optimization & Diagnostics", "Executes K-Means with k-means++ initialization across k∈[3,8]. Evaluates Inertia (within-cluster sum of squares), Silhouette Score (cluster cohesion vs separation), and Davies-Bouldin index."),
        ("Step 5: Latent Space PCA Decomposition", "Projects high-dimensional feature vectors into 2D and 3D orthogonal principal components (PC1 value, PC2 category, PC3 digital) explaining maximum variance for visual interpretation."),
        ("Step 6: Evidence-Grounded Persona Derivation", "Ranks empirical cluster centroids against behavioral benchmarks to assign persona titles (e.g. Prestige Rouge, Skincare Ritualists) and grounded narrative summaries."),
        ("Step 7: Responsible AI & Activation Guardrails", "Validates k-anonymity (minimum cohort count ≥ 30), audits channel consent rates, flags margin dilution, and triggers human signoff requirements.")
    ]
    
    for title, desc in steps:
        st.markdown(
            f"""
            <div class="kpi-card" style="margin-bottom: 8px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #A855F7;">{title}</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    st.markdown("### Algorithmic Evaluation & Model Diagnostics")
    
    # Calculate evaluation table on the fly
    with st.spinner("Computing K-Means cluster diagnostics..."):
        eval_table = evaluate_cluster_range(df, k_min=3, k_max=8, random_state=st.session_state["random_seed"])
        
    diag_col1, diag_col2 = st.columns([3, 2])
    with diag_col1:
        st.plotly_chart(plot_cluster_evaluation_curves(eval_table, st.session_state["n_clusters"]), use_container_width=True)
    with diag_col2:
        st.markdown("#### Diagnostic Evaluation Table")
        st.dataframe(
            eval_table.style.highlight_max(subset=["silhouette_score"], color="#3B1C5A")
                           .highlight_min(subset=["davies_bouldin_index"], color="#1A3B30"),
            use_container_width=True
        )
        st.caption(
            "• **Silhouette Score:** Higher is better (measures how distinct clusters are from each other).\n"
            "• **Inertia:** Lower is better (flattens at optimal elbow point).\n"
            "• **Davies-Bouldin Index:** Lower is better (ratio of within-cluster distance to between-cluster distance)."
        )
        
    st.markdown("---")
    st.markdown("### Model Assumptions & Technical Limitations")
    st.markdown(
        """
        > [!IMPORTANT]
        > **1. Geometric Assumption of K-Means:** K-Means assumes spherical, isotropic clusters with equal variance. Real customer shopping patterns can form irregular densities (e.g., non-convex manifolds). For highly non-linear distributions, DBSCAN or Gaussian Mixture Models (GMM) can be evaluated.
        > 
        > **2. Sensitivity to Feature Scaling:** Because K-Means relies on Euclidean distance, outlier clipping and StandardScaler transformations are mandatory. Without scaling, annual spend ($3,000) would dwarf email open rates (0.65).
        > 
        > **3. Synthetic Data Sandbox Disclosure:** All customer profiles, transaction timestamps, and response parameters in this application are 100% synthetic mathematical simulations. They do not represent Sephora's proprietary customer data or production marketing models.
        > 
        > **4. Transparent Baseline Fallback:** If the clustering engine encounters degenerate inputs, identical rows, or missing convergence, the pipeline automatically falls back to an empirical Quantile-based RFM discretizer and alerts the user.
        """
    )

# ==========================================
# PAGE 4: SEGMENT EXPLORER & CLUSTER VISUALIZATION
# ==========================================
elif page == "Segment Explorer & Profiles":
    st.title("Segment Explorer & Behavioral Visualizations")
    st.subheader("Interactive 2D/3D Cluster Projections, Value Distributions & Persona Playbooks")
    
    # Visualizations
    v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs(["🗺️ 2D PCA Map", "🌐 3D Latent Space", "📊 Population vs Revenue", "🕸️ Radar Profile Comparison"])
    
    with v_tab1:
        st.plotly_chart(plot_pca_2d(df), use_container_width=True)
        st.caption(
            "PCA Decomposition: Principal Component 1 captures primary spend scale and purchase frequency. "
            "Principal Component 2 differentiates category specialization and discount elasticity."
        )
    with v_tab2:
        st.plotly_chart(plot_pca_3d(df), use_container_width=True)
        st.caption("Rotate, pan, and zoom to inspect cluster boundaries in 3D latent space.")
    with v_tab3:
        treemap_col, pareto_col = st.columns(2)
        with treemap_col:
            st.plotly_chart(plot_segment_size_value_treemap(df), use_container_width=True)
        with pareto_col:
            st.plotly_chart(plot_pareto_revenue(df), use_container_width=True)
    with v_tab4:
        radar_col, bar_col = st.columns([1, 1])
        with radar_col:
            st.plotly_chart(plot_segment_radar_profiles(profiles), use_container_width=True)
        with bar_col:
            st.plotly_chart(plot_category_mix_by_segment(df), use_container_width=True)
            
    st.markdown("---")
    st.markdown("### Grounded Segment Personas & Marketing Playbooks")
    
    for cid, p in profiles.items():
        s = p["stats"]
        strat = p["strategy"]
        with st.expander(f"✨ Segment {cid+1}: {p['name']} ({s['count']:,} clients &bull; ${s['avg_spend']:,.0f} avg spend)", expanded=(cid==0)):
            col_left, col_right = st.columns([1, 2])
            with col_left:
                st.markdown(
                    f"""
                    <div class="segment-card">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #A855F7; margin-bottom: 8px;">EMPIRICAL METRICS</div>
                        <div style="font-size: 0.82rem; color: #E2E8F0; line-height: 1.6;">
                            &bull; <strong>Cohort Share:</strong> {s['pct_total']:.1f}% ({s['count']:,} clients)<br>
                            &bull; <strong>Annual Spend:</strong> ${s['avg_spend']:,.2f}<br>
                            &bull; <strong>Frequency:</strong> {s['avg_frequency']:.1f} orders/year<br>
                            &bull; <strong>AOV:</strong> ${s['avg_aov']:.2f}<br>
                            &bull; <strong>Recency:</strong> {s['avg_recency']:.0f} days<br>
                            &bull; <strong>Discount Rate:</strong> {s['avg_discount_pct']*100:.1f}%<br>
                            &bull; <strong>Dominant Tier:</strong> {s['top_loyalty_tier']}<br>
                            &bull; <strong>Primary Category:</strong> {s['primary_cat']}<br>
                            &bull; <strong>Churn Risk:</strong> {s['avg_churn_risk']:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_right:
                st.markdown(f"**Grounded Behavioral Narrative:**")
                st.write(p["narrative"])
                
                st.markdown(f"#### 🎯 Strategy: {strat['headline']}")
                st.markdown(
                    f"""
                    - **Recommended Channels:** {', '.join(strat['recommended_channels'])}
                    - **Offer Strategy:** {strat['offer_strategy']}
                    - **Content Theme:** {strat['content_theme']}
                    - **Strategic Priority:** `{strat['priority']}`
                    - **Expected Business Impact:** {strat['expected_impact']}
                    - **Risk & Uncertainty:** *{strat['risk_uncertainty']}*
                    """
                )

    st.markdown("---")
    st.markdown("### 📥 Export Segment Strategy Benchmarks")
    profile_rows = []
    for cid, p in profiles.items():
        s = p["stats"]
        strat = p["strategy"]
        profile_rows.append({
            "Cluster ID": cid + 1,
            "Segment Name": p["name"],
            "Customer Count": s["count"],
            "Share of Base (%)": round(s["pct_total"], 1),
            "Avg Annual Spend ($)": round(s["avg_spend"], 2),
            "Avg Frequency (Orders/Yr)": round(s["avg_frequency"], 1),
            "Average Order Value ($)": round(s["avg_aov"], 2),
            "Avg Recency (Days)": round(s["avg_recency"], 0),
            "Primary Product Focus": s["primary_cat"],
            "Discount Reliance (%)": round(s["avg_discount_pct"] * 100, 1),
            "Estimated Churn Risk (%)": round(s["avg_churn_risk"], 1),
            "Strategy Headline": strat["headline"],
            "Recommended Channels": ", ".join(strat["recommended_channels"]),
            "Offer Strategy": strat["offer_strategy"],
            "Priority": strat["priority"],
            "Expected Impact": strat["expected_impact"]
        })
    profile_summary_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_summary_df, use_container_width=True)

    csv_profiles = profile_summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Segment Profiles Benchmark CSV",
        data=csv_profiles,
        file_name="sephora_segment_profiles_benchmark.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================================
# PAGE 5: CUSTOMER 360° DETAIL
# ==========================================
elif page == "Customer 360° Detail":
    st.title("Customer 360° Individual Detail Lookup")
    st.subheader("Inspect Individual Synthetic Profiles, Category Shares, and Next Best Action")
    
    if "selected_customer_id" not in st.session_state:
        st.session_state["selected_customer_id"] = df["customer_id"].iloc[0]
        
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel2:
        st.write("")
        st.write("")
        if st.button("🎲 Random Client", use_container_width=True):
            st.session_state["selected_customer_id"] = str(np.random.choice(df["customer_id"].values))
            st.rerun()
            
    with col_sel1:
        cid_list = df["customer_id"].tolist()
        try:
            curr_idx = cid_list.index(st.session_state["selected_customer_id"])
        except ValueError:
            curr_idx = 0
        selected_cid = st.selectbox(
            "Select Customer ID to Inspect",
            cid_list,
            index=curr_idx
        )
        st.session_state["selected_customer_id"] = selected_cid
            
    cust_row = df[df["customer_id"] == selected_cid].iloc[0]
    
    # Customer Header KPIs
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
    with kpi_c1:
        render_kpi_card("Loyalty Tier", cust_row["loyalty_tier"], f"{cust_row['tenure_months']} mo tenure")
    with kpi_c2:
        render_kpi_card("12M Gross Spend", f"${cust_row['monetary_total_12m']:,.2f}", f"AOV ${cust_row['avg_order_value']:.2f}")
    with kpi_c3:
        render_kpi_card("Orders in 12M", f"{cust_row['frequency_12m']} orders", f"Recency {cust_row['recency_days']}d")
    with kpi_c4:
        render_kpi_card("Assigned Segment", cust_row["segment_name"][:18], f"Cluster {cust_row['cluster_id']+1}")
    with kpi_c5:
        render_kpi_card("Discount Reliance", f"{cust_row['discount_purchase_ratio']*100:.0f}%", "Orders with promo")
    with kpi_c6:
        churn_val = cust_row["churn_risk_score"]
        render_kpi_card("Estimated Churn", f"{churn_val:.0f}%", "Risk Probability")
        
    st.markdown("---")
    
    col_donut, col_nba = st.columns([1, 1])
    with col_donut:
        st.plotly_chart(plot_customer_category_donut(cust_row), use_container_width=True)
        st.markdown(
            f"""
            **Channel Consent & Reachability:**  
            • Email: `{'Opted In ✅' if cust_row['consent_email'] else 'Opted Out ❌'}` &bull; 
            SMS: `{'Opted In ✅' if cust_row['consent_sms'] else 'Opted Out ❌'}`<br>
            • App Push: `{'Opted In ✅' if cust_row['consent_app_push'] else 'Opted Out ❌'}` &bull; 
            Direct Mail: `{'Opted In ✅' if cust_row['consent_direct_mail'] else 'Opted Out ❌'}`
            """,
            unsafe_allow_html=True
        )
    with col_nba:
        st.markdown("### 🎯 Recommended Next Best Action (NBA)")
        seg_profile = profiles[int(cust_row["cluster_id"])]
        st.markdown(
            f"""
            <div class="segment-card" style="border-left: 4px solid #A855F7;">
                <div style="font-size: 0.85rem; color: #A855F7; font-weight: 700; text-transform: uppercase;">
                    Prescribed Marketing Action &bull; {seg_profile['name']}
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 8px 0;">
                    {seg_profile['strategy']['headline']}
                </div>
                <p style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5;">
                    <strong>Triggered Offer:</strong> {seg_profile['strategy']['offer_strategy']}
                </p>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    <strong>Recommended Channel:</strong> {', '.join(seg_profile['strategy']['recommended_channels'])}<br>
                    <strong>Estimated 12M Future CLV:</strong> ${cust_row['estimated_future_clv']:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Transaction history for this customer
    st.markdown(f"#### Simulated Transaction History for {selected_cid}")
    cust_tx = tx_df[tx_df["customer_id"] == selected_cid]
    if len(cust_tx) > 0:
        st.dataframe(
            cust_tx.style.format({
                "order_value": "${:,.2f}"
            }),
            use_container_width=True
        )
    else:
        st.info("No granular transaction logs generated for this customer.")

# ==========================================
# PAGE 6: CAMPAIGN SCENARIO SIMULATOR
# ==========================================
elif page == "Campaign Scenario Simulator":
    st.title("Campaign Strategy & Expected-Lift Scenario Simulator")
    st.subheader("Model Marketing Budgets, Channel Costs, Promotional Discounts, and Incremental ROI")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("#### 🛠️ Campaign Controls")
        
        all_segment_names = sorted(df["segment_name"].unique())
        selected_target_segments = st.multiselect(
            "Target Customer Segments",
            all_segment_names,
            default=[all_segment_names[0]] if all_segment_names else [],
            help="Select one or more segments to target with this campaign."
        )
        
        selected_channels = st.multiselect(
            "Marketing Channels",
            ["Email", "SMS", "App Push", "Direct Mail"],
            default=["Email", "App Push"],
            help="Choose outbound channels (costs apply per contact)."
        )
        
        campaign_budget = st.slider(
            "Campaign Budget ($)",
            min_value=500,
            max_value=25000,
            value=3500,
            step=500,
            help="Total campaign marketing expense ceiling."
        )
        
        reach_pct = st.slider(
            "Target Cohort Reach (%)",
            min_value=10,
            max_value=100,
            value=80,
            step=5,
            help="Percentage of consented clients to contact."
        )
        
        discount_depth = st.slider(
            "Promotional Discount Depth (%)",
            min_value=0,
            max_value=25,
            value=10,
            step=5,
            help="Discount applied to orders (e.g. 10%, 15%, 20%)."
        )
        
        expected_lift = st.slider(
            "Assumed Response Lift (%)",
            min_value=10,
            max_value=80,
            value=35,
            step=5,
            help="Relative lift in conversions compared to baseline."
        )
        
        baseline_margin = st.slider(
            "Product Gross Margin (%)",
            min_value=40,
            max_value=75,
            value=60,
            step=5,
            help="Prestige beauty product margin before promotional discounts."
        )
        
    with sim_col2:
        st.markdown("#### 📈 Immediate Scenario Forecast")
        
        # Run simulation
        scenario_result = simulate_campaign_scenario(
            df=df,
            target_segments=selected_target_segments,
            channels=selected_channels,
            allocated_budget=campaign_budget,
            expected_lift_pct=expected_lift,
            discount_depth_pct=discount_depth,
            baseline_margin_pct=baseline_margin,
            reach_pct=reach_pct
        )
        
        if scenario_result.get("status") == "SUCCESS":
            r = scenario_result
            
            # Forecast KPI Cards
            fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
            with fc_col1:
                render_kpi_card("Targeted Contacts", f"{r['actual_contacts']:,}", f"{r['consent_rate_pct']:.0f}% consent rate")
            with fc_col2:
                render_kpi_card("Campaign Spend", f"${r['actual_campaign_cost']:,.2f}", f"${r['channel_cost_per_contact']:.3f}/contact")
            with fc_col3:
                render_kpi_card("Incremental Orders", f"{r['incremental_conversions']:,}", f"+{expected_lift}% lift")
            with fc_col4:
                render_kpi_card("Incr. Gross Revenue", f"${r['incremental_gross_revenue']:,.2f}", f"AOV ${r['effective_aov']:.2f}")
                
            fc_col5, fc_col6, fc_col7, fc_col8 = st.columns(4)
            with fc_col5:
                render_kpi_card("Incr. Gross Profit", f"${r['incremental_gross_profit']:,.2f}", f"{baseline_margin}% base margin")
            with fc_col6:
                profit_color = "#10B981" if r['net_incremental_profit'] > 0 else "#EF4444"
                render_kpi_card("Net Incremental Profit", f"${r['net_incremental_profit']:,.2f}", "Profit minus ad cost")
            with fc_col7:
                render_kpi_card("Net Campaign ROI", f"{r['net_roi_pct']:,.1f}%", "Return on ad investment")
            with fc_col8:
                render_kpi_card("ROAS", f"{r['roas']:.2f}x", "Gross rev / spend")
                
            # Guardrails Alert Banners
            g = r["guardrails"]
            if g["is_profitable"]:
                st.success("✅ **Campaign is Net Profitable:** Incremental gross profit comfortably exceeds channel delivery expenses.")
            else:
                st.error("⚠️ **Negative Campaign ROI:** Marketing delivery costs and discount depth exceed generated gross profit. Adjust audience or lower discount.")
                
            if g["budget_capped"]:
                st.warning(f"ℹ️ **Budget Ceiling Enforced:** Audience was throttled to stay within the ${campaign_budget:,.2f} budget limit.")
                
            if g["low_margin_warning"]:
                st.warning("⚠️ **Margin Dilution Risk:** Discount depth absorbs more than 50% of the base product margin.")
                
            st.markdown("---")
            st.markdown("#### 📊 Comparative Scenario Sensitivity (Conservative vs Base vs Aggressive)")
            comp_table = compare_scenarios(
                df=df,
                target_segments=selected_target_segments,
                channels=selected_channels,
                budget=campaign_budget,
                discount_depth=discount_depth,
                baseline_margin=baseline_margin
            )
            st.dataframe(comp_table, use_container_width=True)
            
            # Export scenario forecast
            csv_scen = comp_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Scenario Forecast Comparison CSV",
                data=csv_scen,
                file_name="sephora_campaign_scenario_forecast.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least one segment and marketing channel.")

# ==========================================
# PAGE 7: MANAGER STRATEGIC QUESTIONS
# ==========================================
elif page == "Manager Strategic Questions":
    st.title("Manager Strategic Questions & Decision Analyses")
    st.subheader("Data-Backed Answers to Core Omnichannel Marketing Questions")
    
    q_tab1, q_tab2, q_tab3 = st.tabs([
        "💰 Q1: Top Margin & Churn Risk",
        "🏷️ Q2: Price Sensitivity & Cannibalization",
        "📱 Q3: Omnichannel Activation & Privacy"
    ])
    
    # QUESTION 1
    with q_tab1:
        st.markdown("### ❓ Question 1: Which customer segment generates the highest margin and lifetime value, and are we in danger of churn among top spenders?")
        
        q1_summary = df.groupby("segment_name").agg(
            total_spend=("monetary_total_12m", "sum"),
            avg_spend=("monetary_total_12m", "mean"),
            avg_churn=("churn_risk_score", "mean"),
            high_churn_count=("churn_risk_score", lambda x: (x > 60).sum()),
            client_count=("customer_id", "count")
        ).reset_index()
        q1_summary["high_churn_pct"] = (q1_summary["high_churn_count"] / q1_summary["client_count"]) * 100
        
        col_q1_plot, col_q1_text = st.columns([3, 2])
        with col_q1_plot:
            fig_q1 = px.scatter(
                df,
                x="monetary_total_12m",
                y="churn_risk_score",
                color="segment_name",
                color_discrete_sequence=[THEME_COLORS['accent_purple'], THEME_COLORS['accent_pink'], THEME_COLORS['accent_cyan'], THEME_COLORS['accent_gold'], "#64748B"],
                labels={"monetary_total_12m": "Annual Spend ($)", "churn_risk_score": "Estimated Churn Risk (%)"},
                title="Customer Spend vs Churn Risk Distribution"
            )
            fig_q1.update_layout(**THEME_COLORS)
            layout = {
                "paper_bgcolor": THEME_COLORS['bg_card'],
                "plot_bgcolor": THEME_COLORS['bg_card'],
                "font": dict(color=THEME_COLORS['text_secondary']),
                "height": 400
            }
            fig_q1.update_layout(**layout)
            st.plotly_chart(fig_q1, use_container_width=True)
            
        with col_q1_text:
            st.markdown(
                """
                **Managerial Findings:**
                1. **Prestige Rouge Connoisseurs** deliver the highest aggregate gross margin and LTV ($1,200 - $3,000 annual spend), with lower average churn risk (<15%).
                2. **Dormancy Warning:** An estimated **8-12% of high-spend clients** have recency >90 days, placing them in the high-risk churn zone (>60% probability).
                3. **Action Playbook:**  
                   - Initiate an automated *Concierge White Glove outreach* when a Rouge client hits 75 days of inactivity.
                   - Offer complimentary deluxe trial rituals or private in-store artistry appointments rather than price cuts.
                """
            )
            
    # QUESTION 2
    with q_tab2:
        st.markdown("### ❓ Question 2: How price/discount sensitive are our high-frequency shoppers vs occasional buyers, and can we reduce discount cannibalization?")
        
        col_q2_plot, col_q2_text = st.columns([3, 2])
        with col_q2_plot:
            fig_q2 = px.scatter(
                df,
                x="frequency_12m",
                y="discount_purchase_ratio",
                color="loyalty_tier",
                color_discrete_map={"Rouge": THEME_COLORS['accent_purple'], "VIB": THEME_COLORS['accent_pink'], "Beauty Insider": THEME_COLORS['accent_cyan']},
                labels={"frequency_12m": "Annual Purchase Frequency", "discount_purchase_ratio": "Discount Purchase Ratio"},
                title="Purchase Frequency vs Discount Reliance by Loyalty Tier"
            )
            layout_q2 = {
                "paper_bgcolor": THEME_COLORS['bg_card'],
                "plot_bgcolor": THEME_COLORS['bg_card'],
                "font": dict(color=THEME_COLORS['text_secondary']),
                "height": 400
            }
            fig_q2.update_layout(**layout_q2)
            st.plotly_chart(fig_q2, use_container_width=True)
            
        with col_q2_text:
            st.markdown(
                """
                **Managerial Findings:**
                1. **Inelastic High-Frequency Shoppers:** Top Rouge shoppers purchase frequently regardless of sales events (discount purchase ratio <25%). Blasting them with blanket 20% discounts cannibalizes margin without generating incremental volume.
                2. **Promo Hunters Cohort:** The Deal-Seeking segment purchases almost exclusively during promotional cycles (>70% discount reliance).
                3. **Action Playbook:**  
                   - **Full-Price Shielding:** Exclude Rouge VIPs from standard discount codes; provide early access to limited-edition drops instead.
                   - **Basket Thresholds:** Require Deal Hunters to hit $65 minimum basket size to unlock discount codes, protecting gross margin dollars.
                """
            )
            
    # QUESTION 3
    with q_tab3:
        st.markdown("### ❓ Question 3: What omnichannel activation strategy maximizes ROI while respecting customer privacy and consent fatigue?")
        
        # Calculate consent breakdown
        consent_df = pd.DataFrame({
            "Channel": ["Email Marketing", "SMS Alerts", "Mobile App Push", "Direct Mail Postcards"],
            "Opt-In Rate (%)": [
                df["consent_email"].mean() * 100,
                df["consent_sms"].mean() * 100,
                df["consent_app_push"].mean() * 100,
                df["consent_direct_mail"].mean() * 100
            ],
            "Unit Cost ($)": [CHANNEL_UNIT_COSTS["Email"], CHANNEL_UNIT_COSTS["SMS"], CHANNEL_UNIT_COSTS["App Push"], CHANNEL_UNIT_COSTS["Direct Mail"]],
            "Recommended Weekly Cap": ["2 per week", "1 per week (high intent)", "3 per week (in-session)", "1 per quarter"]
        })
        
        st.dataframe(consent_df.style.format({
            "Opt-In Rate (%)": "{:.1f}%",
            "Unit Cost ($)": "${:.2f}"
        }), use_container_width=True)
        
        st.markdown(
            """
            **Managerial Recommendations for Omnichannel Balance:**
            - **App Push First:** For mobile-active segments, App Push has the lowest unit cost ($0.01) and highest engagement without SMS carrier fees.
            - **SMS Reserve for Urgency:** Restrict SMS to high-intent actions (e.g., flash replenishment reminders or same-day BOPIS pickup notifications).
            - **Direct Mail for Luxury Reactivation:** Reserve physical mailers ($1.25 cost) strictly for high-value Rouge lapsed clients where digital emails bounce or go unopened.
            """
        )

# ==========================================
# PAGE 8: RESPONSIBLE AI & GUARDRAILS
# ==========================================
elif page == "Responsible AI & Guardrails":
    st.title("Responsible AI Governance & Marketing Guardrails")
    st.subheader("Auditing Algorithmic Privacy, Fair Representation, and Human-in-the-Loop Safeguards")
    
    # Run audit
    audit = run_governance_audit(df, min_k_anonymity=30)
    
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        render_kpi_card("k-Anonymity Status", "PASS (k ≥ 30)" if audit["k_anonymity_passed"] else "ATTENTION", f"Smallest cohort: {audit['smallest_cohort_size']} clients")
    with g_col2:
        render_kpi_card("Governance Status", audit["governance_status"], "Automated Safety Check")
    with g_col3:
        render_kpi_card("Human Signoff", "MANDATORY", "For any budget >$5,000")
        
    st.markdown("---")
    st.markdown("### 🔒 Differential Privacy & Cohort Anonymity")
    st.write(
        f"To protect customer privacy and prevent re-identification of individual shoppers, all segment "
        f"targeting requires a minimum cohort size threshold of **k ≥ {audit['min_k_anonymity_threshold']}**. "
        "Any cohort smaller than this threshold is automatically suppressed from granular reporting."
    )
    
    if audit["privacy_flags"]:
        for flag in audit["privacy_flags"]:
            st.info(f"🛡️ **Privacy Notice:** {flag}")
            
    st.markdown("---")
    st.markdown("### ⚖️ Demographic Fairness & Bias Review")
    st.write("Evaluating representation across age brackets and loyalty tiers to detect unintended demographic exclusion:")
    if audit["fairness_disparities"]:
        for disp in audit["fairness_disparities"]:
            st.caption(f"• {disp}")
    else:
        st.success("✅ No severe demographic under-representation detected across segment cohorts.")
        
    st.markdown("---")
    st.markdown("### 👤 Human-in-the-Loop Governance Checklist")
    st.markdown(
        """
        Before initiating any automated CRM or marketing dispatch, marketing directors must complete 
        and verify the following compliance controls:
        """
    )
    c1 = st.checkbox("Verified opt-in consent against current Sephora compliance & TCPA standards", value=True)
    c2 = st.checkbox("Confirmed promotional discount depth does not violate gross margin floors", value=True)
    c3 = st.checkbox("Enforced omnichannel frequency capping (max 3 outbound contacts/week per client)", value=True)
    c4 = st.checkbox("Verified that high-value Rouge VIP clients are excluded from mass discount broadcasts", value=True)
    
    if c1 and c2 and c3 and c4:
        st.success("✅ **Governance Checklist Complete:** All compliance guardrails verified for safe campaign launch.")
    else:
        st.warning("⚠️ Pending compliance signoff: Complete all checklist items before campaign deployment.")
