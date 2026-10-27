# 💄 Sephora AI Customer Segmentation & Marketing Intelligence System

> **DISCLAIMER:**  
> **SYNTHETIC DEMO DATA — NOT REAL COMPANY DATA.**  
> This application is an illustrative prototype engineered for Sephora marketing architectures. All client records, order values, product mixes, discount behaviors, and response distributions are 100% synthetically simulated. It does not access, scrape, or represent Sephora's proprietary systems, private consumer data, or production models.

---

## 📌 Executive Summary & Business Problem

In omnichannel prestige beauty retail, generic mass-market discount blasts cause severe gross margin dilution and induce marketing unsubscribe fatigue among top-spending loyalists. Sephora’s customer base exhibits stark behavioral heterogeneity—ranging from high-spending **Rouge VIP clinical skincare aficionados** to **viral trend makeup shoppers** and **deal-seeking seasonal promotional hunters**.

This production-grade Streamlit application discovers natural behavioral customer segments by combining multi-dimensional feature engineering (RFM, normalized Shannon entropy category diversity, promotional discount sensitivity, and omnichannel engagement) with unsupervised **K-Means clustering** and **Principal Component Analysis (PCA)**. 

Marketing directors and CRM analysts can explore data-grounded personas, simulate campaign lift scenarios against custom budgets with real-time ROI/ROAS feedback, and enforce responsible AI guardrails (k-anonymity, consent verification, and margin safety).

---

## 🚀 Quickstart Guide (Windows-Friendly)

Follow these exact steps in Windows PowerShell or Command Prompt to launch the application:

### Step 1: Open Terminal & Navigate to Project Directory
```powershell
cd C:\Users\DELL\.gemini\antigravity\scratch\sephora_customer_segmentation
```

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
# Create virtual environment named .venv
python -m venv .venv

# Activate the virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1
# (Or in standard Command Prompt: .\.venv\Scripts\activate.bat)
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Interactive Web Application
```powershell
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 🏗️ Project Architecture & File Structure

```text
sephora_customer_segmentation/
│
├── app.py                      # Main Streamlit web application & multi-page dashboard
├── requirements.txt            # Python dependencies (streamlit, pandas, numpy, scikit-learn, plotly)
├── README.md                   # Comprehensive system documentation & setup guide
├── test_modules.py             # Quick backend smoke tests
├── test_app_and_pipeline.py    # Unit test suite covering data, features, clustering, scenarios & UI
│
└── src/                        # Core algorithmic modules
    ├── __init__.py             # Python package marker
    ├── synthetic_data.py       # Fictional customer & transaction generator + data dictionary
    ├── features.py             # RFM, Shannon entropy, discount sensitivity, engagement & churn metrics
    ├── clustering.py           # StandardScaler, K-Means++, Silhouette/Inertia diagnostics, PCA 2D/3D & personas
    ├── scenarios.py            # Campaign economics simulator, channel unit costs, budget & lift modeling
    ├── guardrails.py           # Responsible AI audits (k-anonymity >= 30, fairness, consent verification)
    └── ui_components.py        # Dark luxury Sephora theme CSS, KPI cards, banners & Plotly charts
```

---

## 🧠 Algorithmic Pipeline & Technical Workflow

The mathematical pipeline executes real computations without mock or static data:

1. **Synthetic Data Synthesis (`src/synthetic_data.py`):**
   - Simulates customer cohorts across 5 distinct behavioral archetypes with realistic continuous Dirichlet and Beta noise.
   - Generates 25+ attributes: 12-month spend, order frequencies, category distributions (*Skincare, Makeup, Fragrance, Haircare, Bath & Body*), discount reliance, return rates, email open/click rates, mobile app sessions, and channel opt-in consents.

2. **Feature Engineering (`src/features.py`):**
   - **RFM Composite Scores:** Quintile-ranked Recency, Frequency, and Monetary scores combined into a weighted index.
   - **Category Diversity (Normalized Shannon Entropy):**
     $$H_{\text{norm}} = -\frac{\sum_{c=1}^{5} p_c \ln(p_c)}{\ln(5)}$$
     Distinguishes single-category specialists ($H \to 0$) from broad-basket luxury explorers ($H \to 1$).
   - **Discount Sensitivity Index:** Blends discount order share and average promotional depth into a 0–100 scale.
   - **Omnichannel Engagement Score:** Integrates mobile app activity, email open rates, and click-through rates.
   - **Calibrated Churn Risk Probability:** Logit-based churn model flagging inactive or declining shoppers.

3. **Clustering & Dimensionality Reduction (`src/clustering.py`):**
   - Applies `StandardScaler` to prevent high-magnitude spend figures from dominating rates.
   - Runs `KMeans(n_clusters=k, init='k-means++', n_init=10)`.
   - Evaluates Inertia (elbow curve), Silhouette score, and Davies-Bouldin index across $k \in [3, 8]$.
   - Computes 2D and 3D PCA coordinate mappings for cluster visual boundary exploration.

4. **Evidence-Grounded Persona Derivation:**
   - Sorts empirical centroids to dynamically assign persona titles:
     - **Prestige Rouge Connoisseurs:** Top spenders ($1,200+), high frequency, low discount reliance.
     - **Skincare Ritualists:** Dominant skincare share (>45%), high AOV, high replenishment loyalty.
     - **Glam & Trend Makeup Seekers:** Dominant makeup share (>45%), mobile app active, viral launch buyers.
     - **Deal-Seeking Promo Hunters:** High discount usage (>60%), high return rate, low full-price conversion.
     - **Lapsed Occasional Shoppers:** High recency (>150 days), low engagement, high churn risk.

5. **Campaign Simulation & ROI Modeler (`src/scenarios.py`):**
   - Outbound channel unit costs: Email ($0.02), SMS ($0.08), Mobile Push ($0.01), Direct Mail ($1.25).
   - Simulates baseline vs. incremental conversions, promotional margin dilution, net incremental profit, ROI %, and Return on Ad Spend (ROAS).

6. **Responsible AI Guardrails (`src/guardrails.py`):**
   - **k-Anonymity Verification:** Checks minimum cohort size ($k \ge 30$) to prevent individual shopper re-identification.
   - **Demographic Balance Audit:** Inspects age group distributions across segments for systematic exclusion.
   - **Opt-in Consent Compliance:** Filters non-consenting users by channel (TCPA / GDPR compliance).
   - **Human Signoff Checklist:** Mandates human review for high-impact promotional deployments.

---

## 📊 Application Pages Overview

1. **Overview & Business Value:** Executive problem statement, Sephora omnichannel context, high-level KPIs, and strategic value levers.
2. **Data Explorer & Dictionary:** Multi-dimensional filters (Loyalty Tier, Category, Spend, Recency), customer tables, granular transaction logs, data dictionary, and 1-click CSV export.
3. **How the Algorithm Works:** Connected 7-step visual architecture, live K-Means diagnostic evaluation curves (Inertia & Silhouette across $k \in [3, 8]$), and explicit model limitations disclosures.
4. **Segment Explorer & Profiles:** Interactive 2D/3D PCA scatter plots, Population vs. Revenue Treemap, Pareto/Lorenz concentration curve, behavioral radar profiles, and tailored marketing playbooks.
5. **Customer 360° Detail:** Individual client lookup by ID, category spend donut breakdown, channel reachability status, transaction history, and tailored Next Best Action (NBA).
6. **Campaign Scenario Simulator:** Interactive marketing budget allocator, channel selection, discount depth slider, expected-lift scenario generator, immediate net profit/ROI calculation, and comparative scenario matrix.
7. **Manager Strategic Questions:** Three deep-dive strategic inquiries:
   - *Q1: Which customer segment generates the highest margin, and are top spenders at risk of churn?*
   - *Q2: How price-sensitive are high-frequency vs. occasional shoppers, and how can we eliminate discount cannibalization?*
   - *Q3: What omnichannel activation strategy maximizes ROI while respecting privacy and fatigue?*
8. **Responsible AI & Guardrails:** Algorithmic governance audit, differential privacy k-anonymity compliance, demographic representation check, and human-in-the-loop signoff checklist.

---

## 🧪 Testing Report & Validation

The codebase was validated via automated tests and live server probes:

1. **Unit Test Suite (`test_app_and_pipeline.py`):**
   - **9 out of 9 tests passed** in 4.04s (`Ran 9 tests in 4.042s - OK`).
   - Validated: Synthetic data generation, feature engineering bounds, Shannon entropy normalization, K-Means clustering stability, diagnostic curve computation, campaign scenario economics, governance audit checks, Plotly chart builders, and `app.py` syntax compilation.
2. **End-to-End Server Launch Test:**
   - Streamlit was launched headlessly and probed with an automated HTTP request.
   - Result: **HTTP 200 OK** response received successfully.
3. **Guardrail Validations:**
   - Tested k-anonymity thresholds ($k \ge 30$).
   - Tested budget capping logic when audience size exceeds budget limits.
   - Tested margin dilution warnings when discount depth exceeds 50% of gross margin.

---

## ⚠️ Limitations & Methodological Disclosures

- **Unsupervised Cluster Shapes:** K-Means assumes isotropic, convex clusters. Real human retail purchase dynamics may exhibit irregular or non-linear density shapes.
- **Synthetic Data Sandbox:** Generated cohorts reflect stylized retail distributions; findings should not be interpreted as empirical measurements of Sephora's actual customer base.
- **Human-in-the-Loop Requirement:** This system acts as a decision-support copilot. High-impact automated campaign dispatching and promotional budget commitments require manual human approval.
