"""
Comprehensive Test Suite for Sephora Customer Segmentation Prototype
Validates data generation, feature engineering, clustering stability,
scenario calculations, guardrails, and UI components.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

from src.synthetic_data import generate_synthetic_data, get_data_dictionary
from src.features import compute_engineered_features, calculate_shannon_entropy
from src.clustering import perform_clustering_pipeline, evaluate_cluster_range
from src.scenarios import simulate_campaign_scenario, compare_scenarios
from src.guardrails import run_governance_audit
from src.ui_components import (
    plot_pca_2d,
    plot_pca_3d,
    plot_segment_size_value_treemap,
    plot_pareto_revenue,
    plot_segment_radar_profiles,
    plot_category_mix_by_segment,
    plot_customer_category_donut,
    plot_cluster_evaluation_curves
)

class TestSephoraSegmentation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n--- Running Sephora AI Segmentation Tests ---")
        cls.raw_df, cls.tx_df = generate_synthetic_data(n_customers=600, random_seed=42)
        cls.eng_df = compute_engineered_features(cls.raw_df)
        cls.clustered_df, cls.meta = perform_clustering_pipeline(cls.eng_df, n_clusters=5, random_state=42)
        
    def test_synthetic_data_generation(self):
        self.assertEqual(len(self.raw_df), 600)
        self.assertGreater(len(self.tx_df), 1000)
        self.assertTrue(self.raw_df["is_synthetic"].all())
        self.assertIn("customer_id", self.raw_df.columns)
        self.assertIn("loyalty_tier", self.raw_df.columns)
        self.assertIn("monetary_total_12m", self.raw_df.columns)
        
    def test_data_dictionary(self):
        dict_df = get_data_dictionary()
        self.assertGreater(len(dict_df), 20)
        self.assertIn("Column", dict_df.columns)
        self.assertIn("Description", dict_df.columns)
        
    def test_feature_engineering(self):
        self.assertIn("category_entropy", self.eng_df.columns)
        self.assertIn("rfm_composite", self.eng_df.columns)
        self.assertIn("discount_sensitivity_score", self.eng_df.columns)
        self.assertIn("omnichannel_engagement_score", self.eng_df.columns)
        self.assertIn("churn_risk_score", self.eng_df.columns)
        self.assertIn("estimated_future_clv", self.eng_df.columns)
        
        # Verify Shannon entropy bounds [0, 1]
        self.assertTrue((self.eng_df["category_entropy"] >= 0.0).all())
        self.assertTrue((self.eng_df["category_entropy"] <= 1.0).all())
        
        # Verify churn risk is a percentage [0, 100]
        self.assertTrue((self.eng_df["churn_risk_score"] >= 0.0).all())
        self.assertTrue((self.eng_df["churn_risk_score"] <= 100.0).all())
        
    def test_clustering_pipeline(self):
        self.assertIn("cluster_id", self.clustered_df.columns)
        self.assertIn("segment_name", self.clustered_df.columns)
        self.assertIn("pca_1", self.clustered_df.columns)
        self.assertIn("pca_2", self.clustered_df.columns)
        self.assertIn("pca_3", self.clustered_df.columns)
        
        self.assertEqual(len(self.clustered_df["cluster_id"].unique()), 5)
        self.assertGreater(self.meta["silhouette_score"], 0.20)
        self.assertGreater(self.meta["pca_variance"]["total_explained"], 40.0)
        
    def test_cluster_range_evaluation(self):
        eval_df = evaluate_cluster_range(self.eng_df, k_min=3, k_max=6, random_state=42)
        self.assertEqual(len(eval_df), 4)
        self.assertIn("silhouette_score", eval_df.columns)
        self.assertIn("inertia", eval_df.columns)
        self.assertIn("davies_bouldin_index", eval_df.columns)
        
    def test_campaign_scenarios(self):
        target_name = list(self.meta["profiles"].values())[0]["name"]
        
        # Normal campaign
        res = simulate_campaign_scenario(
            df=self.clustered_df,
            target_segments=[target_name],
            channels=["Email", "App Push"],
            allocated_budget=5000,
            expected_lift_pct=35.0,
            discount_depth_pct=10.0,
            baseline_margin_pct=60.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["actual_contacts"], 0)
        self.assertGreater(res["incremental_conversions"], 0)
        self.assertIn("net_roi_pct", res)
        
        # Compare scenarios
        comp_df = compare_scenarios(
            df=self.clustered_df,
            target_segments=[target_name],
            channels=["Email", "App Push"],
            budget=5000,
            discount_depth=10.0
        )
        self.assertEqual(len(comp_df), 3)
        
    def test_governance_guardrails(self):
        audit = run_governance_audit(self.clustered_df, min_k_anonymity=30)
        self.assertIn("k_anonymity_passed", audit)
        self.assertIn("governance_status", audit)
        self.assertIn("consent_metrics", audit)
        self.assertTrue(audit["consent_metrics"]["email_opt_in_pct"] > 0)
        
    def test_plotly_chart_builders(self):
        fig2d = plot_pca_2d(self.clustered_df)
        self.assertIsNotNone(fig2d)
        
        fig3d = plot_pca_3d(self.clustered_df)
        self.assertIsNotNone(fig3d)
        
        fig_treemap = plot_segment_size_value_treemap(self.clustered_df)
        self.assertIsNotNone(fig_treemap)
        
        fig_pareto = plot_pareto_revenue(self.clustered_df)
        self.assertIsNotNone(fig_pareto)
        
        fig_radar = plot_segment_radar_profiles(self.meta["profiles"])
        self.assertIsNotNone(fig_radar)
        
        fig_cat = plot_category_mix_by_segment(self.clustered_df)
        self.assertIsNotNone(fig_cat)
        
        first_cust = self.clustered_df.iloc[0]
        fig_donut = plot_customer_category_donut(first_cust)
        self.assertIsNotNone(fig_donut)
        
    def test_app_py_syntax(self):
        # Verify app.py can compile without SyntaxError
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, "app.py", "exec")
        self.assertIsNotNone(compiled)

if __name__ == "__main__":
    unittest.main()
