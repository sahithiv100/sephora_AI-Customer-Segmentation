"""
Quick test of core backend modules
"""
from src.synthetic_data import generate_synthetic_data, get_data_dictionary
from src.features import compute_engineered_features
from src.clustering import perform_clustering_pipeline, evaluate_cluster_range
from src.scenarios import simulate_campaign_scenario, compare_scenarios
from src.guardrails import run_governance_audit

def main():
    print("Testing Sephora segmentation modules...")
    cust, tx = generate_synthetic_data(n_customers=500, random_seed=42)
    print(f"Generated {len(cust)} customers, {len(tx)} transactions.")
    
    cust_eng = compute_engineered_features(cust)
    print(f"Features engineered. Total columns: {len(cust_eng.columns)}")
    
    eval_df = evaluate_cluster_range(cust_eng, k_min=3, k_max=6)
    print("Cluster evaluation table:")
    print(eval_df)
    
    clustered_df, meta = perform_clustering_pipeline(cust_eng, n_clusters=5)
    print(f"Clustering complete. Silhouette score: {meta['silhouette_score']}")
    print(f"Discovered segments: {list(meta['profiles'].keys())}")
    for cid, p in meta["profiles"].items():
        print(f" - Segment {cid}: {p['name']} ({p['stats']['count']} users, ${p['stats']['avg_spend']:.2f} avg spend)")
        
    audit = run_governance_audit(clustered_df)
    print(f"Governance status: {audit['governance_status']}, K-Anon pass: {audit['k_anonymity_passed']}")
    
    first_seg = list(meta['profiles'].values())[0]['name']
    scen = simulate_campaign_scenario(
        clustered_df,
        target_segments=[first_seg],
        channels=['Email', 'App Push'],
        allocated_budget=5000,
        expected_lift_pct=35.0,
        discount_depth_pct=10.0
    )
    print(f"Scenario test: Contacts: {scen['actual_contacts']}, Net Incremental Profit: ${scen['net_incremental_profit']:,.2f}, ROI: {scen['net_roi_pct']:.1f}%")
    
    comp_df = compare_scenarios(
        clustered_df,
        target_segments=[first_seg],
        channels=['Email', 'App Push'],
        budget=5000,
        discount_depth=10.0
    )
    print("Comparison table:")
    print(comp_df)
    print("\nALL MODULE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
