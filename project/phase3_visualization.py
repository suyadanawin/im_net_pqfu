import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Configuration
DATA_DIR = 'outputs_phase3/metrics/'
OUTPUT_FILE = 'phase3_analysis_report.png'

def load_data():
    # Use the subfolder path you specified
    try:
        df_last = pd.read_csv(os.path.join(DATA_DIR, 'oracle_last_layer_roundwise.csv'))
        df_per = pd.read_csv(os.path.join(DATA_DIR, 'oracle_per_layer_roundwise.csv'))
        df_metrics = pd.read_csv(os.path.join(DATA_DIR, 'oracle_round_metrics.csv'))
        
        with open(os.path.join(DATA_DIR, 'oracle_summary.json'), 'r') as f:
            summary = json.load(f)
            
        return df_last, df_per, df_metrics, summary
    except Exception as e:
        print(f"Error loading files from {DATA_DIR}: {e}")
        return None

def plot_unlearning():
    data = load_data()
    if not data: return
    df_last, df_per, df_metrics, summary = data

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    # 1. FC Layer Convergence (The U-Curve)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_last['round'], df_last['fc_weight_l2_distance'], 'b-o', markersize=4, label='Weight L2')
    ax1.set_title("FC Weight Distance to Oracle", fontweight='bold')
    
    # Highlight the minimum (Optimal Forgetting)
    min_val = df_last['fc_weight_l2_distance'].min()
    min_round = df_last.loc[df_last['fc_weight_l2_distance'].idxmin(), 'round']
    ax1.axvline(x=min_round, color='green', linestyle='--', label=f'Min: {min_val:.2f} (R{min_round})')
    ax1.legend()

    # 2. Accuracy Trend (Utility)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_metrics['round'], df_metrics['oracle_val_acc'], 'r-', linewidth=2)
    ax2.set_title(f"Utility: Max Acc {summary['best_val_acc']}%", fontweight='bold')
    ax2.set_ylabel("Validation Accuracy %")

    # 3. Layer-wise Heatmap (Filtered for Weights)
    ax3 = fig.add_subplot(gs[1:, :])
    # Filter for weights only (ignoring bias/BN) to see structural changes clearly
    df_weights = df_per[df_per['layer_name'].str.contains('weight')]
    pivot_df = df_weights.pivot(index='layer_name', columns='round', values='l2_distance')
    
    sns.heatmap(pivot_df, cmap="mako", ax=ax3, cbar_kws={'label': 'L2 Distance'})
    ax3.set_title("Structural Convergence: All Weights vs Oracle", fontweight='bold')

    plt.suptitle(f"Phase 3: Federated Unlearning Analysis (Client {summary['forget_client_id']})", fontsize=18)
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"✅ Success! Report saved as: {os.path.abspath(OUTPUT_FILE)}")

# Add this to your plotting function
def plot_advanced_metrics(df_metrics, df_last):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Global vs Local Drift
    ax1.plot(df_metrics['round'], df_metrics['oracle_to_full_l2'], label='Global Model Distance', color='purple')
    ax1.plot(df_last['round'], df_last['fc_weight_l2_distance'], label='FC Layer Distance', color='blue')
    ax1.set_title("Global vs. Local Convergence")
    ax1.legend()

    # Plot 2: Update Energy
    ax2.plot(df_metrics['round'], df_metrics['oracle_update_norm'], color='orange', label='Update Norm')
    ax2.set_title("Unlearning Update Stability")
    ax2.set_ylabel("Norm Value")
    ax2.legend()
    
    plt.savefig('outputs_phase3/metrics/advanced_dynamics.png')
if __name__ == "__main__":
    plot_unlearning()