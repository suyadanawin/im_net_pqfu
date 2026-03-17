import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# The exact path you found in your PowerShell
file_path = os.path.join('outputs_phase2', 'stats', 'client_class_distribution.csv')

if os.path.exists(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # We drop 'total_samples' because it's too large and will ruin the heatmap scale
    heatmap_data = df.drop(columns=['total_samples']).set_index('client_id')

    # Create the plot
    plt.figure(figsize=(24, 10))
    
    # We plot the first 100 classes for better visibility
    sns.heatmap(heatmap_data.iloc[:, :100], 
                cmap='YlGnBu', 
                cbar_kws={'label': 'Number of Samples per Class'})

    plt.title('Phase 2 Data Heterogeneity: Dirichlet Allocation (Alpha=0.1)', fontsize=16)
    plt.xlabel('Class ID (0-99)', fontsize=12)
    plt.ylabel('Client ID', fontsize=12)
    
    # Save it to your Phase 2 outputs folder
    output_plot = os.path.join('outputs_phase2', 'plots', 'data_heterogeneity_heatmap.png')
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    
    print(f"✅ Success! Heatmap saved to: {output_plot}")
else:
    print(f"❌ Error: File not found at {file_path}")