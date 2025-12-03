"""
Create a comprehensive comparison graph of all modality combination results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from your testing
results = {
    'Visual Only': {
        'correlation': None,  # You didn't share this one
        'mae': None,
        'color': '#FF6B6B'
    },
    'Audio Only': {
        'correlation': None,  # You didn't share this one
        'mae': None,
        'color': '#4ECDC4'
    },
    'Text Only': {
        'correlation': -0.0604,
        'mae': 0.4893,
        'color': '#95E1D3'
    },
    'Visual + Audio': {
        'correlation': 0.0214,
        'mae': 0.6642,
        'color': '#FFA07A'
    },
    'Visual + Text': {
        'correlation': 0.1128,
        'mae': 0.4749,
        'color': '#98D8C8'
    },
    'Audio + Text': {
        'correlation': -0.0281,
        'mae': 0.4835,
        'color': '#F7DC6F'
    },
    'All Three (Visual + Audio + Text)': {
        'correlation': 0.6360,
        'mae': 0.9172,
        'color': '#52D017'
    }
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Multimodal Sentiment Analysis: Modality Combination Comparison', 
             fontsize=18, fontweight='bold', y=1.02)

# Extract data
combinations = list(results.keys())
correlations = [results[c]['correlation'] if results[c]['correlation'] is not None else 0 for c in combinations]
maes = [results[c]['mae'] if results[c]['mae'] is not None else 0 for c in combinations]
colors = [results[c]['color'] for c in combinations]

# Filter out None values for display
valid_indices = [i for i, corr in enumerate(correlations) if corr is not None]
valid_combinations = [combinations[i] for i in valid_indices]
valid_correlations = [correlations[i] for i in valid_indices]
valid_maes = [maes[i] for i in valid_indices]
valid_colors = [colors[i] for i in valid_indices]

# Plot 1: Correlation Comparison
bars1 = ax1.bar(range(len(valid_combinations)), valid_correlations, 
                color=valid_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Modality Combination', fontsize=13, fontweight='bold')
ax1.set_ylabel('Pearson Correlation Coefficient', fontsize=13, fontweight='bold')
ax1.set_title('Test Correlation by Modality Combination', fontsize=15, fontweight='bold')
ax1.set_xticks(range(len(valid_combinations)))
ax1.set_xticklabels(valid_combinations, rotation=45, ha='right', fontsize=10)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([-0.15, 0.75])

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, valid_correlations)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.03,
             f'{val:.4f}',
             ha='center', va='bottom' if height >= 0 else 'top', 
             fontsize=10, fontweight='bold')

# Highlight the best result
best_idx = valid_correlations.index(max(valid_correlations))
bars1[best_idx].set_edgecolor('gold')
bars1[best_idx].set_linewidth(3)

# Plot 2: MAE Comparison
bars2 = ax2.bar(range(len(valid_combinations)), valid_maes, 
                color=valid_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Modality Combination', fontsize=13, fontweight='bold')
ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
ax2.set_title('Test MAE by Modality Combination', fontsize=15, fontweight='bold')
ax2.set_xticks(range(len(valid_combinations)))
ax2.set_xticklabels(valid_combinations, rotation=45, ha='right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, valid_maes)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Highlight the best result (lowest MAE)
best_mae_idx = valid_maes.index(min(valid_maes))
bars2[best_mae_idx].set_edgecolor('gold')
bars2[best_mae_idx].set_linewidth(3)

plt.tight_layout()

# Save the figure
output_path = Path(__file__).parent / 'modality_combinations_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison graph saved to: {output_path}")

# Also create a summary table
fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
for combo in valid_combinations:
    corr = results[combo]['correlation']
    mae = results[combo]['mae']
    corr_str = f'{corr:.4f}' if corr is not None else 'N/A'
    mae_str = f'{mae:.4f}' if mae is not None else 'N/A'
    table_data.append([combo, corr_str, mae_str])

table = ax.table(cellText=table_data,
                 colLabels=['Modality Combination', 'Correlation', 'MAE'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style the header
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best correlation row (if it exists)
if 'All Three (Visual + Audio + Text)' in valid_combinations:
    best_combo_idx = valid_combinations.index('All Three (Visual + Audio + Text)')
    for i in range(3):
        table[(best_combo_idx + 1, i)].set_facecolor('#E8F5E9')
        table[(best_combo_idx + 1, i)].set_text_props(weight='bold')

plt.title('Modality Combination Results Summary', fontsize=16, fontweight='bold', pad=20)

# Save table
table_path = Path(__file__).parent / 'modality_combinations_summary_table.png'
plt.savefig(table_path, dpi=300, bbox_inches='tight')
print(f"Summary table saved to: {table_path}")

plt.show()

