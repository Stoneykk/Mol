"""Generate Stage 2 comparison plot (D-MPNN vs MoLFormer)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
datasets_reg = ['ESOL', 'FreeSolv', 'Lipophilicity']
dmpnn_reg = [0.7935, 2.5163, 0.5890]
molformer_reg = [0.8461, 2.6063, 0.6363]
molformer_reg_std = [0.054, 0.093, 0.008]

datasets_cls = ['BBBP', 'Tox21', 'ClinTox']
dmpnn_cls = [0.8121, 0.7532, 0.8797]
molformer_cls = [0.9358, 0.7724, 0.9865]
molformer_cls_std = [0.002, 0.004, 0.002]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Regression plot
x_reg = np.arange(len(datasets_reg))
width = 0.35
ax1.bar(x_reg - width/2, dmpnn_reg, width, label='D-MPNN (Stage 1)', color='#2E86AB', alpha=0.8)
ax1.bar(x_reg + width/2, molformer_reg, width, label='MoLFormer (3 seeds)', 
        yerr=molformer_reg_std, capsize=5, color='#A23B72', alpha=0.8, error_kw={'elinewidth': 2})
ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test RMSE (↓ lower is better)', fontsize=12, fontweight='bold')
ax1.set_title('Regression Tasks', fontsize=14, fontweight='bold')
ax1.set_xticks(x_reg)
ax1.set_xticklabels(datasets_reg)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add winner checkmarks
for i, (d, m) in enumerate(zip(dmpnn_reg, molformer_reg)):
    if d < m:
        ax1.text(i - width/2, d + max(molformer_reg)*0.02, '✓', 
                ha='center', fontsize=18, color='green', fontweight='bold')

# Classification plot
x_cls = np.arange(len(datasets_cls))
ax2.bar(x_cls - width/2, dmpnn_cls, width, label='D-MPNN (Stage 1)', color='#2E86AB', alpha=0.8)
ax2.bar(x_cls + width/2, molformer_cls, width, label='MoLFormer (3 seeds)', 
        yerr=molformer_cls_std, capsize=5, color='#A23B72', alpha=0.8, error_kw={'elinewidth': 2})
ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test ROC-AUC (↑ higher is better)', fontsize=12, fontweight='bold')
ax2.set_title('Classification Tasks', fontsize=14, fontweight='bold')
ax2.set_xticks(x_cls)
ax2.set_xticklabels(datasets_cls)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0.7, 1.02])

# Add winner checkmarks
for i, (d, m) in enumerate(zip(dmpnn_cls, molformer_cls)):
    if m > d:
        ax2.text(i + width/2, m + 0.01, '✓', 
                ha='center', fontsize=18, color='green', fontweight='bold')

# Overall styling
fig.suptitle('Stage 2: D-MPNN vs MoLFormer Comparison (chemprop v2 scaffold split)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
output_path = 'assets/stage2_molformer_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to {output_path}")
