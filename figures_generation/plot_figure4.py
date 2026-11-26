import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

base_dir = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels"

models = [
    ("Patient 1 (r=4, a=8)", "target_adapt_lpcnn_p1_val_p4_r4_a8", "#1f77b4", "-"),
    ("Patient 1 (r=8, a=8)", "target_adapt_lpcnn_p1_val_p4_r8_a8", "#aec7e8", "--"),
    ("Patient 2 (r=4, a=8)", "target_adapt_lpcnn_p2_val_p4_r4_a8", "#2ca02c", "-"),
    ("Patient 2 (r=8, a=8)", "target_adapt_lpcnn_p2_val_p4_r8_a8", "#98df8a", "--"),
    ("Patient 3 (r=4, a=8)", "target_adapt_lpcnn_p3_val_p4_r4_a8", "#d62728", "-"),
    ("Patient 3 (r=8, a=8)", "target_adapt_lpcnn_p3_val_p4_r8_a8", "#ff9896", "--"),
]

for label, folder, color, ls in models:
    csv_path = os.path.join(base_dir, folder, "epoch_metrics_summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        epochs = df["epoch"].values
        val_loss = df["val_loss_p4"].values
        ax1.plot(epochs, val_loss, label=label, color=color, linestyle=ls, linewidth=2.2, marker='o', markersize=4)

ax1.set_title("(a) Target Validation Loss Convergence (Patient 4)", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("Calibration Epoch", fontsize=11, fontweight='bold')
ax1.set_ylabel("Validation Loss (L1 on Patient 4)", fontsize=11, fontweight='bold')
ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.6)

# 2. Physics Parameter Divergence (lambda_k)
p2_csv = os.path.join(base_dir, "target_adapt_lpcnn_p2_val_p4_r8_a8", "epoch_metrics_summary.csv")
df_p2 = pd.read_csv(p2_csv)
epochs = df_p2["epoch"].values
l0 = df_p2["lambda_0"].values
l1 = df_p2["lambda_1"].values
l2 = df_p2["lambda_2"].values

ax2.plot(epochs, l0, label=r"Stage 0 ($\lambda_0$): Initial De-streaking", color="#e74c3c", linewidth=2.5, marker='s', markersize=5)
ax2.plot(epochs, l1, label=r"Stage 1 ($\lambda_1$): Structure Reconstruction", color="#f39c12", linewidth=2.5, marker='^', markersize=5)
ax2.plot(epochs, l2, label=r"Stage 2 ($\lambda_2$): Fine Edge Enforcement", color="#2980b9", linewidth=2.5, marker='D', markersize=5)

ax2.axhline(0.05, color='gray', linestyle=':', linewidth=1.5, label=r"Initialization ($\lambda_{\mathrm{init}} = 0.05$)")

ax2.set_title(r"(b) Autonomous Divergence of Physical Step Sizes ($\lambda_k$)", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel("Calibration Epoch", fontsize=11, fontweight='bold')
ax2.set_ylabel(r"Data-Consistency Step Size ($\lambda_k$)", fontsize=11, fontweight='bold')
ax2.legend(loc="upper left", frameon=True, fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
out_png = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/figure4_convergence_and_physics.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print("Figure 4 successfully generated and saved at " + out_png)
