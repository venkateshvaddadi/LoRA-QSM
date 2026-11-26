import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Self-contained 3D WideResNet + LoRA definitions
class Conv3d_LoRA(nn.Module):
    def __init__(self, base_conv, rank=4, alpha=8.0):
        super().__init__()
        self.base_conv = base_conv
        for param in self.base_conv.parameters():
            param.requires_grad = False
        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        kernel_size = base_conv.kernel_size
        stride = base_conv.stride
        padding = base_conv.padding
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Conv3d(in_channels, rank, kernel_size, stride=stride, padding=padding, bias=False)
        self.lora_B = nn.Conv3d(rank, out_channels, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.base_conv(x) + self.scaling * self.lora_B(self.lora_A(x))

# Load Spinet definitions directly
sys.path.append("/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet")
from dw_WideResNet_models import WideResNet_3d
from Spinet_QSM_pipeline import Spinet_QSM, calculate_dipole_kernel

patient_id = 6
orient = 1
data_dir = f"/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/patient_{patient_id}/"

print(f"Loading Patient {patient_id} Orientation {orient}...")
pha_np = sio.loadmat(os.path.join(data_dir, f"phs{orient}.mat"))['phs']
msk_np = sio.loadmat(os.path.join(data_dir, f"msk{orient}.mat"))['msk']
cos_np = sio.loadmat(os.path.join(data_dir, f"cos{orient}.mat"))['cos']

pha = torch.tensor(pha_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
msk = torch.tensor(msk_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

H, W, D = pha.shape[2], pha.shape[3], pha.shape[4]
pad_D = (16 - (D % 16)) % 16
if pad_D > 0:
    pha = torch.nn.functional.pad(pha, (0, pad_D, 0, 0, 0, 0), mode='constant', value=0)
    msk = torch.nn.functional.pad(msk, (0, pad_D, 0, 0, 0, 0), mode='constant', value=0)

k_matrix = calculate_dipole_kernel(torch.zeros_like(pha), b0_direction=[0, 0, 1]).to(device)

base_weights = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet/savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Jul_10_12_15_pm_model_K_3_given_data_dw_WideResNet_data_source_1/Spinet_QSM_model_14_.pth'

print("Inference: Shared Baseline...")
dw_base = WideResNet_3d(in_channels=1, out_channels=1, depth=20, widen_factor=4).to(device)
dw_base.load_state_dict(torch.load(base_weights, map_location=device), strict=True)
shared_model = Spinet_QSM(denoiser=dw_base, K=3, lambda_init=0.05).to(device)
shared_model.eval()
with torch.no_grad():
    recon_base = shared_model(pha, k_matrix, msk)[:, :, :, :, :D].squeeze().cpu().numpy()

# Target Calibrated LoRA (Patient 2, r=4, a=8)
print("Inference: Target Calibrated LoRA (P2)...")
calib_ckpt_path = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/target_adapt_lpcnn_p2_val_p4_r4_a8/best_val_model.pth"
calib_ckpt = torch.load(calib_ckpt_path, map_location=device)

# Build LoRA Spinet
dw_lora = WideResNet_3d(in_channels=1, out_channels=1, depth=20, widen_factor=4).to(device)
dw_lora.load_state_dict(torch.load(base_weights, map_location=device), strict=True)
calib_model = Spinet_QSM(denoiser=dw_lora, K=3, lambda_init=0.05).to(device)

# Load stage models
for k in range(3):
    calib_model.model_dic[f'model_{k}'].load_state_dict(calib_ckpt[f'model_{k}'])

calib_model.eval()
with torch.no_grad():
    recon_calib = calib_model(pha, k_matrix, msk)[:, :, :, :, :D].squeeze().cpu().numpy()

cos_msk = cos_np * msk_np
recon_base_msk = recon_base * msk_np
recon_calib_msk = recon_calib * msk_np

slice_idx = 55
gt_slice = cos_msk[:, :, slice_idx]
base_slice = recon_base_msk[:, :, slice_idx]
calib_slice = recon_calib_msk[:, :, slice_idx]

err_base = np.abs(base_slice - gt_slice) * 5.0
err_calib = np.abs(calib_slice - gt_slice) * 5.0

print("Plotting Figure 3...")
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), dpi=300)

cmap_qsm = 'gray'
vmin, vmax = -0.15, 0.15
cmap_err = 'hot'
vmax_err = 0.25

titles_top = [
    "Ground Truth (COSMOS)",
    "Shared Baseline (Zero-Shot)",
    "Target-Calibrated LoRA (Proposed)"
]
imgs_top = [gt_slice, base_slice, calib_slice]

for j in range(3):
    ax = axes[0, j]
    im = ax.imshow(np.rot90(imgs_top[j]), cmap=cmap_qsm, vmin=vmin, vmax=vmax)
    ax.set_title(titles_top[j], fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')

axes[1, 0].imshow(np.rot90(gt_slice), cmap=cmap_qsm, vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Reference Anatomy", fontsize=12, fontweight='bold', pad=8)
axes[1, 0].axis('off')

axes[1, 1].imshow(np.rot90(err_base), cmap=cmap_err, vmin=0, vmax=vmax_err)
axes[1, 1].set_title("Baseline Error (5x)", fontsize=12, fontweight='bold', pad=8)
axes[1, 1].axis('off')

im_err = axes[1, 2].imshow(np.rot90(err_calib), cmap=cmap_err, vmin=0, vmax=vmax_err)
axes[1, 2].set_title("Proposed Calibrated Error (5x)", fontsize=12, fontweight='bold', pad=8)
axes[1, 2].axis('off')

# Colorbars
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.35])
fig.colorbar(im, cax=cbar_ax1, label='Susceptibility (ppm)')
cbar_ax2 = fig.add_axes([0.92, 0.10, 0.015, 0.35])
fig.colorbar(im_err, cax=cbar_ax2, label='Absolute Error x5 (ppm)')

plt.subplots_adjust(right=0.90, hspace=0.15, wspace=0.08)
out_fig = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/figure3_lpcnn_reconstructions.png"
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print("Successfully generated Figure 3 at " + out_fig)
