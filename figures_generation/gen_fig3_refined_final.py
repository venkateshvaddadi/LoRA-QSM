import os, sys
sys.path.append('/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet')
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from modules.metrics import compute_rmse, compute_hfen

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LoRA_Conv3d(nn.Module):
    def __init__(self, conv, rank=4, alpha=8.0):
        super().__init__()
        self.conv_A = nn.Conv3d(conv.in_channels, rank, conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=False)
        self.conv_B = nn.Conv3d(rank, conv.out_channels, 1, stride=1, padding=0, bias=False)
        self.scale = alpha / rank

    def forward(self, x):
        return self.conv_B(self.conv_A(x)) * self.scale

class Stage_LoRA(nn.Module):
    def __init__(self, conv_list, rank=4, alpha=8.0):
        super().__init__()
        self.lora_layers = nn.ModuleList([LoRA_Conv3d(c, rank, alpha) for c in conv_list])
        self.lambda_val = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))

def dipole_kernel_np(matrix_size, b0=[0, 0, 1]):
    Y, X, Z = np.meshgrid(
        np.linspace(-int(matrix_size[1]/2), int(matrix_size[1]/2)-1, matrix_size[1]),
        np.linspace(-int(matrix_size[0]/2), int(matrix_size[0]/2)-1, matrix_size[0]),
        np.linspace(-int(matrix_size[2]/2), int(matrix_size[2]/2)-1, matrix_size[2])
    )
    D = 1/3 - (X*b0[0] + Y*b0[1] + Z*b0[2])**2 / (X**2 + Y**2 + Z**2 + 1e-8)
    D = np.fft.fftshift(D)
    D[0, 0, 0] = 0
    return torch.tensor(D, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

# Load Data: Patient 6 Orientation 1
data_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/patient_6/'
phs_np = sio.loadmat(os.path.join(data_dir, 'phs1.mat'))['phs']
msk_np = sio.loadmat(os.path.join(data_dir, 'msk1.mat'))['msk']
cos_np = sio.loadmat(os.path.join(data_dir, 'cos1.mat'))['cos']

phs = torch.tensor(phs_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
msk = torch.tensor(msk_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
H, W, D = phs.shape[2], phs.shape[3], phs.shape[4]
pad_D = (16 - (D % 16)) % 16
if pad_D > 0:
    phs = torch.nn.functional.pad(phs, (0, pad_D, 0, 0, 0, 0))
    msk = torch.nn.functional.pad(msk, (0, pad_D, 0, 0, 0, 0))

Dk = dipole_kernel_np(phs.shape[2:])
base_weights = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet/savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Jul_10_12_15_pm_model_K_3_given_data_dw_WideResNet_data_source_1/Spinet_QSM_model_14_.pth'

dw = WideResNet().to(device)
dw.load_state_dict(torch.load(base_weights, map_location=device), strict=False)
dw.eval()

conv_list = [m for m in dw.modules() if isinstance(m, nn.Conv3d)]
model_dic = {}
for k in range(3):
    model_dic[f'model_{k}'] = Stage_LoRA(conv_list, rank=4, alpha=8.0).to(device)

active_stage = None
def make_hook(layer_idx):
    def hook(module, input_tensor, output):
        if active_stage is not None:
            return output + model_dic[f'model_{active_stage}'].lora_layers[layer_idx](input_tensor[0])
        return output
    return hook

for i, conv in enumerate(conv_list):
    conv.register_forward_hook(make_hook(i))

# Exact stats from Patient 2 training
sus_mean = 0.000504
sus_std = 0.029451

def z_complex_to_z_real(z):
    return torch.cat([z.real, z.imag], dim=1)

def z_real_to_z_complex(z):
    return torch.complex(z[:, 0:1], z[:, 1:2])

def run_unroll(use_lora=False, lambdas=[0.05, 0.05, 0.05]):
    global active_stage
    phs_F = Dk * torch.fft.fftn(phs, dim=[2, 3, 4])
    x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4]) * msk
    
    for k in range(3):
        active_stage = k if use_lora else None
        lmbda = float(lambdas[k])
        
        x_k_complex = x_k_complex * msk
        x_k_real = z_complex_to_z_real(x_k_complex).float()
        x_k_real = (x_k_real - sus_mean) / sus_std
        
        z_k_real = dw(x_k_real)
        z_k_real = torch.cat([z_k_real, torch.zeros_like(z_k_real)], dim=1)
        z_k_real = z_k_real * sus_std + sus_mean
        z_k_complex = z_real_to_z_complex(z_k_real)
        
        numerator = Dk * torch.fft.fftn(phs * msk, dim=[2, 3, 4]) + lmbda * torch.fft.fftn(z_k_complex, dim=[2, 3, 4])
        denominator = Dk**2 + lmbda
        x_k_complex = torch.fft.ifftn(numerator / denominator, dim=[2, 3, 4]) * msk
        
    active_stage = None
    return torch.real(x_k_complex[:, :, :, :, :D]).squeeze().cpu().numpy()

# 1. Shared Baseline (Zero-Shot on LPCNN)
print('Running 1: Shared SpiNet (Zero-Shot)...')
with torch.no_grad():
    recon_base = run_unroll(use_lora=False, lambdas=[0.05, 0.05, 0.05])

# 2. Stage-LoRA finetuned on SNU itself (Zero-Shot on LPCNN)
print('Running 2: Stage-LoRA (SNU in-domain trained)...')
snu_lora_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/Spinet_QSM_denoisers_with_stage_lora/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Nov_22_11_18_am_model_K_3_given_data_dw_WideResNet_stage_LoRA/Spinet_QSM_model_12_.pth'
snu_ckpt = torch.load(snu_lora_path, map_location=device)
for k in range(3):
    model_dic[f'model_{k}'].load_state_dict(snu_ckpt[f'model_{k}'])
    model_dic[f'model_{k}'].eval()

with torch.no_grad():
    recon_snu_lora = run_unroll(use_lora=True, lambdas=[0.05, 0.05, 0.05])

# 3. Stage-LoRA calibrated on LPCNN target data (Patient 2, r=4, a=8)
print('Running 3: Stage-LoRA (LPCNN Target-Calibrated)...')
calib_ckpt_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/target_adapt_lpcnn_p2_val_p4_r4_a8/best_val_model.pth'
calib_ckpt = torch.load(calib_ckpt_path, map_location=device)
for k in range(3):
    model_dic[f'model_{k}'].load_state_dict(calib_ckpt[f'model_{k}'])
    model_dic[f'model_{k}'].eval()

summary_csv = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/target_adapt_lpcnn_p2_val_p4_r4_a8/epoch_metrics_summary.csv'
df = pd.read_csv(summary_csv)
best_row = df.loc[df['val_loss_p4'].idxmin()]
lambdas_calib = [best_row['lambda_0'], best_row['lambda_1'], best_row['lambda_2']]

with torch.no_grad():
    recon_lpcnn_lora = run_unroll(use_lora=True, lambdas=lambdas_calib)

cos_msk = cos_np * msk_np
recon_base_msk = recon_base * msk_np
recon_snu_msk = recon_snu_lora * msk_np
recon_lpcnn_msk = recon_lpcnn_lora * msk_np

rmse_base = compute_rmse(recon_base_msk, cos_msk)
hfen_base = compute_hfen(recon_base_msk, cos_msk)

rmse_snu = compute_rmse(recon_snu_msk, cos_msk)
hfen_snu = compute_hfen(recon_snu_msk, cos_msk)

rmse_lpcnn = compute_rmse(recon_lpcnn_msk, cos_msk)
hfen_lpcnn = compute_hfen(recon_lpcnn_msk, cos_msk)

print("Base: NRMSE = %.2f%%, HFEN = %.2f%%" % (rmse_base, hfen_base))
print("SNU LoRA: NRMSE = %.2f%%, HFEN = %.2f%%" % (rmse_snu, hfen_snu))
print("LPCNN Calib: NRMSE = %.2f%%, HFEN = %.2f%%" % (rmse_lpcnn, hfen_lpcnn))

slice_idx = 55
gt_slice = cos_msk[:, :, slice_idx]
base_slice = recon_base_msk[:, :, slice_idx]
snu_slice = recon_snu_msk[:, :, slice_idx]
lpcnn_slice = recon_lpcnn_msk[:, :, slice_idx]

err_base = np.abs(base_slice - gt_slice) * 5.0
err_snu = np.abs(snu_slice - gt_slice) * 5.0
err_lpcnn = np.abs(lpcnn_slice - gt_slice) * 5.0

# ----------------- PLOTTING -----------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9.2), dpi=300)

cmap_qsm = 'gray'
vmin, vmax = -0.15, 0.15
cmap_err = 'hot'
vmax_err = 0.25

titles_row1 = [
    "Ground Truth\n(COSMOS Reference)",
    "Shared SpiNet (Zero-Shot)\nNRMSE: %.2f%% | HFEN: %.2f%%" % (rmse_base, hfen_base),
    "SpiNet + LoRA (SNU In-Domain)\nNRMSE: %.2f%% | HFEN: %.2f%%" % (rmse_snu, hfen_snu),
    "SpiNet + LoRA (LPCNN Calibrated)\nNRMSE: %.2f%% | HFEN: %.2f%%" % (rmse_lpcnn, hfen_lpcnn)
]
imgs_row1 = [gt_slice, base_slice, snu_slice, lpcnn_slice]

for j in range(4):
    ax = axes[0, j]
    im = ax.imshow(np.rot90(imgs_row1[j]), cmap=cmap_qsm, vmin=vmin, vmax=vmax)
    ax.set_title(titles_row1[j], fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

axes[1, 0].imshow(np.rot90(gt_slice), cmap=cmap_qsm, vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Reference Anatomy\n(COSMOS)", fontsize=11.5, fontweight='bold', pad=8)
axes[1, 0].axis('off')

err_imgs = [err_base, err_snu, err_lpcnn]
titles_row2 = [
    "Baseline Error (5x)\nvs. COSMOS",
    "SNU LoRA Error (5x)\nvs. COSMOS",
    "LPCNN Calibrated Error (5x)\nvs. COSMOS (Lowest Artifacts)"
]

for j in range(3):
    ax = axes[1, j+1]
    im_err = ax.imshow(np.rot90(err_imgs[j]), cmap=cmap_err, vmin=0, vmax=vmax_err)
    ax.set_title(titles_row2[j], fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

fig.text(0.04, 0.73, 'QSM Reconstructions', va='center', rotation='vertical', fontsize=14, fontweight='bold')
fig.text(0.04, 0.28, 'Error Maps (5x)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

cbar_ax1 = fig.add_axes([0.915, 0.54, 0.015, 0.35])
fig.colorbar(im, cax=cbar_ax1, label='Susceptibility (ppm)')
cbar_ax2 = fig.add_axes([0.915, 0.10, 0.015, 0.35])
fig.colorbar(im_err, cax=cbar_ax2, label='Absolute Error x5 (ppm)')

plt.subplots_adjust(left=0.08, right=0.90, hspace=0.18, wspace=0.08)
out_fig = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/figure3_lpcnn_reconstructions_4col.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print("Figure 3 successfully generated at " + out_fig)
