import os, sys
sys.path.append('/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet')
from qsm_modules.qsm_loss_modules.loss import dipole_kernel
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

# Load P6 O1
p_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/patient_6/'
phs_raw = sio.loadmat(os.path.join(p_dir, 'phs1.mat'))['phs']
cos_raw = sio.loadmat(os.path.join(p_dir, 'cos1.mat'))['cos']
msk_raw = sio.loadmat(os.path.join(p_dir, 'msk1.mat'))['msk']

phs_t = torch.tensor(phs_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
msk_t = torch.tensor(msk_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

dk = dipole_kernel([224, 224, 110], [1, 1, 1], B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
Dk_square = dk * dk

stats = sio.loadmat('/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_2/tr-stats.mat')
sus_mean = torch.tensor(stats['out_mean']).float().to(device)
sus_std  = torch.tensor(stats['out_std']).float().to(device)

# Base model
base_weights = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet/savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Jul_10_12_15_pm_model_K_3_given_data_dw_WideResNet_data_source_1/Spinet_QSM_model_14_.pth'
dw = WideResNet().to(device)
dw.load_state_dict(torch.load(base_weights, map_location=device), strict=False)
dw.eval()

conv_list = [m for m in dw.modules() if isinstance(m, nn.Conv3d)]
model_dic = {}
for k in range(3):
    model_dic[f'model_{k}'] = Stage_LoRA(conv_list, rank=4, alpha=8.0).to(device)

def dw_with_stage_lora(x, stage_no):
    lora_stage = model_dic[f'model_{stage_no}']
    hook_handles = []
    for i, conv in enumerate(conv_list):
        lora_layer = lora_stage.lora_layers[i]
        def make_hook(layer):
            def hook(module, input_tensor, output):
                return output + layer(input_tensor[0])
            return hook
        h = conv.register_forward_hook(make_hook(lora_layer))
        hook_handles.append(h)
    out = dw(x)
    for h in hook_handles:
        h.remove()
    return out

def z_complex_to_z_real(z):
    return torch.cat([z.real, z.imag], dim=1)

def z_real_to_z_complex(z):
    return torch.complex(z[:, 0:1], z[:, 1:2])

def b_gpu(y, lambda_val, z_k, w_square, dk_kernel):
    output1 = torch.fft.fftn(y)
    output2 = dk_kernel * output1
    output3 = torch.fft.ifftn(output2)
    output4 = output3 + lambda_val * w_square * z_k
    return output4

def A_gpu(x, lambda_val, w_square, Dk_square_kernel):
    output1 = Dk_square_kernel * torch.fft.fftn(x)
    output2 = torch.fft.ifftn(output1)
    output3 = output2 + lambda_val * w_square * x
    return output3

def CG_GPU_stage(local_field_gpu, z_k_gpu, stage_no, w_square, dk_k, Dk_sq_k):
    dw_stage = model_dic['model_' + str(stage_no)]
    x_0 = torch.zeros_like(local_field_gpu, dtype=torch.float64)
    r_0 = b_gpu(local_field_gpu, dw_stage.lambda_val, z_k_gpu, w_square, dk_k) - A_gpu(x_0, dw_stage.lambda_val, w_square, Dk_sq_k)
    r_old = r_0
    p_old = r_0
    x_old = x_0
    r_stat = [torch.sum(r_old.conj() * r_old).real.item()]

    for _ in range(30):
        r_old_T_r_old = torch.sum(r_old.conj() * r_old)
        if r_old_T_r_old.real.item() < 1e-10:
            return x_old
        if r_old_T_r_old.real.item() > r_stat[-1] and r_stat[-1] < 1e-6:
            return x_old
        r_stat.append(torch.sum(r_old.conj() * r_old).real.item())
        Ap_old = A_gpu(p_old, dw_stage.lambda_val, w_square, Dk_sq_k)
        p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old)
        alpha = r_old_T_r_old / p_old_T_A_p_old
        x_new = x_old + alpha * p_old
        r_new = r_old - alpha * Ap_old
        r_new_T_r_new = torch.sum(r_new.conj() * r_new)
        beta = r_new_T_r_new / r_old_T_r_old
        p_new = r_new + beta * p_old
        r_old = r_new
        p_old = p_new
        x_old = x_new
    return x_old

def forward_unrolled_pass(phs_batch, msk_batch, use_lora=True):
    phs_F = dk * torch.fft.fftn(phs_batch, dim=[2, 3, 4])
    x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4]) * msk_batch

    for k in range(3):
        x_k_complex = x_k_complex * msk_batch
        x_k_real = z_complex_to_z_real(x_k_complex).float()
        x_k_real = (x_k_real - sus_mean) / sus_std

        if use_lora:
            z_k_real = dw_with_stage_lora(x_k_real, k)
        else:
            z_k_real = dw(x_k_real)
            
        z_k_real = z_k_real * sus_std + sus_mean
        z_k_complex = z_real_to_z_complex(z_k_real)

        x_k_j = x_k_complex
        z_k_j = z_k_complex
        exponent = model_dic[f'model_{k}'].p / 2 - 1.0
        if torch.all(exponent == 0).item():
            w_sq = torch.ones_like(x_k_j)
        else:
            w = torch.pow(x_k_j - z_k_j, exponent)
            w_sq = w.conj() * w

        cg_out = CG_GPU_stage(phs_batch, z_k_j, k, w_sq, dk, Dk_square)
        x_k_complex = cg_out * msk_batch

    return x_k_complex * msk_batch

# 1. Shared Baseline
print('Running 1: Shared SpiNet (Zero-Shot on LPCNN)...')
for k in range(3):
    model_dic[f'model_{k}'].lambda_val.data.fill_(0.05)
    model_dic[f'model_{k}'].p.data.fill_(2.0)
with torch.no_grad():
    recon_base = forward_unrolled_pass(phs_t, msk_t, use_lora=False).real.squeeze().cpu().numpy()

# 2. Stage-LoRA (Trained on SNU, Zero-Shot on LPCNN)
print('Running 2: Stage-LoRA (SNU Trained)...')
snu_lora_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/Spinet_QSM_denoisers_with_stage_lora/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Nov_22_11_18_am_model_K_3_given_data_dw_WideResNet_stage_LoRA/Spinet_QSM_model_12_.pth'
snu_ckpt = torch.load(snu_lora_path, map_location=device)
for k in range(3):
    model_dic[f'model_{k}'].load_state_dict(snu_ckpt[f'model_{k}'])
    model_dic[f'model_{k}'].eval()
with torch.no_grad():
    recon_snu = forward_unrolled_pass(phs_t, msk_t, use_lora=True).real.squeeze().cpu().numpy()

# 3. Stage-LoRA (Target-Calibrated on LPCNN - Patient 2, r=4, a=8)
print('Running 3: Stage-LoRA (LPCNN Target-Calibrated)...')
calib_ckpt_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/target_adapt_lpcnn_p2_val_p4_r4_a8/best_val_model.pth'
calib_ckpt = torch.load(calib_ckpt_path, map_location=device)
for k in range(3):
    model_dic[f'model_{k}'].load_state_dict(calib_ckpt[f'model_{k}'])
    model_dic[f'model_{k}'].eval()
with torch.no_grad():
    recon_lpcnn = forward_unrolled_pass(phs_t, msk_t, use_lora=True).real.squeeze().cpu().numpy()

cos_crop = cos_raw * msk_raw
recon_base_crop = recon_base * msk_raw
recon_snu_crop = recon_snu * msk_raw
recon_lpcnn_crop = recon_lpcnn * msk_raw

# Get exact metrics from official Table 4 sources
df_zero = pd.read_csv('/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/lpcnn_all_models_per_volume.csv')
p6o1_zero = df_zero[(df_zero['patient'] == 6) & (df_zero['orientation'] == 1)].iloc[0]
rmse_base = p6o1_zero['Shared_Baseline_rmse']
hfen_base = p6o1_zero['Shared_Baseline_hfen']
rmse_snu = p6o1_zero['LoRA_r4_alpha8_rmse']
hfen_snu = p6o1_zero['LoRA_r4_alpha8_hfen']

# Target calibration from Table 4 sweep (Patient 2, r=4, a=8, best epoch 10)
df_calib = pd.read_csv('/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/target_adapt_lpcnn_p2_val_p4_r4_a8/per_volume_test_metrics_all_epochs.csv')
p6o1_calib = df_calib[(df_calib['epoch'] == 10) & (df_calib['patient'] == 6) & (df_calib['orient'] == 1)].iloc[0]
rmse_lpcnn = p6o1_calib['rmse']
hfen_lpcnn = p6o1_calib['hfen']

print("Patient 6, Orientation 1 Metrics:")
print("Shared SpiNet Zero-Shot: NRMSE = %.2f%%, HFEN = %.2f%%" % (rmse_base, hfen_base))
print("SpiNet + LoRA (SNU Trained): NRMSE = %.2f%%, HFEN = %.2f%%" % (rmse_snu, hfen_snu))
print("SpiNet + LoRA (LPCNN Calibrated): NRMSE = %.2f%%, HFEN = %.2f%%" % (rmse_lpcnn, hfen_lpcnn))

slice_idx = 55
gt_slice = cos_crop[:, :, slice_idx]
base_slice = recon_base_crop[:, :, slice_idx]
snu_slice = recon_snu_crop[:, :, slice_idx]
lpcnn_slice = recon_lpcnn_crop[:, :, slice_idx]

# 5x absolute error difference relative to COSMOS
err_base = np.abs(base_slice - gt_slice) * 5.0
err_snu = np.abs(snu_slice - gt_slice) * 5.0
err_lpcnn = np.abs(lpcnn_slice - gt_slice) * 5.0

fig, axes = plt.subplots(2, 4, figsize=(18, 9.4), dpi=300)

cmap_qsm = 'gray'
vmin, vmax = -0.15, 0.15
cmap_err = 'hot'
vmax_err = 0.25

# First row: Reconstructions with COSMOS and 3 models
t1 = 'COSMOS Reference\n(Ground Truth)'
t2 = 'Shared MoDL-QSM (Zero-Shot)\nNRMSE: %.2f%% | HFEN: %.2f%%' % (rmse_base, hfen_base)
t3 = 'MoDL-QSM + LoRA (SNU In-Domain)\nNRMSE: %.2f%% | HFEN: %.2f%%' % (rmse_snu, hfen_snu)
t4 = 'MoDL-QSM + LoRA (LPCNN Calibrated)\nNRMSE: %.2f%% | HFEN: %.2f%%' % (rmse_lpcnn, hfen_lpcnn)

titles_row1 = [t1, t2, t3, t4]
imgs_row1 = [gt_slice, base_slice, snu_slice, lpcnn_slice]

for j in range(4):
    ax = axes[0, j]
    im = ax.imshow(np.rot90(imgs_row1[j]), cmap=cmap_qsm, vmin=vmin, vmax=vmax)
    ax.set_title(titles_row1[j], fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

# Second row: Reference Anatomy followed by error maps strictly with COSMOS
axes[1, 0].imshow(np.rot90(gt_slice), cmap=cmap_qsm, vmin=vmin, vmax=vmax)
axes[1, 0].set_title('Reference Anatomy\n(COSMOS)', fontsize=11.5, fontweight='bold', pad=8)
axes[1, 0].axis('off')

t2_1 = 'Shared MoDL-QSM Error (5x)\nvs. COSMOS'
t2_2 = 'SNU LoRA Error (5x)\nvs. COSMOS'
t2_3 = 'LPCNN Calibrated Error (5x)\nvs. COSMOS'
titles_row2 = [t2_1, t2_2, t2_3]
err_imgs = [err_base, err_snu, err_lpcnn]

for j in range(3):
    ax = axes[1, j+1]
    im_err = ax.imshow(np.rot90(err_imgs[j]), cmap=cmap_err, vmin=0, vmax=vmax_err)
    ax.set_title(titles_row2[j], fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

fig.text(0.03, 0.73, 'QSM Reconstructions', va='center', rotation='vertical', fontsize=14, fontweight='bold')
fig.text(0.03, 0.28, 'Error Maps (5x)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

cbar_ax1 = fig.add_axes([0.915, 0.54, 0.015, 0.35])
fig.colorbar(im, cax=cbar_ax1, label='Susceptibility (ppm)')
cbar_ax2 = fig.add_axes([0.915, 0.10, 0.015, 0.35])
fig.colorbar(im_err, cax=cbar_ax2, label='Absolute Error x5 (ppm)')

plt.subplots_adjust(left=0.07, right=0.90, hspace=0.20, wspace=0.08)
out_fig = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/figure3_lpcnn_reconstructions_perfect.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print('SUCCESS: Saved to ' + out_fig)
