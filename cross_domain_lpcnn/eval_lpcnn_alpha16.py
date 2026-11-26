#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, csv, time
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

current_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM'
spinet_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet'
os.chdir(current_dir)
sys.path.insert(0, spinet_dir)

from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from modules.metrics import (
    compute_hfen,
    compute_psnr,
    compute_rmse,
    compute_ssim_numpy_version_2
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running evaluation on device:", device)

def z_real_to_z_complex(z_real):
    return torch.complex(z_real[:, 0:1, ...], z_real[:, 1:2, ...])

def z_complex_to_z_real(z_complex):
    return torch.cat([z_complex.real, z_complex.imag], dim=1)

def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N, 16.0)) * 16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0], N_dif[0]), (N_dif[1], N_dif[1]), (N_dif[2], N_dif[2]))
    pad_field = np.pad(input_field, pad_width=npad, mode='constant', constant_values=0)
    return pad_field, N_dif

def crop_data(result_pad, N_dif):
    result_pad = np.squeeze(result_pad)
    N_p = np.shape(result_pad)
    return result_pad[N_dif[0]:N_p[0]-N_dif[0], N_dif[1]:N_p[1]-N_dif[1], N_dif[2]:N_p[2]-N_dif[2]]

dk_cache = {}
def get_dipole_kernels(matrix_size):
    size_key = tuple(matrix_size)
    if size_key not in dk_cache:
        voxel_size = [1, 1, 1]
        k = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
        dk_cache[size_key] = (k, k * k)
    return dk_cache[size_key]

csv_path = '../QSM_data/data_for_experiments/given_data/data_source_1/'
stats = sio.loadmat(csv_path + 'csv_files/tr-stats.mat')
sus_mean = torch.tensor(stats['out_mean']).to(device)
sus_std = torch.tensor(stats['out_std']).to(device)

class LoRA_Conv3d(nn.Module):
    def __init__(self, base_conv, rank, alpha):
        super(LoRA_Conv3d, self).__init__()
        self.conv_A = nn.Conv3d(
            base_conv.in_channels, rank, kernel_size=base_conv.kernel_size,
            stride=base_conv.stride, padding=base_conv.padding, dilation=base_conv.dilation, bias=False
        )
        self.conv_B = nn.Conv3d(rank, base_conv.out_channels, kernel_size=1, bias=False)
        self.scale = alpha / rank
    def forward(self, x):
        return self.conv_B(self.conv_A(x)) * self.scale

class Stage_LoRA(nn.Module):
    def __init__(self, conv_list, rank, alpha, initial_lambda, p):
        super(Stage_LoRA, self).__init__()
        self.lora_layers = nn.ModuleList([
            LoRA_Conv3d(conv, rank, alpha) for conv in conv_list
        ])
        self.lambda_val = nn.Parameter(initial_lambda.detach().clone())
        self.register_buffer('p', p.detach().clone())

def A_gpu(x, lambda_val, w_square, Dk_square):
    output1 = Dk_square * torch.fft.fftn(x)
    output2 = torch.fft.ifftn(output1)
    return output2 + lambda_val * w_square * x

def b_gpu(y, lambda_val, z_k, w_square, dk):
    output1 = torch.fft.fftn(y)
    output2 = dk * output1
    output3 = torch.fft.ifftn(output2)
    return output3 + lambda_val * w_square * z_k

def CG_GPU(local_field_gpu, z_k_gpu, lambda_val, w_square, dk, Dk_square):
    x_0 = torch.zeros_like(local_field_gpu, dtype=torch.float64)
    r_0 = b_gpu(local_field_gpu, lambda_val, z_k_gpu, w_square, dk) - A_gpu(x_0, lambda_val, w_square, Dk_square)
    r_old, p_old, x_old = r_0, r_0, x_0
    r_stat = [torch.sum(r_old.conj() * r_old).real.item()]
    for i in range(30):
        r_old_T_r_old = torch.sum(r_old.conj() * r_old)
        if r_old_T_r_old.real.item() < 1e-10:
            return x_old
        if r_old_T_r_old.real.item() > r_stat[-1] and r_stat[-1] < 1e-6:
            return x_old
        r_stat.append(torch.sum(r_old.conj() * r_old).real.item())
        Ap_old = A_gpu(p_old, lambda_val, w_square, Dk_square)
        alpha = r_old_T_r_old / torch.sum(p_old.conj() * Ap_old)
        x_new = x_old + alpha * p_old
        r_new = r_old - alpha * Ap_old
        beta = torch.sum(r_new.conj() * r_new) / r_old_T_r_old
        p_new = r_new + beta * p_old
        r_old, p_old, x_old = r_new, p_new, x_new
    return x_old

lpcnn_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/'
patients = list(range(1, 9))
orientations = list(range(1, 5))

def evaluate_model(model_type, ckpt_path, rank=4, alpha=16.0):
    print("=================================================================")
    print("Evaluating:", model_type, "rank:", rank, "alpha:", alpha)
    print("Checkpoint:", ckpt_path)
    print("=================================================================")
    dw = WideResNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    dw.load_state_dict(state['shared_denoiser'])
    dw.eval()
    conv_list = [m for m in dw.modules() if isinstance(m, nn.Conv3d)]
    stage_models = {}
    for s in range(3):
        st = Stage_LoRA(conv_list, rank, alpha, dw.lambda_val, dw.p).to(device)
        st.load_state_dict(state[f'model_{s}'])
        st.eval()
        stage_models[s] = st
    
    active_stage = [None]
    for conv_i, conv in enumerate(conv_list):
        def make_hook(ci):
            def hook(module, inputs, output):
                s = active_stage[0]
                if s is not None:
                    return output + stage_models[s].lora_layers[ci](inputs[0])
                return output
            return hook
        conv.register_forward_hook(make_hook(conv_i))
        
    def forward_denoiser(x, stage):
        active_stage[0] = stage
        out = dw(x)
        active_stage[0] = None
        return out, stage_models[stage].lambda_val, stage_models[stage].p

    results = []
    with torch.no_grad():
        for pat in patients:
            for ori in orientations:
                p_file = f"{lpcnn_path}patient_{pat}/phs{ori}.mat"
                c_file = f"{lpcnn_path}patient_{pat}/cos{ori}.mat"
                m_file = f"{lpcnn_path}patient_{pat}/msk{ori}.mat"
                
                phs_raw = sio.loadmat(p_file)['phs']
                cos_raw = sio.loadmat(c_file)['cos']
                msk_raw = sio.loadmat(m_file)['msk']
                
                phs_pad, n_dif = padding_data(phs_raw)
                msk_pad, _ = padding_data(msk_raw)
                pad_shape = phs_pad.shape
                
                dk, Dk_square = get_dipole_kernels(pad_shape)
                
                phs = torch.tensor(phs_pad).float().unsqueeze(0).unsqueeze(0).to(device)
                msk = torch.tensor(msk_pad).float().unsqueeze(0).unsqueeze(0).to(device)
                
                phs_F = torch.fft.fftn(phs, dim=[2, 3, 4]) * dk
                x_k = torch.fft.ifftn(phs_F, dim=[2, 3, 4])
                
                for k in range(3):
                    x_k = x_k * msk
                    x_real = (z_complex_to_z_real(x_k).float() - sus_mean) / sus_std
                    z_real, stage_lambda, stage_p = forward_denoiser(x_real, k)
                    z_real = z_real * sus_std + sus_mean
                    z_k = z_real_to_z_complex(z_real)
                    
                    exponent = stage_p / 2.0 - 1.0
                    if torch.all(exponent == 0).item():
                        w_sq = torch.ones_like(x_k)
                    else:
                        w = torch.pow(x_k - z_k, exponent)
                        w_sq = w.conj() * w
                    x_k = CG_GPU(phs, z_k, stage_lambda, w_sq, dk, Dk_square) * msk
                
                recon_pad = (x_k.real * msk).squeeze().cpu().numpy()
                recon = crop_data(recon_pad, n_dif)
                cos_crop = cos_raw * msk_raw
                recon_crop = recon * msk_raw
                
                ssim_v = float(compute_ssim_numpy_version_2(recon_crop, cos_crop))
                psnr_v = float(compute_psnr(recon_crop, cos_crop))
                rmse_v = float(compute_rmse(recon_crop, cos_crop))
                hfen_v = float(compute_hfen(recon_crop, cos_crop))
                
                results.append({'patient': pat, 'orientation': ori, 'ssim': ssim_v, 'psnr': psnr_v, 'rmse': rmse_v, 'hfen': hfen_v})
                print(f"Pat {pat} Ori {ori}: SSIM={ssim_v:.4f}, PSNR={psnr_v:.2f}, RMSE={rmse_v:.2f}, HFEN={hfen_v:.2f}")

    ssims = [r['ssim'] for r in results]
    psnrs = [r['psnr'] for r in results]
    rmses = [r['rmse'] for r in results]
    hfens = [r['hfen'] for r in results]
    
    summary = {
        'model': model_type,
        'ssim': f"{np.mean(ssims):.4f} +/- {np.std(ssims):.4f}",
        'psnr': f"{np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f}",
        'rmse': f"{np.mean(rmses):.2f} +/- {np.std(rmses):.2f}",
        'hfen': f"{np.mean(hfens):.2f} +/- {np.std(hfens):.2f}"
    }
    print(f"--> SUMMARY for {model_type}: SSIM={summary['ssim']}, PSNR={summary['psnr']}, RMSE={summary['rmse']}, HFEN={summary['hfen']}")
    
    csv_out = f"/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/lpcnn_cross_domain_{model_type}.csv"
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['patient', 'orientation', 'ssim', 'psnr', 'rmse', 'hfen'])
        w.writeheader()
        w.writerows(results)
    return summary

if __name__ == '__main__':
    r4_a16_ckpt = 'savedModels/Spinet_QSM_denoisers_with_stage_lora/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Nov_23_03_16_pm_model_K_3_given_data_dw_WideResNet_stage_LoRA_r4_alpha16/Spinet_QSM_model_18_.pth'
    r8_a16_ckpt = 'savedModels/Spinet_QSM_denoisers_with_stage_lora/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Nov_23_03_17_pm_model_K_3_given_data_dw_WideResNet_stage_LoRA_r8_alpha16/Spinet_QSM_model_18_.pth'
    
    s_r4 = evaluate_model("Stage_LoRA_r4_alpha16_ep18", r4_a16_ckpt, rank=4, alpha=16.0)
    s_r8 = evaluate_model("Stage_LoRA_r8_alpha16_ep18", r8_a16_ckpt, rank=8, alpha=16.0)
    
    print("\n=======================================================")
    print("FINAL CROSS-DOMAIN COMPARISON (alpha=16) ON LPCNN DATA")
    print("=======================================================")
    for s in [s_r4, s_r8]:
        print(f"{s['model']}: SSIM={s['ssim']} | PSNR={s['psnr']} | RMSE={s['rmse']} | HFEN={s['hfen']}")
