#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target Domain Adaptation of SpiNet-QSM using Stage-Specific LoRA Adapters.
Configurable via command-line arguments:
--patient: Training patient ID (1, 2, or 3)
--val_patient: Validation patient ID (default: 4)
--rank: LoRA rank (4 or 8)
--alpha: LoRA scaling factor (8.0 or 16.0)
--epochs: Number of epochs (default: 15)
--batch_size: Batch size (default: 2)
--lr: Learning rate (default: 1e-4)
--device_id: GPU ID (0 or 1)
"""

import os
import sys
import time
import math
import csv
import logging
import argparse
from datetime import datetime
import numpy as np
import scipy.io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Target LoRA Adaptation on LPCNN")
parser.add_argument("--patient", type=int, required=True, help="Training Patient ID (1, 2, 3)")
parser.add_argument("--val_patient", type=int, default=4, help="Validation Patient ID (default: 4)")
parser.add_argument("--rank", type=int, default=4, choices=[4, 8], help="LoRA rank")
parser.add_argument("--alpha", type=float, default=16.0, choices=[8.0, 16.0], help="LoRA alpha")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID (0 or 1)")
args = parser.parse_args()

current_directory = os.path.dirname(os.path.abspath(__file__))
spinet_directory = os.path.abspath(os.path.join(current_directory, '../Spinet'))
os.chdir(current_directory)
sys.path.insert(0, spinet_directory)

from qsm_modules.qsm_loss_modules.loss import dipole_kernel, sobel_kernel, total_loss_l1
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from modules.metrics import (
    compute_hfen,
    compute_psnr,
    compute_rmse,
    compute_ssim_numpy_version_2
)

# ----------------- CONFIGURATION -----------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

K_unrolling = 3
batch_size = args.batch_size
num_epochs = args.epochs
learning_rate = args.lr

LoRA_rank = args.rank
LoRA_alpha = args.alpha
train_patient = args.patient
val_patient = args.val_patient

matrix_size = [64, 64, 64]
voxel_size = [1, 1, 1]

# Paths
train_patches_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches/'
train_csv_path    = f'/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_{train_patient}/all_patches.csv'
val_csv_path      = f'/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_{val_patient}/all_patches.csv'
stats_path        = f'/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_{train_patient}/tr-stats.mat'

raw_lpcnn_path    = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/'
test_patients     = [5, 6, 7, 8]
test_orientations = [1, 2, 3, 4]

restore_weights_path = (
    '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet/savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/'
    'experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/'
    'Jul_10_12_15_pm_model_K_3_given_data_dw_WideResNet_data_source_1/'
    'Spinet_QSM_model_14_.pth'
)

output_dir = os.path.join(
    current_directory,
    f'savedModels/target_adapt_lpcnn_p{train_patient}_val_p{val_patient}_r{LoRA_rank}_a{int(LoRA_alpha)}/'
)
os.makedirs(output_dir, exist_ok=True)

# Logging
log_file = os.path.join(output_dir, 'training.log')
logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"=== Target Adaptation Experiment: P{train_patient} -> Val P{val_patient} | r={LoRA_rank}, alpha={LoRA_alpha} on GPU {args.device_id} ===")
logging.info(f"Train Patches: {train_csv_path}")
logging.info(f"Val Patches:   {val_csv_path}")
logging.info(f"Test Cohort:   Patients {test_patients}, Orientations {test_orientations}")

# ----------------- DATA LOADERS -----------------
train_dataset = mydataloader(train_csv_path, train_patches_dir, training=True)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

val_dataset   = mydataloader(val_csv_path, train_patches_dir, training=True)
val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

logging.info(f"Train batches: {len(train_loader)} ({len(train_dataset)} patches from Patient {train_patient})")
logging.info(f"Val batches:   {len(val_loader)} ({len(val_dataset)} patches from Patient {val_patient})")

# Normalization stats from training patient
stats = scipy.io.loadmat(stats_path)
sus_mean = torch.tensor(stats['out_mean']).float().to(device)
sus_std  = torch.tensor(stats['out_std']).float().to(device)
logging.info(f"Norm stats (P{train_patient}): mean={sus_mean.item():.6f}, std={sus_std.item():.6f}")

# ----------------- KERNELS & LOSS -----------------
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
Dk_square = dk * dk
ss = sobel_kernel().float().to(device)

def get_test_kernels(vol_shape):
    dk_t = dipole_kernel(vol_shape, voxel_size, B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
    return dk_t, dk_t * dk_t

dk_test_110, Dk_square_test_110 = get_test_kernels([224, 224, 110])

# Pre-load test volumes into memory
logging.info("Pre-loading holdout test volumes (Patients 5-8, 16 volumes)...")
test_data_cache = []
for p in test_patients:
    for o in test_orientations:
        p_dir = os.path.join(raw_lpcnn_path, f'patient_{p}')
        phs = scipy.io.loadmat(os.path.join(p_dir, f'phs{o}.mat'))['phs']
        cos = scipy.io.loadmat(os.path.join(p_dir, f'cos{o}.mat'))['cos']
        msk = scipy.io.loadmat(os.path.join(p_dir, f'msk{o}.mat'))['msk']

        phs_t = torch.tensor(phs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        msk_t = torch.tensor(msk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        test_data_cache.append({
            'patient': p, 'orient': o,
            'phs_t': phs_t, 'msk_t': msk_t,
            'cos_crop': (cos * msk).astype(np.float32),
            'msk_raw': msk.astype(np.float32)
        })
logging.info(f"Loaded {len(test_data_cache)} test volumes into memory cache.")

# ----------------- COMPLEX HELPERS -----------------
def z_real_to_z_complex(z_real):
    return torch.complex(
        z_real[:, 0, :, :, :].unsqueeze(1),
        z_real[:, 1, :, :, :].unsqueeze(1)
    )

def z_complex_to_z_real(z_complex):
    return torch.cat([z_complex.real, z_complex.imag], axis=1)

# ----------------- MODEL DEFINITION -----------------
class LoRA_Conv3d(nn.Module):
    def __init__(self, base_conv, rank, alpha):
        super(LoRA_Conv3d, self).__init__()
        if base_conv.groups != 1:
            raise ValueError('LoRA_Conv3d requires groups=1')
        self.conv_A = nn.Conv3d(
            base_conv.in_channels,
            rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            bias=False,
            padding_mode=base_conv.padding_mode
        )
        self.conv_B = nn.Conv3d(
            rank,
            base_conv.out_channels,
            kernel_size=1,
            bias=False
        )
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.conv_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.conv_B.weight)

    def forward(self, x):
        return self.conv_B(self.conv_A(x)) * self.scale

class Stage_LoRA(nn.Module):
    def __init__(self, conv_list, initial_lambda, p):
        super(Stage_LoRA, self).__init__()
        self.lora_layers = nn.ModuleList([
            LoRA_Conv3d(conv, LoRA_rank, LoRA_alpha) for conv in conv_list
        ])
        self.lambda_val = nn.Parameter(torch.tensor([float(initial_lambda)], dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor([float(p)], dtype=torch.float32))

# Load Foundation Model (Frozen W0)
dw = WideResNet().to(device)
if not os.path.exists(restore_weights_path):
    raise FileNotFoundError(f"Shared checkpoint not found: {restore_weights_path}")
dw.load_state_dict(torch.load(restore_weights_path, map_location=device), strict=True)
for param in dw.parameters():
    param.requires_grad = False
logging.info(f"Loaded and FROZEN shared base WideResNet from {restore_weights_path}")

# Collect base Conv3d layers
convolution_list = [module for module in dw.modules() if isinstance(module, nn.Conv3d)]
if len(convolution_list) == 0:
    raise ValueError('No Conv3d layers found in the shared denoiser')

# Hook-based stage LoRA dispatch
model_dic = {}
active_stage = None

for i in range(K_unrolling):
    model_dic['model_' + str(i)] = Stage_LoRA(
        convolution_list,
        dw.lambda_val.item(),
        dw.p.item()
    ).to(device)

    for conv_i, conv in enumerate(convolution_list):
        stage_lora = model_dic['model_' + str(i)].lora_layers[conv_i]

        def add_stage_lora(module, inputs, output, stage_no=i, lora=stage_lora):
            if active_stage == stage_no:
                return output + lora(inputs[0])
            return output

        conv.register_forward_hook(add_stage_lora)

num_trainable = sum(p.numel() for mod in model_dic.values() for p in mod.parameters() if p.requires_grad)
num_frozen = sum(p.numel() for p in dw.parameters())
logging.info(f"Trainable LoRA Parameters: {num_trainable:,} | Frozen Base Parameters: {num_frozen:,}")

dw_paramters_dic_list = [{'params': model_dic[f'model_{i}'].parameters()} for i in range(K_unrolling)]
optimizer = optim.Adam(dw_paramters_dic_list, lr=learning_rate)

def dw_with_stage_lora(x_input, stage_no):
    global active_stage
    active_stage = stage_no
    try:
        output = dw(x_input)
    finally:
        active_stage = None
    return output

# ----------------- CG DATA CONSISTENCY -----------------
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

# ----------------- UNROLLED FORWARD PASS -----------------
def forward_unrolled_pass(phs_batch, msk_batch, dk_k, Dk_sq_k):
    curr_batch = phs_batch.shape[0]
    dk_rep = dk_k.repeat(curr_batch, 1, 1, 1, 1) if curr_batch > 1 else dk_k
    phs_F = dk_rep * torch.fft.fftn(phs_batch, dim=[2, 3, 4])
    x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4]) * msk_batch

    for k in range(K_unrolling):
        x_k_complex = x_k_complex * msk_batch
        x_k_real = z_complex_to_z_real(x_k_complex).float()
        x_k_real = (x_k_real - sus_mean) / sus_std

        z_k_real = dw_with_stage_lora(x_k_real, k)
        z_k_real = z_k_real * sus_std + sus_mean
        z_k_complex = z_real_to_z_complex(z_k_real)

        solutions = []
        for j in range(curr_batch):
            x_k_j = x_k_complex[j:j+1]
            z_k_j = z_k_complex[j:j+1]

            exponent = model_dic[f'model_{k}'].p / 2 - 1.0
            if torch.all(exponent == 0).item():
                w_sq = torch.ones_like(x_k_j)
            else:
                w = torch.pow(x_k_j - z_k_j, exponent)
                w_sq = w.conj() * w

            cg_out = CG_GPU_stage(
                phs_batch[j:j+1],
                z_k_j,
                k,
                w_sq,
                dk_k,
                Dk_sq_k
            )
            cg_out = cg_out * msk_batch[j:j+1]
            solutions.append(cg_out)

        x_k_complex = torch.cat(solutions, dim=0) * msk_batch

    return x_k_complex * msk_batch

# ----------------- FULL VOLUME TEST EVALUATION -----------------
def evaluate_test_cohort(epoch_no):
    dw.eval()
    for stage_no in range(K_unrolling):
        model_dic[f'model_{stage_no}'].eval()
        
    results = []
    with torch.no_grad():
        for item in test_data_cache:
            p = item['patient']
            o = item['orient']
            phs_t = item['phs_t']
            msk_t = item['msk_t']
            cos_crop = item['cos_crop']
            msk_raw  = item['msk_raw']

            recon_c = forward_unrolled_pass(phs_t, msk_t, dk_test_110, Dk_square_test_110)
            recon = recon_c.real.squeeze().cpu().numpy()
            recon_crop = recon * msk_raw

            ssim = float(compute_ssim_numpy_version_2(recon_crop, cos_crop))
            psnr = float(compute_psnr(recon_crop, cos_crop))
            rmse = float(compute_rmse(recon_crop, cos_crop))
            hfen = float(compute_hfen(recon_crop, cos_crop))

            results.append({
                'epoch': epoch_no,
                'patient': p, 'orient': o,
                'ssim': ssim, 'psnr': psnr, 'rmse': rmse, 'hfen': hfen
            })

    ssims = [r['ssim'] for r in results]
    psnrs = [r['psnr'] for r in results]
    rmses = [r['rmse'] for r in results]
    hfens = [r['hfen'] for r in results]

    summary = {
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses)),
        'hfen_mean': float(np.mean(hfens)), 'hfen_std': float(np.std(hfens))
    }
    return summary, results

# ----------------- CSV TRACKERS -----------------
epoch_summary_csv = os.path.join(output_dir, 'epoch_metrics_summary.csv')
with open(epoch_summary_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epoch', f'train_loss_p{train_patient}', f'val_loss_p{val_patient}',
        'test_ssim_mean', 'test_ssim_std',
        'test_psnr_mean', 'test_psnr_std',
        'test_rmse_mean', 'test_rmse_std',
        'test_hfen_mean', 'test_hfen_std',
        'lambda_0', 'lambda_1', 'lambda_2', 'time_sec'
    ])
    writer.writeheader()

per_volume_test_csv = os.path.join(output_dir, 'per_volume_test_metrics_all_epochs.csv')
with open(per_volume_test_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epoch', 'patient', 'orient', 'ssim', 'psnr', 'rmse', 'hfen'
    ])
    writer.writeheader()

# ----------------- TRAINING LOOP -----------------
best_val_loss = float('inf')
logging.info(f"Starting Target Adaptation Training (P{train_patient} -> Val P{val_patient}, r={LoRA_rank}, a={LoRA_alpha})...")

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()

    # 1. Train epoch
    dw.train()
    for module in dw.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()
    for i in range(K_unrolling):
        model_dic[f'model_{i}'].train()

    running_train_loss = 0.0
    for phs, msk, sus, _ in train_loader:
        phs = phs.to(device)
        msk = msk.to(device)
        sus = sus.to(device)

        optimizer.zero_grad()
        recon_c = forward_unrolled_pass(phs, msk, dk, Dk_square)
        loss = total_loss_l1(
            chi=recon_c.real,
            y=sus,
            b=phs,
            d=dk,
            m=msk,
            sobel=ss
        )
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * phs.size(0)

    train_loss = running_train_loss / len(train_dataset)

    # 2. Validation epoch
    dw.eval()
    for i in range(K_unrolling):
        model_dic[f'model_{i}'].eval()

    running_val_loss = 0.0
    with torch.no_grad():
        for phs, msk, sus, _ in val_loader:
            phs = phs.to(device)
            msk = msk.to(device)
            sus = sus.to(device)

            recon_c = forward_unrolled_pass(phs, msk, dk, Dk_square)
            loss = total_loss_l1(
                chi=recon_c.real,
                y=sus,
                b=phs,
                d=dk,
                m=msk,
                sobel=ss
            )
            running_val_loss += loss.item() * phs.size(0)

    val_loss = running_val_loss / len(val_dataset)

    # 3. Test Evaluation on ALL 16 volumes of Patients 5, 6, 7, 8
    test_summary, test_results = evaluate_test_cohort(epoch)
    elapsed = time.time() - epoch_start

    lambdas = [model_dic[f'model_{k}'].lambda_val.item() for k in range(K_unrolling)]

    # Log results
    logging.info(
        f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] | "
        f"TrainLoss(P{train_patient}): {train_loss:.5f} | ValLoss(P{val_patient}): {val_loss:.5f} || "
        f"TEST(P5-8): SSIM={test_summary['ssim_mean']:.4f}+/-{test_summary['ssim_std']:.4f}, "
        f"PSNR={test_summary['psnr_mean']:.2f}dB, RMSE={test_summary['rmse_mean']:.2f}%, HFEN={test_summary['hfen_mean']:.2f}% | "
        f"lambdas: {[round(l, 4) for l in lambdas]}"
    )

    # Append to CSVs
    with open(epoch_summary_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', f'train_loss_p{train_patient}', f'val_loss_p{val_patient}',
            'test_ssim_mean', 'test_ssim_std',
            'test_psnr_mean', 'test_psnr_std',
            'test_rmse_mean', 'test_rmse_std',
            'test_hfen_mean', 'test_hfen_std',
            'lambda_0', 'lambda_1', 'lambda_2', 'time_sec'
        ])
        writer.writerow({
            'epoch': epoch,
            f'train_loss_p{train_patient}': train_loss,
            f'val_loss_p{val_patient}': val_loss,
            'test_ssim_mean': test_summary['ssim_mean'],
            'test_ssim_std': test_summary['test_ssim_std'] if 'test_ssim_std' in test_summary else test_summary['ssim_std'],
            'test_psnr_mean': test_summary['psnr_mean'],
            'test_psnr_std': test_summary['psnr_std'],
            'test_rmse_mean': test_summary['rmse_mean'],
            'test_rmse_std': test_summary['rmse_std'],
            'test_hfen_mean': test_summary['hfen_mean'],
            'test_hfen_std': test_summary['hfen_std'],
            'lambda_0': lambdas[0], 'lambda_1': lambdas[1], 'lambda_2': lambdas[2],
            'time_sec': round(elapsed, 1)
        })

    with open(per_volume_test_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'patient', 'orient', 'ssim', 'psnr', 'rmse', 'hfen'
        ])
        writer.writerows(test_results)

    # Save Checkpoints
    ckpt_save = {
        'epoch': epoch,
        'shared_denoiser': dw.state_dict(),
        'LoRA_rank': LoRA_rank,
        'LoRA_alpha': LoRA_alpha,
        f'train_loss_p{train_patient}': train_loss,
        f'val_loss_p{val_patient}': val_loss,
        'test_summary': test_summary
    }
    for k in range(K_unrolling):
        ckpt_save[f'model_{k}'] = model_dic[f'model_{k}'].state_dict()

    torch.save(ckpt_save, os.path.join(output_dir, f'ckpt_epoch_{epoch}.pth'))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(ckpt_save, os.path.join(output_dir, 'best_val_model.pth'))
        logging.info(f"--> Saved NEW BEST model (Val Loss: {best_val_loss:.5f})")

logging.info(f"\nTarget Adaptation on Patient {train_patient} (r={LoRA_rank}, a={LoRA_alpha}) Complete!")
print("Finished successfully!")
