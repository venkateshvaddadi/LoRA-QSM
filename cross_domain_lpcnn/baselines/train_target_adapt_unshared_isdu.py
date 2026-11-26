#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Fine-Tuning of Unshared MoDL-QSM (ISDU-QSMNet) on Single Target Patient of LPCNN.
Initial weights: SNU pre-trained unshared checkpoint model_49.pt (K=3, models 0, 1, 2)
Path: /media/medimg/Data/venkatesh_data/QSM_venkatesh/SpiNet_QSM_with_individual_denoisers/savedModels/Spinet_QSM_denoisers_with_unshared_weights/experiments_on_given_data/dw_WideResNet/Oct_04_09_04_pm_model_K_3_given_data_dw_WideResNet/model_49.pt

Total trainable parameters: 1,344,822 (3 separate WideResNet denoisers + stage physics parameters).

Configurable via CLI:
--patient: Training patient ID (1, 2, or 3)
--val_patient: Validation patient ID (default: 4)
--epochs: Number of epochs (default: 10)
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Full Unshared (ISDU) MoDL-QSM Fine-tuning on LPCNN")
parser.add_argument("--patient", type=int, required=True, help="Training Patient ID (1, 2, 3)")
parser.add_argument("--val_patient", type=int, default=4, help="Validation Patient ID (default: 4)")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--device_id", type=int, default=1, help="CUDA device ID (0 or 1)")
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

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

K_unrolling = 3
batch_size = args.batch_size
num_epochs = args.epochs
learning_rate = args.lr
train_patient = args.patient
val_patient = args.val_patient

matrix_size = [64, 64, 64]
voxel_size = [1, 1, 1]

train_patches_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches/'
train_csv_path    = f'/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_{train_patient}/all_patches.csv'
val_csv_path      = f'/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_{val_patient}/all_patches.csv'
stats_path        = f'/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_{train_patient}/tr-stats.mat'

raw_lpcnn_path    = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/'
test_patients     = [5, 6, 7, 8]
test_orientations = [1, 2, 3, 4]

unshared_ckpt_path = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/SpiNet_QSM_with_individual_denoisers/savedModels/Spinet_QSM_denoisers_with_unshared_weights/experiments_on_given_data/dw_WideResNet/Oct_04_09_04_pm_model_K_3_given_data_dw_WideResNet/model_49.pt"

output_dir = os.path.join(
    current_directory,
    f'savedModels/target_adapt_unshared_isdu_p{train_patient}_val_p{val_patient}/'
)
os.makedirs(output_dir, exist_ok=True)

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

logging.info(f"=== Unshared (ISDU) MoDL-QSM Full Fine-Tuning: P{train_patient} -> Val P{val_patient} on GPU {args.device_id} ===")
logging.info(f"Initial Checkpoint: {unshared_ckpt_path}")
logging.info(f"Train Patches: {train_csv_path}")
logging.info(f"Val Patches:   {val_csv_path}")
logging.info(f"Test Cohort:   Patients {test_patients}, Orientations {test_orientations}")

train_dataset = mydataloader(train_csv_path, train_patches_dir, training=True)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

val_dataset   = mydataloader(val_csv_path, train_patches_dir, training=True)
val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

logging.info(f"Train batches: {len(train_loader)} ({len(train_dataset)} patches from Patient {train_patient})")
logging.info(f"Val batches:   {len(val_loader)} ({len(val_dataset)} patches from Patient {val_patient})")

stats = scipy.io.loadmat(stats_path)
sus_mean = torch.tensor(stats['out_mean']).float().to(device)
sus_std  = torch.tensor(stats['out_std']).float().to(device)
logging.info(f"Norm stats (P{train_patient}): mean={sus_mean.item():.6f}, std={sus_std.item():.6f}")

dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
Dk_square = dk * dk
ss = sobel_kernel().float().to(device)

def get_test_kernels(vol_shape):
    dk_t = dipole_kernel(vol_shape, voxel_size, B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
    return dk_t, dk_t * dk_t

dk_test_110, Dk_square_test_110 = get_test_kernels([224, 224, 110])

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

def z_real_to_z_complex(z_real):
    return torch.complex(
        z_real[:, 0, :, :, :].unsqueeze(1),
        z_real[:, 1, :, :, :].unsqueeze(1)
    )

def z_complex_to_z_real(z_complex):
    return torch.cat([z_complex.real, z_complex.imag], axis=1)

# Model setup: 3 Unshared WideResNet denoisers
if not os.path.exists(unshared_ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {unshared_ckpt_path}")

ckpt = torch.load(unshared_ckpt_path, map_location=device)
denoisers = nn.ModuleList([WideResNet().to(device) for _ in range(K_unrolling)])

for k in range(K_unrolling):
    denoisers[k].load_state_dict(ckpt[f'model_{k}'], strict=True)
    logging.info(f"Loaded denoiser {k} from checkpoint. lambda_val={denoisers[k].lambda_val.item():.4f}, p={denoisers[k].p.item():.4f}")

# All parameters are trainable!
for k in range(K_unrolling):
    for param in denoisers[k].parameters():
        param.requires_grad = True

num_trainable = sum(p.numel() for p in denoisers.parameters() if p.requires_grad)
logging.info(f"Trainable Parameters (Full Unshared ISDU Model): {num_trainable:,}")

optimizer = optim.Adam(denoisers.parameters(), lr=learning_rate)

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

def CG_GPU_stage(local_field_gpu, z_k_gpu, w_square, dk_k, Dk_sq_k, lambda_val, p_val):
    x_0 = torch.zeros_like(local_field_gpu, dtype=torch.float64)
    r_0 = b_gpu(local_field_gpu, lambda_val, z_k_gpu, w_square, dk_k) - A_gpu(x_0, lambda_val, w_square, Dk_sq_k)
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
        Ap_old = A_gpu(p_old, lambda_val, w_square, Dk_sq_k)
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

def forward_unrolled_pass(phs_batch, msk_batch, dk_k, Dk_sq_k):
    curr_batch = phs_batch.shape[0]
    dk_rep = dk_k.repeat(curr_batch, 1, 1, 1, 1) if curr_batch > 1 else dk_k
    phs_F = dk_rep * torch.fft.fftn(phs_batch, dim=[2, 3, 4])
    x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4]) * msk_batch

    for k in range(K_unrolling):
        dw_k = denoisers[k]
        x_k_complex = x_k_complex * msk_batch
        x_k_real = z_complex_to_z_real(x_k_complex).float()
        x_k_real = (x_k_real - sus_mean) / sus_std

        z_k_real = dw_k(x_k_real)
        z_k_real = z_k_real * sus_std + sus_mean
        z_k_complex = z_real_to_z_complex(z_k_real)

        solutions = []
        for j in range(curr_batch):
            x_k_j = x_k_complex[j:j+1]
            z_k_j = z_k_complex[j:j+1]

            exponent = dw_k.p / 2 - 1.0
            if torch.all(exponent == 0).item():
                w_sq = torch.ones_like(x_k_j)
            else:
                w = torch.pow(x_k_j - z_k_j, exponent)
                w_sq = w.conj() * w

            cg_out = CG_GPU_stage(
                phs_batch[j:j+1],
                z_k_j,
                w_sq,
                dk_k,
                Dk_sq_k,
                dw_k.lambda_val,
                dw_k.p
            )
            solutions.append(cg_out)

        x_k_complex = torch.cat(solutions, dim=0)

    return x_k_complex * msk_batch

def evaluate_test_cohort(epoch):
    denoisers.eval()
    results = []
    with torch.no_grad():
        for item in test_data_cache:
            p = item['patient']
            o = item['orient']
            phs_t = item['phs_t']
            msk_t = item['msk_t']
            cos_crop = item['cos_crop']
            msk_raw = item['msk_raw']

            out_vol = forward_unrolled_pass(phs_t, msk_t, dk_test_110, Dk_square_test_110)
            recon = (out_vol.real * msk_t).squeeze().cpu().numpy().astype(np.float32)

            s = compute_ssim_numpy_version_2(cos_crop, recon)
            p_val = compute_psnr(cos_crop, recon)
            r = compute_rmse(cos_crop, recon)
            h = compute_hfen(cos_crop, recon)

            results.append({
                'epoch': epoch,
                'patient': p,
                'orient': o,
                'ssim': float(s),
                'psnr': float(p_val),
                'rmse': float(r),
                'hfen': float(h)
            })

    ssims = [res['ssim'] for res in results]
    psnrs = [res['psnr'] for res in results]
    rmses = [res['rmse'] for res in results]
    hfens = [res['hfen'] for res in results]

    summary = {
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses)),
        'hfen_mean': float(np.mean(hfens)), 'hfen_std': float(np.std(hfens))
    }
    return summary, results

epoch_summary_csv = os.path.join(output_dir, 'epoch_test_summary.csv')
per_volume_test_csv = os.path.join(output_dir, 'per_volume_test_metrics.csv')

with open(epoch_summary_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epoch', f'train_loss_p{train_patient}', f'val_loss_p{val_patient}',
        'test_ssim_mean', 'test_ssim_std',
        'test_psnr_mean', 'test_psnr_std',
        'test_rmse_mean', 'test_rmse_std',
        'test_hfen_mean', 'test_hfen_std',
        'time_sec'
    ])
    writer.writeheader()

with open(per_volume_test_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epoch', 'patient', 'orient', 'ssim', 'psnr', 'rmse', 'hfen'
    ])
    writer.writeheader()

logging.info("Evaluating Zero-Shot Checkpoint (Epoch 0 before training)...")
summary_0, results_0 = evaluate_test_cohort(epoch=0)
logging.info(
    f"Epoch 0 (Zero-Shot) || TEST(P5-8): SSIM={summary_0['ssim_mean']:.4f}+/-{summary_0['ssim_std']:.4f}, "
    f"PSNR={summary_0['psnr_mean']:.2f}dB, RMSE={summary_0['rmse_mean']:.2f}%, HFEN={summary_0['hfen_mean']:.2f}%"
)

with open(epoch_summary_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epoch', f'train_loss_p{train_patient}', f'val_loss_p{val_patient}',
        'test_ssim_mean', 'test_ssim_std',
        'test_psnr_mean', 'test_psnr_std',
        'test_rmse_mean', 'test_rmse_std',
        'test_hfen_mean', 'test_hfen_std',
        'time_sec'
    ])
    writer.writerow({
        'epoch': 0,
        f'train_loss_p{train_patient}': 0.0,
        f'val_loss_p{val_patient}': 0.0,
        'test_ssim_mean': summary_0['ssim_mean'],
        'test_ssim_std': summary_0['ssim_std'],
        'test_psnr_mean': summary_0['psnr_mean'],
        'test_psnr_std': summary_0['psnr_std'],
        'test_rmse_mean': summary_0['rmse_mean'],
        'test_rmse_std': summary_0['rmse_std'],
        'test_hfen_mean': summary_0['hfen_mean'],
        'test_hfen_std': summary_0['hfen_std'],
        'time_sec': 0.0
    })

with open(per_volume_test_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epoch', 'patient', 'orient', 'ssim', 'psnr', 'rmse', 'hfen'
    ])
    writer.writerows(results_0)

best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()
    denoisers.train()
    train_loss_acc = 0.0

    for i, data in enumerate(train_loader):
        phs, msk, sus, _ = data
        phs = phs.to(device)
        msk = msk.to(device)
        sus = sus.to(device)

        optimizer.zero_grad()
        out_recon = forward_unrolled_pass(phs, msk, dk, Dk_square)
        loss = total_loss_l1(chi=out_recon.real, y=sus, b=phs, d=dk, m=msk, sobel=ss)
        loss.backward()
        optimizer.step()

        train_loss_acc += loss.item()

    train_loss = train_loss_acc / len(train_loader)

    # --- Validation ---
    denoisers.eval()
    val_loss_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            phs, msk, sus, _ = data
            phs = phs.to(device)
            msk = msk.to(device)
            sus = sus.to(device)

            out_recon = forward_unrolled_pass(phs, msk, dk, Dk_square)
            loss = total_loss_l1(chi=out_recon.real, y=sus, b=phs, d=dk, m=msk, sobel=ss)
            val_loss_acc += loss.item()

    val_loss = val_loss_acc / len(val_loader)

    # --- Test Cohort Evaluation ---
    test_summary, test_results = evaluate_test_cohort(epoch)
    elapsed = time.time() - epoch_start

    logging.info(
        f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] | "
        f"TrainLoss(P{train_patient}): {train_loss:.5f} | ValLoss(P{val_patient}): {val_loss:.5f} || "
        f"TEST(P5-8): SSIM={test_summary['ssim_mean']:.4f}+/-{test_summary['ssim_std']:.4f}, "
        f"PSNR={test_summary['psnr_mean']:.2f}dB, RMSE={test_summary['rmse_mean']:.2f}%, HFEN={test_summary['hfen_mean']:.2f}%"
    )

    with open(epoch_summary_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', f'train_loss_p{train_patient}', f'val_loss_p{val_patient}',
            'test_ssim_mean', 'test_ssim_std',
            'test_psnr_mean', 'test_psnr_std',
            'test_rmse_mean', 'test_rmse_std',
            'test_hfen_mean', 'test_hfen_std',
            'time_sec'
        ])
        writer.writerow({
            'epoch': epoch,
            f'train_loss_p{train_patient}': train_loss,
            f'val_loss_p{val_patient}': val_loss,
            'test_ssim_mean': test_summary['ssim_mean'],
            'test_ssim_std': test_summary['ssim_std'],
            'test_psnr_mean': test_summary['psnr_mean'],
            'test_psnr_std': test_summary['psnr_std'],
            'test_rmse_mean': test_summary['rmse_mean'],
            'test_rmse_std': test_summary['rmse_std'],
            'test_hfen_mean': test_summary['hfen_mean'],
            'test_hfen_std': test_summary['hfen_std'],
            'time_sec': round(elapsed, 1)
        })

    with open(per_volume_test_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'patient', 'orient', 'ssim', 'psnr', 'rmse', 'hfen'
        ])
        writer.writerows(test_results)

    ckpt_save = {
        'epoch': epoch,
        'model_states': [denoisers[k].state_dict() for k in range(K_unrolling)],
        f'train_loss_p{train_patient}': train_loss,
        f'val_loss_p{val_patient}': val_loss,
        'test_summary': test_summary
    }
    torch.save(ckpt_save, os.path.join(output_dir, f'ckpt_epoch_{epoch}.pth'))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(ckpt_save, os.path.join(output_dir, 'best_val_model.pth'))
        logging.info(f"--> Saved NEW BEST model (Val Loss: {best_val_loss:.5f})")

logging.info(f"\nUnshared (ISDU) Fine-Tuning on Patient {train_patient} Complete!")
print("Finished successfully!")
