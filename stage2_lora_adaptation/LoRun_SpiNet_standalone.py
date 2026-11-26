#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRun-SpiNet: Deep LoRA-Unfolding Network for Quantitative Susceptibility Mapping (QSM)
========================================================================================
Inspired by "Deep LoRA-Unfolding Networks for Image Restoration" (Wang et al., IEEE TIP 2026).

Combines:
  1. SpiNet (Iterative QSM unrolling via Majorization-Minimization and Conjugate Gradient).
  2. LoRun (Single shared pre-trained base denoiser with lightweight stage-specific LoRA adapters).

Benefits:
  - Eliminates the parameter and GPU memory explosion of K independent unrolled denoisers.
  - Resolves the inability of a single static denoiser to adapt to stage-specific noise/artifact levels.
  - Fully self-contained: Architectures (WideResNet, DeepQSM, QSMnet), stage-aware 3D LoRA wrapper,
    physics-based forward dipole kernel & CG solver, dataset loader, metrics, and training pipeline.
"""

import os
import sys
import time
import math
import random
import argparse
from datetime import datetime

import numpy as np
import scipy.io
from scipy.signal import convolve as convn
import pandas as pd
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. REPRODUCIBILITY & LOGGING
# ==============================================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Logger(object):
    def __init__(self, filename="experiment_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# 2. COMPLEX / REAL DATA CONVERSIONS (FOR 2-CHANNEL 3D REPRESENTATION)
# ==============================================================================

def z_real_to_z_complex(z_real):
    """
    Converts 2-channel real tensor [B, 2, D, H, W] (real, imag) to complex tensor [B, 1, D, H, W].
    """
    return torch.complex(z_real[:, 0:1, ...], z_real[:, 1:2, ...])


def z_complex_to_z_real(z_complex):
    """
    Converts complex tensor [B, 1, D, H, W] to 2-channel real tensor [B, 2, D, H, W].
    """
    return torch.cat([z_complex.real, z_complex.imag], dim=1)


# ==============================================================================
# 3. BASE DENOISER ARCHITECTURES (2-channel input/output for SpiNet)
# ==============================================================================

class wBasicBlock(nn.Module):
    def __init__(self, inplanes=32, planes=32, dropout_rate=0.0):
        super(wBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class WideResNet(nn.Module):
    """
    3D WideResNet Denoiser configured for SpiNet 2-channel (real, imag) representations.
    """
    def __init__(self, in_channels=2, out_channels=2, num_blocks=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = [wBasicBlock(32, 32) for _ in range(num_blocks)]
        self.gen = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Sequential(*layers),
            nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.gen(x)


class DeepQSM(nn.Module):
    """
    3D U-Net Denoiser (DeepQSM) with 2-channel input/output.
    """
    def __init__(self, in_channels=2, out_channels=2):
        super(DeepQSM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Down path
        self.conv10 = nn.Sequential(nn.Conv3d(in_channels, 16, 3, 1, 1), nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv3d(16, 16, 3, 1, 1), nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.down1 = nn.MaxPool3d(2, 2)

        self.conv20 = nn.Sequential(nn.Conv3d(16, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.conv21 = nn.Sequential(nn.Conv3d(32, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.down2 = nn.MaxPool3d(2, 2)

        self.conv30 = nn.Sequential(nn.Conv3d(32, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.conv31 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.down3 = nn.MaxPool3d(2, 2)

        self.conv40 = nn.Sequential(nn.Conv3d(64, 128, 3, 1, 1), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.conv41 = nn.Sequential(nn.Conv3d(128, 128, 3, 1, 1), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.down4 = nn.MaxPool3d(2, 2)

        # Bottleneck
        self.convm1 = nn.Sequential(nn.Conv3d(128, 256, 3, 1, 1), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
        self.convm2 = nn.Sequential(nn.Conv3d(256, 256, 3, 1, 1), nn.BatchNorm3d(256), nn.ReLU(inplace=True))

        # Up path
        self.up1 = nn.ConvTranspose3d(256, 128, 2, 2)
        self.conv50 = nn.Sequential(nn.Conv3d(256, 128, 3, 1, 1), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.conv51 = nn.Sequential(nn.Conv3d(128, 128, 3, 1, 1), nn.BatchNorm3d(128), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.conv60 = nn.Sequential(nn.Conv3d(128, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.conv61 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

        self.up3 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.conv70 = nn.Sequential(nn.Conv3d(64, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.conv71 = nn.Sequential(nn.Conv3d(32, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

        self.up4 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.conv80 = nn.Sequential(nn.Conv3d(32, 16, 3, 1, 1), nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.conv81 = nn.Sequential(nn.Conv3d(16, 16, 3, 1, 1), nn.BatchNorm3d(16), nn.ReLU(inplace=True))

        self.out = nn.Conv3d(16, out_channels, 1, 1)

    def forward(self, x):
        x10 = self.conv10(x)
        x11 = self.conv11(x10)
        xd1 = self.down1(x11)

        x20 = self.conv20(xd1)
        x21 = self.conv21(x20)
        xd2 = self.down2(x21)

        x30 = self.conv30(xd2)
        x31 = self.conv31(x30)
        xd3 = self.down3(x31)

        x40 = self.conv40(xd3)
        x41 = self.conv41(x40)
        xd4 = self.down4(x41)

        xm1 = self.convm1(xd4)
        xm2 = self.convm2(xm1)

        xu1 = self.up1(xm2)
        xc1 = torch.cat((xu1, x41), dim=1)
        x50 = self.conv50(xc1)
        x51 = self.conv51(x50)

        xu2 = self.up2(x51)
        xc2 = torch.cat((xu2, x31), dim=1)
        x60 = self.conv60(xc2)
        x61 = self.conv61(x60)

        xu3 = self.up3(x61)
        xc3 = torch.cat((xu3, x21), dim=1)
        x70 = self.conv70(xc3)
        x71 = self.conv71(x70)

        xu4 = self.up4(x71)
        xc4 = torch.cat((xu4, x11), dim=1)
        x80 = self.conv80(xc4)
        x81 = self.conv81(x80)

        return self.out(x81)


# ==============================================================================
# 4. LoRun STAGE-AWARE 3D LoRA MODULE
# ==============================================================================

class StageLoRAConv3d(nn.Module):
    """
    Stage-Aware LoRA 3D Convolution Wrapper for Deep Unfolding Networks.
    Holds ONE frozen base Conv3d layer shared across all K stages, plus
    K stage-specific pairs of low-rank 1x1x1 bottleneck convolutions:
        W^{(k)} = W_base + (alpha / rank) * (lora_B[k] @ lora_A[k])
    """
    def __init__(self, conv_layer, num_stages=4, rank=8, alpha=1.0, freeze_base=True):
        super().__init__()
        self.conv = conv_layer
        self.num_stages = num_stages
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / float(self.rank)
        self.current_stage = 0

        if freeze_base:
            for p in self.conv.parameters():
                p.requires_grad = False

        in_c = conv_layer.in_channels
        out_c = conv_layer.out_channels

        # Stage-specific 1x1x1 Conv3d adapters
        self.lora_A = nn.ModuleList([
            nn.Conv3d(in_c, rank, kernel_size=1, bias=False) for _ in range(num_stages)
        ])
        self.lora_B = nn.ModuleList([
            nn.Conv3d(rank, out_c, kernel_size=1, bias=False) for _ in range(num_stages)
        ])

        # Zero-initialization for B guarantees identity behavior at epoch 0
        for i in range(num_stages):
            nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i].weight)

    def set_stage(self, stage_idx):
        assert 0 <= stage_idx < self.num_stages, f"Invalid stage {stage_idx} for num_stages={self.num_stages}"
        self.current_stage = stage_idx

    def forward(self, x):
        base_out = self.conv(x)
        lora_out = self.scaling * self.lora_B[self.current_stage](self.lora_A[self.current_stage](x))
        return base_out + lora_out


def inject_stage_lora(module, num_stages=4, rank=8, alpha=1.0, freeze_base=True):
    """
    Recursively replaces all Conv3d layers in a module with StageLoRAConv3d wrappers.
    Also freezes BatchNorm layers if freeze_base is True.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            setattr(module, name, StageLoRAConv3d(child, num_stages=num_stages, rank=rank, alpha=alpha, freeze_base=freeze_base))
        elif isinstance(child, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            if freeze_base:
                for p in child.parameters():
                    p.requires_grad = False
        else:
            inject_stage_lora(child, num_stages=num_stages, rank=rank, alpha=alpha, freeze_base=freeze_base)
    return module


def set_network_stage(module, stage_idx):
    """
    Broadcasts the stage index to all StageLoRAConv3d instances in the network.
    """
    for m in module.modules():
        if isinstance(m, StageLoRAConv3d):
            m.set_stage(stage_idx)


# ==============================================================================
# 5. PHYSICS ENGINE & CONJUGATE GRADIENT SOLVER
# ==============================================================================

def dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]):
    """
    Computes continuous 3D magnetic dipole kernel in k-space:
    D(k) = 1/3 - (k . B0)^2 / |k|^2
    """
    [Y, X, Z] = np.meshgrid(
        np.linspace(-int(matrix_size[1] / 2), int(matrix_size[1] / 2) - 1, matrix_size[1]),
        np.linspace(-int(matrix_size[0] / 2), int(matrix_size[0] / 2) - 1, matrix_size[0]),
        np.linspace(-int(matrix_size[2] / 2), int(matrix_size[2] / 2) - 1, matrix_size[2])
    )

    X = X / (matrix_size[0]) * voxel_size[0]
    Y = Y / (matrix_size[1]) * voxel_size[1]
    Z = Z / (matrix_size[2]) * voxel_size[2]

    numerator = np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2])
    denominator = np.square(X) + np.square(Y) + np.square(Z) + np.finfo(float).eps
    D = 1.0 / 3.0 - np.divide(numerator, denominator)
    D = np.where(np.isnan(D), 0, D)

    # Roll center to origin (equivalent to ifftshift)
    D = np.roll(D, int(np.floor(matrix_size[0] / 2)), axis=0)
    D = np.roll(D, int(np.floor(matrix_size[1] / 2)), axis=1)
    D = np.roll(D, int(np.floor(matrix_size[2] / 2)), axis=2)

    return torch.from_numpy(np.float32(D)).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]


def sobel_kernel():
    s = [
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    ]
    s = torch.FloatTensor(s)
    sx = s
    sy = s.permute(1, 2, 0)
    sz = s.permute(2, 0, 1)
    return torch.stack([sx, sy, sz]).unsqueeze(1)


class SpiNetCGSolver:
    """
    GPU-accelerated Conjugate Gradient (CG) solver for SpiNet data-fidelity MM subproblem:
    (D^H D + lambda * W^2) x = D^H b + lambda * W^2 z
    """
    def __init__(self, D, D_square, max_iter=30, tol=1e-10):
        self.D = D
        self.D_square = D_square
        self.max_iter = max_iter
        self.tol = tol

    def apply_A(self, x, lambda_val, w_square):
        # A(x) = D^2 * x + lambda * w^2 * x
        output1 = self.D_square * torch.fft.fftn(x, dim=[2, 3, 4])
        output2 = torch.fft.ifftn(output1, dim=[2, 3, 4])
        return output2 + lambda_val * (w_square * x)

    def apply_b(self, local_field, lambda_val, z_k, w_square):
        # b_rhs = D^H * local_field + lambda * w^2 * z_k
        output1 = self.D * torch.fft.fftn(local_field, dim=[2, 3, 4])
        output2 = torch.fft.ifftn(output1, dim=[2, 3, 4])
        return output2 + lambda_val * (w_square * z_k)

    def solve(self, local_field, z_k, w_square, lambda_val):
        """
        Solves for batch size 1 (called per sample or batched).
        """
        x_old = torch.zeros_like(local_field, dtype=torch.complex64)
        b_vec = self.apply_b(local_field, lambda_val, z_k, w_square)
        r_old = b_vec - self.apply_A(x_old, lambda_val, w_square)
        p_old = r_old.clone()

        for _ in range(self.max_iter):
            r_old_T_r_old = torch.sum(r_old.conj() * r_old).real
            if r_old_T_r_old.item() < self.tol:
                break

            Ap_old = self.apply_A(p_old, lambda_val, w_square)
            p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old).real + 1e-12

            alpha = r_old_T_r_old / p_old_T_A_p_old
            x_new = x_old + alpha * p_old
            r_new = r_old - alpha * Ap_old

            r_new_T_r_new = torch.sum(r_new.conj() * r_new).real
            beta = r_new_T_r_new / (r_old_T_r_old + 1e-12)

            p_new = r_new + beta * p_old

            r_old = r_new
            p_old = p_new
            x_old = x_new

        return x_old


# ==============================================================================
# 6. UNROLLED LoRun-SpiNet FULL MODEL
# ==============================================================================

class LoRunSpiNet(nn.Module):
    """
    Deep LoRA-Unfolding Network for QSM (LoRun-SpiNet).
    - Unrolls K iterative MM stages.
    - Uses ONE shared 3D denoiser with stage-specific LoRA adapters.
    - Each stage has learnable physical penalty lambda_k and exponent p_k.
    """
    def __init__(self, base_denoiser, K_unrolling=4, rank=8, alpha=1.0, freeze_base=True, MM_steps=1):
        super().__init__()
        self.K_unrolling = K_unrolling
        self.MM_steps = MM_steps

        # Inject stage-specific LoRAs into the base denoiser
        self.denoiser = inject_stage_lora(
            base_denoiser, num_stages=K_unrolling, rank=rank, alpha=alpha, freeze_base=freeze_base
        )

        # Stage-specific learnable physics parameters
        self.lambda_vals = nn.ParameterList([
            nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
            for _ in range(K_unrolling)
        ])
        self.p_vals = nn.ParameterList([
            nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=False)
            for _ in range(K_unrolling)
        ])

    def forward(self, phs, msk, D, D_square, sus_mean, sus_std):
        batch_size = phs.shape[0]
        cg_solver = SpiNetCGSolver(D, D_square)

        # Initial estimate: x_0 = D^H * phs
        phs_F = torch.fft.fftn(phs, dim=[2, 3, 4])
        x_k_complex = torch.fft.ifftn(D * phs_F, dim=[2, 3, 4])

        for k in range(self.K_unrolling):
            set_network_stage(self.denoiser, k)
            lambda_k = torch.clamp(self.lambda_vals[k], min=1e-4, max=10.0)
            p_k = self.p_vals[k]

            x_k_complex = x_k_complex * msk
            x_k_real = z_complex_to_z_real(x_k_complex).float()

            # Normalization
            x_k_real = (x_k_real - sus_mean) / sus_std

            # Denoiser forward with stage k LoRA
            z_k_real = self.denoiser(x_k_real)

            # Denormalization
            z_k_real = z_k_real * sus_std + sus_mean
            z_k_complex = z_real_to_z_complex(z_k_real)

            # Majorization-Minimization and CG physics step
            solutions = []
            for b in range(batch_size):
                x_b = x_k_complex[b:b+1]
                z_b = z_k_complex[b:b+1]
                phs_b = phs[b:b+1].to(torch.complex64)
                msk_b = msk[b:b+1]

                for _ in range(self.MM_steps):
                    diff = x_b - z_b
                    w = torch.pow(torch.abs(diff) + 1e-6, (p_k / 2.0 - 1.0))
                    w_square = w.conj() * w

                    x_b = cg_solver.solve(phs_b, z_b, w_square, lambda_k)
                    x_b = x_b * msk_b

                solutions.append(x_b)

            x_k_complex = torch.cat(solutions, dim=0)

        # Final predicted susceptibility map (real part)
        chi_final = x_k_complex.real * msk
        return chi_final


# ==============================================================================
# 7. EVALUATION METRICS (PSNR, RMSE, HFEN, SSIM)
# ==============================================================================

def compute_psnr(recon, target):
    img1 = np.asarray(recon, dtype=np.float64).copy()
    img2 = np.asarray(target, dtype=np.float64).copy()
    min_img = min(img1.min(), img2.min())
    img1[img1 != 0] -= min_img
    img2[img2 != 0] -= min_img
    max_img = max(img1.max(), img2.max())
    if max_img != 0:
        img1 = 255.0 * img1 / max_img
        img2 = 255.0 * img2 / max_img
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_rmse(recon, target):
    recon = np.asarray(recon, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    numerator = np.linalg.norm(recon.ravel() - target.ravel())
    denominator = np.linalg.norm(target.ravel())
    if denominator == 0:
        return float('inf')
    return 100.0 * numerator / denominator


def compute_hfen(img1, img2):
    img1 = np.squeeze(img1).astype(np.float64)
    img2 = np.squeeze(img2).astype(np.float64)
    filt_siz = np.array([15, 15, 15])
    sig = np.array([1.5, 1.5, 1.5])
    siz = (filt_siz - 1) / 2
    x, y, z = np.meshgrid(np.arange(-siz[0], siz[0] + 1),
                          np.arange(-siz[1], siz[1] + 1),
                          np.arange(-siz[2], siz[2] + 1), indexing='ij')
    h = np.exp(-(x**2 / (2 * sig[0]**2) + y**2 / (2 * sig[1]**2) + z**2 / (2 * sig[2]**2)))
    h = h / np.sum(h)
    arg = (x**2 / sig[0]**4 + y**2 / sig[1]**4 + z**2 / sig[2]**4 - 
           (1.0 / sig[0]**2 + 1.0 / sig[1]**2 + 1.0 / sig[2]**2))
    H = arg * h
    H = H - (np.sum(H) / np.prod(filt_siz))
    img1_log = convn(img1, H, mode='same')
    img2_log = convn(img2, H, mode='same')
    return compute_rmse(img1_log, img2_log)


def gkernel(sigma, sk):
    size_x = 2 * sk[0] + 1
    size_y = 2 * sk[1] + 1
    size_z = 2 * sk[2] + 1
    gaussKernel = np.zeros((size_x, size_y, size_z))
    for x in range(1, size_x + 1):
        for y in range(1, size_y + 1):
            for z in range(1, size_z + 1):
                radiusSquared = (x - (sk[0] + 1))**2 + (y - (sk[1] + 1))**2 + (z - (sk[2] + 1))**2
                gaussKernel[x-1, y-1, z-1] = np.exp(-radiusSquared / (2 * (sigma**2)))
    return gaussKernel


def compute_ssim_numpy_version_2(img1, img2, sw=None, ind=None):
    """
    Exact translation of the 3D SSIM calculator for QSM (version 2) used across QSM scripts.
    """
    if img1.shape != img2.shape:
        return float('-inf')

    s = img1.shape
    img1 = img1.copy().astype(np.float64)
    img2 = img2.copy().astype(np.float64)

    # Scale dynamic range to 0-255 for QSM
    min_img = min(img1.min(), img2.min())
    img1[img1 != 0] = img1[img1 != 0] - min_img
    img2[img2 != 0] = img2[img2 != 0] - min_img

    max_img = max(img1.max(), img2.max())
    if max_img != 0:
        img1 = 255.0 * img1 / max_img
        img2 = 255.0 * img2 / max_img

    if sw is None:
        sw = [2, 2, 2]
    if (s[0] < sw[0]) or (s[1] < sw[1]) or (s[2] < sw[2]):
        return float('-inf')

    if ind is None:
        ind = np.where(img1 != 0)

    window = gkernel(1.5, sw)
    K = [0.01, 0.03]
    L = 255

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    window = window / np.sum(window)

    mu1 = convn(img1, window, mode='same')
    mu2 = convn(img2, window, mode='same')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convn(img1 * img1, window, mode='same') - mu1_sq
    sigma2_sq = convn(img2 * img2, window, mode='same') - mu2_sq
    sigma12 = convn(img1 * img2, window, mode='same') - mu1_mu2

    if (C1 > 0) and (C2 > 0):
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    temp = np.zeros(img1.shape)
    temp[ind] = 1
    iind = np.where(temp == 0)
    ssim_map[iind] = 1.0

    mssim = np.mean(ssim_map[ind])
    return float(mssim)


# ==============================================================================
# 8. DATASET LOADER
# ==============================================================================

class SpiNetQSMDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.col_name = 'FileName' if 'FileName' in self.names.columns else 'patch_name'

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, str(self.names[self.col_name][idx]))
        data = scipy.io.loadmat(file_name)
        phs = torch.tensor(data['phs'], dtype=torch.float32).unsqueeze(0)
        msk = torch.tensor(data['msk'], dtype=torch.float32).unsqueeze(0)
        sus = torch.tensor(data['susc'], dtype=torch.float32).unsqueeze(0)
        phs = phs * msk
        sus = sus * msk
        return phs, msk, sus, file_name


# ==============================================================================
# 9. FULL-VOLUME EVALUATION
# ==============================================================================

def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N, 16.0)) * 16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0], N_dif[0]), (N_dif[1], N_dif[1]), (N_dif[2], N_dif[2]))
    pad_field = np.pad(input_field, pad_width=npad, mode='constant', constant_values=0)
    return pad_field, N_dif, N_16


def crop_data(result_pad, N_dif):
    result_pad = result_pad.squeeze()
    N_p = np.shape(result_pad)
    return result_pad[
        N_dif[0]:N_p[0] - N_dif[0],
        N_dif[1]:N_p[1] - N_dif[1],
        N_dif[2]:N_p[2] - N_dif[2]
    ]


def evaluate_test_patients(model, patient_list, orientations, raw_data_path, sus_mean, sus_std, device):
    model.eval()
    test_metrics = []

    with torch.no_grad():
        for i in patient_list:
            for j in orientations:
                phs_file = os.path.join(raw_data_path, f"patient_{i}", f"phs{j}.mat")
                msk_file = os.path.join(raw_data_path, f"patient_{i}", f"msk{j}.mat")
                cos_file = os.path.join(raw_data_path, f"patient_{i}", f"cos{j}.mat")

                if not (os.path.exists(phs_file) and os.path.exists(msk_file) and os.path.exists(cos_file)):
                    continue

                phs = scipy.io.loadmat(phs_file)['phs']
                msk = scipy.io.loadmat(msk_file)['msk']
                sus = scipy.io.loadmat(cos_file)['cos']

                phs_pad, N_dif, _ = padding_data(phs)
                msk_pad, _, _ = padding_data(msk)

                phs_t = torch.tensor(phs_pad).unsqueeze(0).unsqueeze(0).to(device).float()
                msk_t = torch.tensor(msk_pad).unsqueeze(0).unsqueeze(0).to(device).float()
                phs_t = phs_t * msk_t

                vol_shape = [phs_pad.shape[0], phs_pad.shape[1], phs_pad.shape[2]]
                D_vol = dipole_kernel(vol_shape, [1, 1, 1], [0, 0, 1]).to(device).float()
                D_vol_sq = D_vol * D_vol

                chi_recon = model(phs_t, msk_t, D_vol, D_vol_sq, sus_mean, sus_std)
                chi_crop = crop_data(chi_recon.detach().cpu().numpy(), N_dif)

                psnr_val = compute_psnr(chi_crop, sus)
                rmse_val = compute_rmse(chi_crop, sus)
                hfen_val = compute_hfen(chi_crop, sus)
                ssim_val = compute_ssim_numpy_version_2(chi_crop, sus)

                test_metrics.append({
                    'patient': i,
                    'orientation': j,
                    'PSNR': psnr_val,
                    'RMSE': rmse_val,
                    'HFEN': hfen_val,
                    'SSIM': ssim_val
                })

    return test_metrics


# ==============================================================================
# 10. MAIN PIPELINE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LoRun-SpiNet: Deep LoRA-Unfolding for QSM")
    parser.add_argument('--base_model', type=str, default='WideResNet', choices=['WideResNet', 'DeepQSM'],
                        help='Base backbone denoiser architecture')
    parser.add_argument('--K_unrolling', type=int, default=4,
                        help='Number of unrolling stages (iterations)')
    parser.add_argument('--MM_steps', type=int, default=1,
                        help='Number of Majorization-Minimization steps per stage')
    parser.add_argument('--rank', type=int, default=8,
                        help='LoRA rank for stage adapters')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='LoRA scaling alpha')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size of 3D patches')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                        help='Freeze shared base denoiser weights (LoRun paradigm)')
    parser.add_argument('--pretrained_base', type=str, default='',
                        help='Optional path to pre-trained base denoiser checkpoint')
    parser.add_argument('--train_csv', type=str,
                        default='/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv',
                        help='Path to train.csv')
    parser.add_argument('--val_csv', type=str,
                        default='/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/val.csv',
                        help='Path to val.csv')
    parser.add_argument('--data_dir', type=str,
                        default='/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches/',
                        help='Path to patch directory')
    parser.add_argument('--stats_file', type=str,
                        default='/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/tr-stats.mat',
                        help='Path to tr-stats.mat')
    parser.add_argument('--raw_data_path', type=str,
                        default='/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/raw_data_names_modified/',
                        help='Path containing full patient testing volumes')
    parser.add_argument('--test_patients', type=int, nargs='+', default=[7, 8, 9, 10, 11, 12],
                        help='Testing patient IDs')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable mixed-precision training with bfloat16 AMP')
    return parser.parse_args()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()
    seed_everything(42)
    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    amp_tag = "_bf16" if args.use_amp else ""
    exp_id = f"LoRun_SpiNet_{args.base_model}_K{args.K_unrolling}_rank{args.rank}{amp_tag}_{timestamp}"
    base_dir = os.path.join("experiments", exp_id)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    sys.stdout = Logger(os.path.join(base_dir, "experiment_log.txt"))

    print("=" * 75)
    print("LoRun-SpiNet: DEEP LoRA-UNFOLDING FOR QUANTITATIVE SUSCEPTIBILITY MAPPING")
    print("=" * 75)
    print(f"Base Backbone       : {args.base_model}")
    print(f"Unrolled Stages (K) : {args.K_unrolling}")
    print(f"MM Steps per Stage  : {args.MM_steps}")
    print(f"LoRA Rank (r)       : {args.rank}")
    print(f"LoRA Alpha (alpha)  : {args.alpha}")
    print(f"Mixed Precision AMP : {args.use_amp} (bfloat16)")
    print(f"Freeze Base Weights : {args.freeze_base}")
    print(f"Learning Rate       : {args.lr}")
    print(f"Epochs              : {args.epochs}")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Experiment Output   : {base_dir}")
    print("=" * 75)

    # 1. Normalization stats
    if os.path.exists(args.stats_file):
        stats = scipy.io.loadmat(args.stats_file)
        sus_mean = torch.tensor(stats['out_mean']).to(device).float()
        sus_std = torch.tensor(stats['out_std']).to(device).float()
        print(f"Loaded normalization stats from {args.stats_file}")
    else:
        print("Warning: Stats file not found, defaulting to mean=0, std=1")
        sus_mean = torch.tensor([0.0]).to(device).float()
        sus_std = torch.tensor([1.0]).to(device).float()

    # 2. Build Base Denoiser
    if args.base_model == 'WideResNet':
        base_denoiser = WideResNet(in_channels=2, out_channels=2).to(device)
    else:
        base_denoiser = DeepQSM(in_channels=2, out_channels=2).to(device)

    if args.pretrained_base and os.path.exists(args.pretrained_base):
        ckpt = torch.load(args.pretrained_base, map_location=device)
        st = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else (ckpt['net'] if 'net' in ckpt else ckpt)
        clean_st = {k.replace("module.", ""): v for k, v in st.items()}
        base_denoiser.load_state_dict(clean_st, strict=False)
        print(f"Loaded pretrained backbone weights from: {args.pretrained_base}")

    # 3. Build Full LoRun-SpiNet
    model = LoRunSpiNet(
        base_denoiser=base_denoiser,
        K_unrolling=args.K_unrolling,
        rank=args.rank,
        alpha=args.alpha,
        freeze_base=args.freeze_base,
        MM_steps=args.MM_steps
    ).to(device)

    # Initialize stage lambda values from pretrained checkpoint if available
    if args.pretrained_base and os.path.exists(args.pretrained_base):
        ckpt_init = torch.load(args.pretrained_base, map_location=device)
        if isinstance(ckpt_init, dict) and 'lambda_val' in ckpt_init:
            init_lambda = float(ckpt_init['lambda_val'].item())
            for param in model.lambda_vals:
                param.data.fill_(init_lambda)
            print(f"Initialized stage-wise lambda values to {init_lambda:.4f} from checkpoint.")

    total_params, trainable_params = count_parameters(model)
    print("\n===== Parameter Distribution =====")
    print(f"Total Model Parameters     : {total_params:,}")
    print(f"Trainable (LoRA + Physics) : {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    print(f"Frozen Backbone Parameters : {total_params - trainable_params:,}")

    # 4. DataLoader & Dipole Kernel for Patches
    if not os.path.exists(args.train_csv):
        print(f"Error: Train CSV not found at {args.train_csv}")
        return

    train_dataset = SpiNetQSMDataset(args.train_csv, args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    print(f"Loaded {len(train_dataset)} training patches ({len(train_loader)} batches).")

    val_dataset = SpiNetQSMDataset(args.val_csv, args.data_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
    print(f"Loaded {len(val_dataset)} validation patches ({len(val_loader)} batches).")

    sample = next(iter(train_loader))
    p_shape = [sample[0].shape[2], sample[0].shape[3], sample[0].shape[4]]
    D_patch = dipole_kernel(p_shape, [1, 1, 1], [0, 0, 1]).to(device).float()
    D_patch_sq = D_patch * D_patch
    print(f"Precomputed 3D Dipole Kernel for patch shape {p_shape} on {device}")

    # 3D Sobel operator for structural gradient loss
    sobel = sobel_kernel().to(device).float()

    # 5. Optimizer & Scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    csv_summary_path = os.path.join(base_dir, "epoch_metrics_summary.csv")
    best_psnr = -1.0

    print("\nStarting LoRun-SpiNet Training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for phs, msk, sus, _ in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            phs = phs.to(device).float()
            msk = msk.to(device).float()
            sus = sus.to(device).float()

            optimizer.zero_grad()

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    chi_pred = model(phs, msk, D_patch, D_patch_sq, sus_mean, sus_std)
                    loss_l1 = torch.mean(torch.abs(chi_pred - sus))
                    grad_diff = F.conv3d((sus - chi_pred), sobel, padding=1)
                    loss_grad = torch.mean(torch.abs(grad_diff))
                    total_loss = loss_l1 + 0.5 * loss_grad
            else:
                chi_pred = model(phs, msk, D_patch, D_patch_sq, sus_mean, sus_std)
                loss_l1 = torch.mean(torch.abs(chi_pred - sus))
                grad_diff = F.conv3d((sus - chi_pred), sobel, padding=1)
                loss_grad = torch.mean(torch.abs(grad_diff))
                total_loss = loss_l1 + 0.5 * loss_grad

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += total_loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time

        # Validation loop on val.csv patches (performed EVERY single epoch)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for phs_val, msk_val, sus_val, _ in tqdm.tqdm(val_loader, desc=f"Val {epoch+1}/{args.epochs}"):
                phs_val = phs_val.to(device).float()
                msk_val = msk_val.to(device).float()
                sus_val = sus_val.to(device).float()

                if args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        chi_pred_val = model(phs_val, msk_val, D_patch, D_patch_sq, sus_mean, sus_std)
                        loss_l1_val = torch.mean(torch.abs(chi_pred_val - sus_val))
                        grad_diff_val = F.conv3d((sus_val - chi_pred_val), sobel, padding=1)
                        loss_grad_val = torch.mean(torch.abs(grad_diff_val))
                        val_total = loss_l1_val + 0.5 * loss_grad_val
                else:
                    chi_pred_val = model(phs_val, msk_val, D_patch, D_patch_sq, sus_mean, sus_std)
                    loss_l1_val = torch.mean(torch.abs(chi_pred_val - sus_val))
                    grad_diff_val = F.conv3d((sus_val - chi_pred_val), sobel, padding=1)
                    loss_grad_val = torch.mean(torch.abs(grad_diff_val))
                    val_total = loss_l1_val + 0.5 * loss_grad_val

                running_val_loss += val_total.item()

        epoch_val_loss = running_val_loss / len(val_loader)

        print(f"\n[Epoch {epoch+1:02d}/{args.epochs:02d}] Train Loss: {epoch_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Time: {epoch_time:.1f}s")

        # Save model details (train_loss and val_loss) to CSV after EVERY single epoch
        loss_history_csv = os.path.join(base_dir, "loss_history.csv")
        loss_row = pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': epoch_val_loss,
            'time_sec': epoch_time
        }])
        loss_row.to_csv(loss_history_csv, mode='a', index=False, header=not os.path.exists(loss_history_csv))

        # Save checkpoint weights after EVERY single epoch
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': epoch_val_loss,
            'K_unrolling': args.K_unrolling,
            'rank': args.rank,
            'base_model': args.base_model
        }
        torch.save(ckpt_data, os.path.join(base_dir, "models", f"checkpoint_epoch_{epoch}.pth"))

        # Evaluation on test patients (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            print("Evaluating full test volumes...")
            test_metrics = evaluate_test_patients(
                model, args.test_patients, orientations=[1, 2, 3, 4],
                raw_data_path=args.raw_data_path, sus_mean=sus_mean, sus_std=sus_std, device=device
            )

            if test_metrics:
                psnrs = [m['PSNR'] for m in test_metrics]
                ssims = [m['SSIM'] for m in test_metrics]
                rmses = [m['RMSE'] for m in test_metrics]
                hfens = [m['HFEN'] for m in test_metrics]

                stats = {
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss,
                    'PSNR_mean': float(np.mean(psnrs)),
                    'PSNR_std': float(np.std(psnrs)),
                    'SSIM_mean': float(np.mean(ssims)),
                    'SSIM_std': float(np.std(ssims)),
                    'RMSE_mean': float(np.mean(rmses)),
                    'HFEN_mean': float(np.mean(hfens)),
                    'time_sec': epoch_time
                }
                print(f"--> Test Evaluation: PSNR={stats['PSNR_mean']:.2f}±{stats['PSNR_std']:.2f} | "
                      f"SSIM={stats['SSIM_mean']:.4f}±{stats['SSIM_std']:.4f} | "
                      f"RMSE={stats['RMSE_mean']:.2f}% | HFEN={stats['HFEN_mean']:.4f}")

                df = pd.DataFrame([stats])
                df.to_csv(csv_summary_path, mode='a', index=False, header=not os.path.exists(csv_summary_path))

                if stats['PSNR_mean'] > best_psnr:
                    best_psnr = stats['PSNR_mean']
                    torch.save(ckpt_data, os.path.join(base_dir, "models", "best_lorun_spinet.pth"))
                    print(f"--> Saved new BEST model (PSNR: {best_psnr:.2f} dB)")

    print("\nLoRun-SpiNet execution completed successfully!")
    print(f"All checkpoints and logs stored in: {base_dir}")


if __name__ == '__main__':
    main()
