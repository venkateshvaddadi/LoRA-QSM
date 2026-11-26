import os
import sys
import argparse
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from LoRun_SpiNet_standalone import (
    LoRunSpiNet, WideResNet, dipole_kernel,
    padding_data, crop_data, compute_psnr, compute_rmse,
    compute_hfen, compute_ssim_numpy_version_2
)

def run_test(checkpoint_path, raw_data_path, stats_file, patient_list=[7], orientations=[1, 2, 3, 4], device='cuda:1', use_amp=True):
    print(f"Evaluating Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Use AMP: {use_amp} | Patients: {patient_list} | Orientations: {orientations}")

    stats = scipy.io.loadmat(stats_file)
    sus_mean = torch.tensor(stats['out_mean']).to(device).float()
    sus_std = torch.tensor(stats['out_std']).to(device).float()

    ckpt = torch.load(checkpoint_path, map_location=device)
    base_model = WideResNet()
    model = LoRunSpiNet(
        base_denoiser=base_model,
        K_unrolling=ckpt.get('K_unrolling', 3),
        rank=ckpt.get('rank', 32),
        alpha=32.0,
        freeze_base=True
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Checkpoint Epoch: {ckpt['epoch'] + 1} | Val Loss: {ckpt['val_loss']:.6f}")
    print("-" * 80)
    print(f"{'Patient':<10}{'Orientation':<15}{'PSNR (dB)':<15}{'SSIM':<15}{'RMSE':<15}{'HFEN':<15}")
    print("-" * 80)

    all_psnr, all_ssim, all_rmse, all_hfen = [], [], [], []

    with torch.no_grad():
        for p in patient_list:
            for o in orientations:
                phs_file = os.path.join(raw_data_path, f"patient_{p}", f"phs{o}.mat")
                msk_file = os.path.join(raw_data_path, f"patient_{p}", f"msk{o}.mat")
                cos_file = os.path.join(raw_data_path, f"patient_{p}", f"cos{o}.mat")

                if not (os.path.exists(phs_file) and os.path.exists(msk_file) and os.path.exists(cos_file)):
                    print(f"Skipping patient_{p} ori {o} (files not found)")
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

                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        chi_recon = model(phs_t, msk_t, D_vol, D_vol_sq, sus_mean, sus_std)
                else:
                    chi_recon = model(phs_t, msk_t, D_vol, D_vol_sq, sus_mean, sus_std)

                chi_crop = crop_data(chi_recon.detach().cpu().numpy(), N_dif)

                psnr_val = compute_psnr(chi_crop, sus)
                rmse_val = compute_rmse(chi_crop, sus)
                hfen_val = compute_hfen(chi_crop, sus)
                ssim_val = compute_ssim_numpy_version_2(chi_crop, sus)

                all_psnr.append(psnr_val)
                all_ssim.append(ssim_val)
                all_rmse.append(rmse_val)
                all_hfen.append(hfen_val)

                print(f"{p:<10}{o:<15}{psnr_val:<15.4f}{ssim_val:<15.4f}{rmse_val:<15.4f}{hfen_val:<15.4f}")

    print("-" * 80)
    if all_psnr:
        print(f"{'MEAN':<10}{'ALL':<15}{np.mean(all_psnr):<15.4f}{np.mean(all_ssim):<15.4f}{np.mean(all_rmse):<15.4f}{np.mean(all_hfen):<15.4f}")
    print("-" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--patients', type=int, nargs='+', default=[7, 8, 9, 10, 11, 12])
    parser.add_argument('--raw_data_path', type=str, default='/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/given_data/raw_data_names_modified/')
    parser.add_argument('--stats_file', type=str, default='/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/tr-stats.mat')
    args = parser.parse_args()

    run_test(args.checkpoint, args.raw_data_path, args.stats_file, args.patients, orientations=[1, 2, 3, 4], device=args.device, use_amp=args.use_amp)
