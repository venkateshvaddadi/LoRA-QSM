import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from LoRun_SpiNet_standalone import (
    LoRunSpiNet, WideResNet, dipole_kernel,
    compute_psnr, compute_rmse, compute_hfen, compute_ssim_numpy_version_2
)

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

def evaluate_snu_k_fold(checkpoint_path, raw_data_path, stats_file,
                        patients_list=[7, 8, 9, 10, 11, 12],
                        orientations=[1, 2, 3, 4, 5],
                        device='cuda:1', use_amp=True, save_csv=True, out_dir=None):
    print("=" * 80)
    print("        LoRun-SpiNet FULL SNU DATASET TESTING (K-FOLD PROTOCOL)")
    print("=" * 80)
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Device        : {device} (AMP bfloat16: {use_amp})")
    print(f"Patients      : {patients_list}")
    print(f"Orientations  : {orientations} (all 5 orientations)")
    print(f"Raw Data Path : {raw_data_path}")
    print("=" * 80)

    # 1. Load tr-stats
    stats = scipy.io.loadmat(stats_file)
    sus_mean = torch.tensor(stats['out_mean']).to(device).float()
    sus_std = torch.tensor(stats['out_std']).to(device).float()

    # 2. Build LoRun-SpiNet & Load Weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    base_model = WideResNet()
    K_unrolling = ckpt.get('K_unrolling', 3)
    rank = ckpt.get('rank', 32)
    model = LoRunSpiNet(
        base_denoiser=base_model,
        K_unrolling=K_unrolling,
        rank=rank,
        alpha=32.0,
        freeze_base=True
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    epoch_num = ckpt.get('epoch', 0) + 1
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f"Loaded Epoch {epoch_num} Checkpoint (Val Loss: {val_loss:.6f})")
    print("-" * 80)
    print(f"{'Patient':<10}{'Orientation':<15}{'PSNR (dB)':<15}{'SSIM':<15}{'RMSE':<15}{'HFEN':<15}")
    print("-" * 80)

    results = []

    with torch.no_grad():
        for p in patients_list:
            for o in orientations:
                phs_file = os.path.join(raw_data_path, f"patient_{p}", f"phs{o}.mat")
                msk_file = os.path.join(raw_data_path, f"patient_{p}", f"msk{o}.mat")
                cos_file = os.path.join(raw_data_path, f"patient_{p}", f"cos{o}.mat")

                if not (os.path.exists(phs_file) and os.path.exists(msk_file) and os.path.exists(cos_file)):
                    print(f"Skipping patient_{p} ori_{o}: File not found")
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

                # Ensure strict mask multiplication like spinet_QSM_test_for_k_fold
                msk_np = np.squeeze(msk)
                sus_np = np.squeeze(sus)
                chi_crop = chi_crop * msk_np

                psnr_val = compute_psnr(chi_crop, sus_np)
                rmse_val = compute_rmse(chi_crop, sus_np)
                hfen_val = compute_hfen(chi_crop, sus_np)
                ssim_val = compute_ssim_numpy_version_2(chi_crop, sus_np)

                ssim_float = ssim_val.item() if hasattr(ssim_val, 'item') else float(ssim_val)

                row = {
                    'patient': p,
                    'orientation': o,
                    'psnr': float(psnr_val),
                    'ssim': ssim_float,
                    'rmse': float(rmse_val),
                    'hfen': float(hfen_val)
                }
                results.append(row)
                print(f"{p:<10}{o:<15}{psnr_val:<15.2f}{ssim_float:<15.4f}{rmse_val:<15.4f}{hfen_val:<15.4f}")

    df = pd.DataFrame(results)

    print("=" * 80)
    print("                      AVERAGE SUMMARY STATISTICS (ALL 30 SCANS)")
    print("=" * 80)
    for m in ['psnr', 'ssim', 'rmse', 'hfen']:
        m_mean = df[m].mean()
        m_std = df[m].std()
        print(f"{m.upper():<10} : {m_mean:.4f} ± {m_std:.4f}")
    print("=" * 80)

    # Patient-wise averages
    print("\nPatient-wise Average Metrics:")
    p_group = df.groupby('patient')[['psnr', 'ssim', 'rmse', 'hfen']].mean()
    print(p_group.to_string())
    print("-" * 80)

    # Save to CSV
    if save_csv:
        if out_dir is None:
            out_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        csv_out_dir = os.path.join(out_dir, "output_csv")
        os.makedirs(csv_out_dir, exist_ok=True)
        csv_file = os.path.join(csv_out_dir, f"lorun_spinet_results_epoch_{epoch_num}_all_5_orientations.csv")
        df.to_csv(csv_file, index=False)
        print(f"Results saved successfully to: {csv_file}")

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test LoRun-SpiNet on SNU K-fold testing set (Patients 7-12, Orientations 1-5)")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument('--raw_data_path', type=str, default='/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/raw_data_names_modified/')
    parser.add_argument('--stats_file', type=str, default='/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/tr-stats.mat')
    parser.add_argument('--patients', type=int, nargs='+', default=[7, 8, 9, 10, 11, 12])
    parser.add_argument('--orientations', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    args = parser.parse_args()

    evaluate_snu_k_fold(
        args.checkpoint,
        args.raw_data_path,
        args.stats_file,
        patients_list=args.patients,
        orientations=args.orientations,
        device=args.device,
        use_amp=args.use_amp
    )
