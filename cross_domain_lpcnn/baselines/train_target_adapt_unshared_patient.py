import os
import sys
import time
import argparse
import numpy as np
import scipy.io
import pandas as pd
import csv
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

current_directory = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM'
spinet_directory = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet'
unshared_directory = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/SpiNet_QSM_with_individual_denoisers'

sys.path.insert(0, current_directory)
sys.path.insert(0, spinet_directory)
sys.path.insert(0, unshared_directory)

from qsm_modules.qsm_loss_modules.loss import dipole_kernel, sobel_kernel, total_loss_l1
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from modules.metrics import (
    compute_hfen,
    compute_psnr,
    compute_rmse,
    compute_ssim_numpy_version_2
)

def z_real_to_z_complex(z_real):
    return torch.complex(z_real[:, 0, :, :, :].unsqueeze(1), z_real[:, 1, :, :, :].unsqueeze(1))

def z_complex_to_z_real(z_complex):
    return torch.cat([z_complex.real, z_complex.imag], dim=1)

def evaluate_test_cohort(model_dic, K_unrolling, test_data_cache, dk_test, Dk_sq_test, sus_mean, sus_std, epoch_no):
    for k in range(K_unrolling):
        model_dic['model_' + str(k)].eval()

    test_metrics = []
    with torch.no_grad():
        for item in test_data_cache:
            p = item['patient']
            o = item['orient']
            phs_t = item['phs_t']
            msk_t = item['msk_t']
            cos_crop = item['cos_crop']
            msk_raw = item['msk_raw']

            phs_F = dk_test * torch.fft.fftn(phs_t, dim=[2, 3, 4])
            x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4]) * msk_t

            for k in range(K_unrolling):
                x_k_complex = x_k_complex * msk_t
                x_k_real = z_complex_to_z_real(x_k_complex).float()
                x_k_real = (x_k_real - sus_mean) / sus_std

                dw = model_dic['model_' + str(k)]
                z_k_real = dw(x_k_real) * sus_std + sus_mean
                z_k_complex = z_real_to_z_complex(z_k_real)

                w = torch.pow((x_k_complex - z_k_complex), ((dw.p / 2.0) - 1.0))
                w_square = w.conj() * w

                def b_gpu(y, lam, z):
                    out1 = torch.fft.fftn(y) * dk_test
                    out3 = torch.fft.ifftn(out1)
                    return out3 + lam * (w_square * z)

                def A_gpu(x, lam):
                    out1 = Dk_sq_test * torch.fft.fftn(x)
                    out2 = torch.fft.ifftn(out1)
                    return out2 + lam * (w_square * x)

                x_0 = torch.zeros_like(phs_t, dtype=torch.complex64)
                r_0 = b_gpu(phs_t, dw.lambda_val, z_k_complex) - A_gpu(x_0, dw.lambda_val)
                p_0 = r_0
                r_old, p_old, x_old = r_0, p_0, x_0

                for _ in range(30):
                    r_old_T_r_old = torch.sum(r_old.conj() * r_old)
                    if r_old_T_r_old.real.item() < 1e-10:
                        break
                    Ap_old = A_gpu(p_old, dw.lambda_val)
                    p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old)
                    if torch.abs(p_old_T_A_p_old) < 1e-12:
                        break
                    alpha = r_old_T_r_old / p_old_T_A_p_old
                    x_new = x_old + alpha * p_old
                    r_new = r_old - alpha * Ap_old
                    r_new_T_r_new = torch.sum(r_new.conj() * r_new)
                    beta = r_new_T_r_new / r_old_T_r_old
                    p_new = r_new + beta * p_old
                    r_old, p_old, x_old = r_new, p_new, x_new

                x_k_complex = x_old * msk_t

            recon_crop = (x_k_complex.real * msk_t).squeeze().cpu().numpy().astype(np.float32) * msk_raw

            ssim_val = float(compute_ssim_numpy_version_2(recon_crop, cos_crop))
            psnr_val = float(compute_psnr(recon_crop, cos_crop))
            rmse_val = float(compute_rmse(recon_crop, cos_crop))
            hfen_val = float(compute_hfen(recon_crop, cos_crop))

            test_metrics.append({
                'epoch': epoch_no,
                'patient': p,
                'orient': o,
                'ssim': ssim_val,
                'psnr': psnr_val,
                'rmse': rmse_val,
                'hfen': hfen_val
            })

    ssims = [m['ssim'] for m in test_metrics]
    psnrs = [m['psnr'] for m in test_metrics]
    rmses = [m['rmse'] for m in test_metrics]
    hfens = [m['hfen'] for m in test_metrics]

    summary = {
        'epoch': epoch_no,
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses)),
        'hfen_mean': float(np.mean(hfens)), 'hfen_std': float(np.std(hfens))
    }
    return summary, test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', type=int, default=1)
    parser.add_argument('--val_patient', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device_id', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.device_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    output_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/target_adapt_unshared_patient_' + str(args.patient) + '_val_p' + str(args.val_patient)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )

    ckpt_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/SpiNet_QSM_with_individual_denoisers/savedModels/Spinet_QSM_denoisers_with_unshared_weights/experiments_on_given_data/dw_WideResNet/Oct_04_09_04_pm_model_K_3_given_data_dw_WideResNet/model_49.pt'
    logging.info('=== Unshared MoDL-QSM (ISDU-QSMNet) Fine-Tuning: Patient ' + str(args.patient) + ' -> Val Patient ' + str(args.val_patient) + ' on GPU ' + str(args.device_id) + ' ===')
    logging.info('Initial Checkpoint: ' + ckpt_path)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    K_unrolling = ckpt['K_unrolling']

    model_dic = {}
    param_list = []
    total_params = 0
    for k in range(K_unrolling):
        dw = WideResNet().to(device)
        dw.load_state_dict(ckpt['model_' + str(k)])
        model_dic['model_' + str(k)] = dw
        param_list.append({'params': dw.parameters()})
        p_cnt = sum(p.numel() for p in dw.parameters() if p.requires_grad)
        total_params += p_cnt
        logging.info('Loaded denoiser ' + str(k) + ': lambda=' + str(round(dw.lambda_val.item(), 4)) + ', p=' + str(round(dw.p.item(), 4)))

    logging.info('Trainable Parameters (3 Unshared Denoisers): ' + str(total_params))
    optimizer = optim.Adam(param_list, lr=args.lr)

    stats_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/tr-stats.mat'
    stats = scipy.io.loadmat(stats_path)
    sus_mean = torch.tensor(stats['out_mean']).float().to(device)
    sus_std = torch.tensor(stats['out_std']).float().to(device)
    logging.info('Normalization Stats: mean=' + str(sus_mean.item()) + ', std=' + str(sus_std.item()))

    patch_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches'
    train_csv = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_' + str(args.patient) + '/all_patches.csv'
    val_csv = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_' + str(args.val_patient) + '/all_patches.csv'

    train_dataset = mydataloader(train_csv, patch_dir, training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    val_dataset = mydataloader(val_csv, patch_dir, training=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
    logging.info('Train batches: ' + str(len(train_loader)) + ' (' + str(len(train_dataset)) + ' patches) | Val batches: ' + str(len(val_loader)) + ' (' + str(len(val_dataset)) + ' patches)')

    raw_lpcnn_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/'
    test_patients = [5, 6, 7, 8]
    test_orientations = [1, 2, 3, 4]
    dk_test_110 = dipole_kernel([224, 224, 110], [1, 1, 1], B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
    Dk_square_test_110 = dk_test_110 * dk_test_110

    logging.info('Pre-loading holdout test volumes (Patients 5-8, 16 volumes)...')
    test_data_cache = []
    for p in test_patients:
        for o in test_orientations:
            p_dir = os.path.join(raw_lpcnn_path, 'patient_' + str(p))
            phs = scipy.io.loadmat(os.path.join(p_dir, 'phs' + str(o) + '.mat'))['phs']
            cos = scipy.io.loadmat(os.path.join(p_dir, 'cos' + str(o) + '.mat'))['cos']
            msk = scipy.io.loadmat(os.path.join(p_dir, 'msk' + str(o) + '.mat'))['msk']

            phs_t = torch.tensor(phs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            msk_t = torch.tensor(msk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            test_data_cache.append({
                'patient': p, 'orient': o,
                'phs_t': phs_t, 'msk_t': msk_t,
                'cos_crop': (cos * msk).astype(np.float32),
                'msk_raw': msk.astype(np.float32)
            })
    logging.info('Loaded ' + str(len(test_data_cache)) + ' test volumes into memory.')

    dk_patch = dipole_kernel([64, 64, 64], [1, 1, 1], B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
    Dk_sq_patch = (dk_patch * dk_patch).to(device)
    ss = sobel_kernel().float().to(device)

    logging.info('Evaluating Zero-Shot Checkpoint (Epoch 0 before training)...')
    test_summary_0, test_results_0 = evaluate_test_cohort(model_dic, K_unrolling, test_data_cache, dk_test_110, Dk_square_test_110, sus_mean, sus_std, 0)
    logging.info('Epoch 0 (Zero-Shot) || TEST(P5-8): SSIM=' + f"{test_summary_0['ssim_mean']:.4f}+/-{test_summary_0['ssim_std']:.4f}" + ', PSNR=' + f"{test_summary_0['psnr_mean']:.2f}dB" + ', RMSE=' + f"{test_summary_0['rmse_mean']:.2f}%" + ', HFEN=' + f"{test_summary_0['hfen_mean']:.2f}%")

    summary_csv = os.path.join(output_dir, 'summary_epochs.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'val_loss',
            'test_ssim_mean', 'test_ssim_std',
            'test_psnr_mean', 'test_psnr_std',
            'test_rmse_mean', 'test_rmse_std',
            'test_hfen_mean', 'test_hfen_std',
            'lambda_0', 'lambda_1', 'lambda_2', 'time_sec'
        ])
        writer.writeheader()
        writer.writerow({
            'epoch': 0, 'train_loss': None, 'val_loss': None,
            'test_ssim_mean': test_summary_0['ssim_mean'],
            'test_ssim_std': test_summary_0['ssim_std'],
            'test_psnr_mean': test_summary_0['psnr_mean'],
            'test_psnr_std': test_summary_0['psnr_std'],
            'test_rmse_mean': test_summary_0['rmse_mean'],
            'test_rmse_std': test_summary_0['rmse_std'],
            'test_hfen_mean': test_summary_0['hfen_mean'],
            'test_hfen_std': test_summary_0['hfen_std'],
            'lambda_0': round(model_dic['model_0'].lambda_val.item(), 4),
            'lambda_1': round(model_dic['model_1'].lambda_val.item(), 4),
            'lambda_2': round(model_dic['model_2'].lambda_val.item(), 4),
            'time_sec': 0.0
        })

    per_vol_csv = os.path.join(output_dir, 'test_metrics_per_volume.csv')
    with open(per_vol_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'patient', 'orient', 'ssim', 'psnr', 'rmse', 'hfen'])
        writer.writeheader()
        writer.writerows(test_results_0)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for k in range(K_unrolling):
            model_dic['model_' + str(k)].train()

        train_loss = 0.0
        for phs_b, msk_b, sus_b, _ in train_loader:
            phs_b = phs_b.to(device)
            msk_b = msk_b.to(device)
            sus_b = sus_b.to(device)

            dk_repeat = dk_patch.repeat(args.batch_size, 1, 1, 1, 1)
            phs_F = torch.fft.fftn(phs_b, dim=[2, 3, 4]) * dk_repeat
            x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4])

            for k in range(K_unrolling):
                x_k_complex = x_k_complex * msk_b
                x_k_real = z_complex_to_z_real(x_k_complex).float()
                x_k_real = (x_k_real - sus_mean) / sus_std

                dw = model_dic['model_' + str(k)]
                z_k_real = dw(x_k_real) * sus_std + sus_mean
                z_k_complex = z_real_to_z_complex(z_k_real)

                solutions = []
                for j in range(args.batch_size):
                    x_j = x_k_complex[j:j+1]
                    z_j = z_k_complex[j:j+1]
                    phs_j = phs_b[j:j+1]
                    msk_j = msk_b[j:j+1]

                    w = torch.pow((x_j - z_j), ((dw.p / 2.0) - 1.0))
                    w_square = w.conj() * w

                    def b_gpu(y, lam, z):
                        out1 = torch.fft.fftn(y) * dk_patch
                        out3 = torch.fft.ifftn(out1)
                        return out3 + lam * (w_square * z)

                    def A_gpu(x, lam):
                        out1 = Dk_sq_patch * torch.fft.fftn(x)
                        out2 = torch.fft.ifftn(out1)
                        return out2 + lam * (w_square * x)

                    x_0 = torch.zeros_like(phs_j, dtype=torch.complex64)
                    r_0 = b_gpu(phs_j, dw.lambda_val, z_j) - A_gpu(x_0, dw.lambda_val)
                    p_0 = r_0
                    r_old, p_old, x_old = r_0, p_0, x_0

                    for _ in range(10):
                        r_old_T_r_old = torch.sum(r_old.conj() * r_old)
                        if r_old_T_r_old.real.item() < 1e-10:
                            break
                        Ap_old = A_gpu(p_old, dw.lambda_val)
                        p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old)
                        if torch.abs(p_old_T_A_p_old) < 1e-12:
                            break
                        alpha = r_old_T_r_old / p_old_T_A_p_old
                        x_new = x_old + alpha * p_old
                        r_new = r_old - alpha * Ap_old
                        r_new_T_r_new = torch.sum(r_new.conj() * r_new)
                        beta = r_new_T_r_new / r_old_T_r_old
                        p_new = r_new + beta * p_old
                        r_old, p_old, x_old = r_new, p_new, x_new

                    solutions.append(x_old * msk_j)

                x_k_complex = torch.cat(solutions, dim=0) * msk_b

            x_k_complex = x_k_complex * msk_b
            optimizer.zero_grad()
            loss = total_loss_l1(chi=x_k_complex.real, y=sus_b, b=phs_b, d=dk_patch, m=msk_b, sobel=ss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        for k in range(K_unrolling):
            model_dic['model_' + str(k)].eval()
        val_loss = 0.0
        with torch.no_grad():
            for phs_b, msk_b, sus_b, _ in val_loader:
                phs_b = phs_b.to(device)
                msk_b = msk_b.to(device)
                sus_b = sus_b.to(device)

                dk_repeat = dk_patch.repeat(args.batch_size, 1, 1, 1, 1)
                phs_F = torch.fft.fftn(phs_b, dim=[2, 3, 4]) * dk_repeat
                x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4])

                for k in range(K_unrolling):
                    x_k_complex = x_k_complex * msk_b
                    x_k_real = z_complex_to_z_real(x_k_complex).float()
                    x_k_real = (x_k_real - sus_mean) / sus_std

                    dw = model_dic['model_' + str(k)]
                    z_k_real = dw(x_k_real) * sus_std + sus_mean
                    z_k_complex = z_real_to_z_complex(z_k_real)

                    solutions = []
                    for j in range(args.batch_size):
                        x_j = x_k_complex[j:j+1]
                        z_j = z_k_complex[j:j+1]
                        phs_j = phs_b[j:j+1]
                        msk_j = msk_b[j:j+1]

                        w = torch.pow((x_j - z_j), ((dw.p / 2.0) - 1.0))
                        w_square = w.conj() * w

                        def b_gpu(y, lam, z):
                            out1 = torch.fft.fftn(y) * dk_patch
                            out3 = torch.fft.ifftn(out1)
                            return out3 + lam * (w_square * z)

                        def A_gpu(x, lam):
                            out1 = Dk_sq_patch * torch.fft.fftn(x)
                            out2 = torch.fft.ifftn(out1)
                            return out2 + lam * (w_square * x)

                        x_0 = torch.zeros_like(phs_j, dtype=torch.complex64)
                        r_0 = b_gpu(phs_j, dw.lambda_val, z_j) - A_gpu(x_0, dw.lambda_val)
                        p_0 = r_0
                        r_old, p_old, x_old = r_0, p_0, x_0

                        for _ in range(10):
                            r_old_T_r_old = torch.sum(r_old.conj() * r_old)
                            if r_old_T_r_old.real.item() < 1e-10:
                                break
                            Ap_old = A_gpu(p_old, dw.lambda_val)
                            p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old)
                            if torch.abs(p_old_T_A_p_old) < 1e-12:
                                break
                            alpha = r_old_T_r_old / p_old_T_A_p_old
                            x_new = x_old + alpha * p_old
                            r_new = r_old - alpha * Ap_old
                            r_new_T_r_new = torch.sum(r_new.conj() * r_new)
                            beta = r_new_T_r_new / r_old_T_r_old
                            p_new = r_new + beta * p_old
                            r_old, p_old, x_old = r_new, p_new, x_new

                        solutions.append(x_old * msk_j)

                    x_k_complex = torch.cat(solutions, dim=0) * msk_b

                x_k_complex = x_k_complex * msk_b
                loss = total_loss_l1(chi=x_k_complex.real, y=sus_b, b=phs_b, d=dk_patch, m=msk_b, sobel=ss)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Test evaluation on holdout P5-8
        test_summary, test_results = evaluate_test_cohort(model_dic, K_unrolling, test_data_cache, dk_test_110, Dk_square_test_110, sus_mean, sus_std, epoch)
        elapsed = time.time() - t0

        lams = [str(round(model_dic['model_' + str(k)].lambda_val.item(), 4)) for k in range(K_unrolling)]
        logging.info('Epoch ' + f'{epoch:02d}' + '/' + str(args.epochs) + ' [' + f'{elapsed:.1f}s' + '] | TrainLoss(P' + str(args.patient) + '): ' + f'{train_loss:.5f}' + ' | ValLoss(P' + str(args.val_patient) + '): ' + f'{val_loss:.5f}' + ' || TEST(P5-8): SSIM=' + f"{test_summary['ssim_mean']:.4f}+/-{test_summary['ssim_std']:.4f}" + ', PSNR=' + f"{test_summary['psnr_mean']:.2f}dB" + ', RMSE=' + f"{test_summary['rmse_mean']:.2f}%" + ', HFEN=' + f"{test_summary['hfen_mean']:.2f}%" + ' | lambdas: ' + str(lams))

        with open(summary_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'epoch', 'train_loss', 'val_loss',
                'test_ssim_mean', 'test_ssim_std',
                'test_psnr_mean', 'test_psnr_std',
                'test_rmse_mean', 'test_rmse_std',
                'test_hfen_mean', 'test_hfen_std',
                'lambda_0', 'lambda_1', 'lambda_2', 'time_sec'
            ])
            writer.writerow({
                'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                'test_ssim_mean': test_summary['ssim_mean'],
                'test_ssim_std': test_summary['ssim_std'],
                'test_psnr_mean': test_summary['psnr_mean'],
                'test_psnr_std': test_summary['psnr_std'],
                'test_rmse_mean': test_summary['rmse_mean'],
                'test_rmse_std': test_summary['rmse_std'],
                'test_hfen_mean': test_summary['hfen_mean'],
                'test_hfen_std': test_summary['hfen_std'],
                'lambda_0': round(model_dic['model_0'].lambda_val.item(), 4),
                'lambda_1': round(model_dic['model_1'].lambda_val.item(), 4),
                'lambda_2': round(model_dic['model_2'].lambda_val.item(), 4),
                'time_sec': round(elapsed, 1)
            })

        with open(per_vol_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'patient', 'orient', 'ssim', 'psnr', 'rmse', 'hfen'])
            writer.writerows(test_results)

        save_state = {
            'K_unrolling': K_unrolling,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_summary': test_summary
        }
        for k in range(K_unrolling):
            save_state['model_' + str(k)] = model_dic['model_' + str(k)].state_dict()

        torch.save(save_state, os.path.join(output_dir, 'model_' + str(epoch) + '.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(save_state, os.path.join(output_dir, 'model_best.pt'))
            logging.info('--> Saved NEW BEST model (Val Loss: ' + f'{val_loss:.5f}' + ')')

    logging.info('=== Unshared Fine-Tuning Complete ===')

if __name__ == '__main__':
    main()
