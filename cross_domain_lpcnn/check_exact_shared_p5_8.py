import os, sys, scipy.io, numpy as np, torch
sys.path.insert(0, '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet')
from qsm_modules.qsm_loss_modules.loss import dipole_kernel
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from modules.metrics import compute_hfen, compute_psnr, compute_rmse, compute_ssim_numpy_version_2

device = torch.device('cuda:0')
ckpt = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet/savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Jul_10_12_15_pm_model_K_3_given_data_dw_WideResNet_data_source_1/Spinet_QSM_model_14_.pth'
stats = scipy.io.loadmat('/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/patient_2/tr-stats.mat')
sus_mean = torch.tensor(stats['out_mean']).float().to(device)
sus_std  = torch.tensor(stats['out_std']).float().to(device)

dw = WideResNet().to(device)
dw.load_state_dict(torch.load(ckpt, map_location=device))
dw.eval()

dk_110 = dipole_kernel([224, 224, 110], [1, 1, 1], B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
Dk_sq_110 = dk_110 * dk_110

def CG_GPU_stage(local_field_gpu, z_k_gpu, w_square):
    x_0 = torch.zeros_like(local_field_gpu, dtype=torch.float64)
    out1 = torch.fft.fftn(local_field_gpu) * dk_110
    out3 = torch.fft.ifftn(out1) + dw.lambda_val * w_square * z_k_gpu
    r_0 = out3 - (torch.fft.ifftn(Dk_sq_110 * torch.fft.fftn(x_0)) + dw.lambda_val * w_square * x_0)
    r_old, p_old, x_old = r_0, r_0, x_0
    r_stat = [torch.sum(r_old.conj() * r_old).real.item()]
    for _ in range(30):
        r_old_T_r_old = torch.sum(r_old.conj() * r_old)
        if r_old_T_r_old.real.item() < 1e-10:
            return x_old
        if r_old_T_r_old.real.item() > r_stat[-1] and r_stat[-1] < 1e-6:
            return x_old
        r_stat.append(torch.sum(r_old.conj() * r_old).real.item())
        Ap_old = torch.fft.ifftn(Dk_sq_110 * torch.fft.fftn(p_old)) + dw.lambda_val * w_square * p_old
        alpha = r_old_T_r_old / torch.sum(p_old.conj() * Ap_old)
        x_new = x_old + alpha * p_old
        r_new = r_old - alpha * Ap_old
        beta = torch.sum(r_new.conj() * r_new) / r_old_T_r_old
        p_new = r_new + beta * p_old
        r_old, p_old, x_old = r_new, p_new, x_new
    return x_old

results = []
raw_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/'
with torch.no_grad():
    for p in [5, 6, 7, 8]:
        for o in [1, 2, 3, 4]:
            p_dir = os.path.join(raw_path, f'patient_{p}')
            phs = scipy.io.loadmat(os.path.join(p_dir, f'phs{o}.mat'))['phs']
            cos = scipy.io.loadmat(os.path.join(p_dir, f'cos{o}.mat'))['cos']
            msk = scipy.io.loadmat(os.path.join(p_dir, f'msk{o}.mat'))['msk']

            phs_t = torch.tensor(phs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            msk_t = torch.tensor(msk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            phs_F = dk_110 * torch.fft.fftn(phs_t, dim=[2, 3, 4])
            x_k_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4]) * msk_t

            for k in range(3):
                x_k_complex = x_k_complex * msk_t
                x_k_real = torch.cat([x_k_complex.real, x_k_complex.imag], dim=1).float()
                x_k_real = (x_k_real - sus_mean) / sus_std
                z_k_real = dw(x_k_real) * sus_std + sus_mean
                z_k_complex = torch.complex(z_k_real[:, 0:1], z_k_real[:, 1:2])
                w_sq = torch.ones_like(x_k_complex)
                cg_out = CG_GPU_stage(phs_t, z_k_complex, w_sq) * msk_t
                x_k_complex = cg_out

            recon = (x_k_complex.real * msk_t).squeeze().cpu().numpy().astype(np.float32) * msk.astype(np.float32)
            cos_crop = (cos * msk).astype(np.float32)

            ssim = float(compute_ssim_numpy_version_2(recon, cos_crop))
            psnr = float(compute_psnr(recon, cos_crop))
            rmse = float(compute_rmse(recon, cos_crop))
            hfen = float(compute_hfen(recon, cos_crop))
            results.append({'patient': p, 'orient': o, 'ssim': ssim, 'psnr': psnr, 'rmse': rmse, 'hfen': hfen})

print('Metrics across Patients 5-8 (16 volumes):')
for k in ['ssim', 'psnr', 'rmse', 'hfen']:
    vals = [r[k] for r in results]
    print(f'{k.upper()}: Mean = {np.mean(vals):.4f}, Std = {np.std(vals):.4f}')
