#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second-stage training of SpiNet-QSM using stage-specific LoRA adapters.

The shared WideResNet is restored from the K=3 checkpoint and kept frozen.
Each unrolling stage has its own LoRA parameters and lambda_val.
"""

#%%

import torch
from torch.utils.data import DataLoader
from torch import nn

import numpy as np
import torch.optim as optim
import tqdm
import scipy.io
import os
import sys
import csv
import logging
from datetime import datetime

#%%

current_directory = os.path.dirname(os.path.abspath(__file__))
spinet_directory = os.path.abspath(os.path.join(current_directory, '../Spinet'))
os.chdir(current_directory)
sys.path.insert(0, spinet_directory)

from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from modules.metrics import (
    compute_hfen,
    compute_psnr,
    compute_rmse,
    compute_ssim_numpy_version_2
)

#%%

def z_real_to_z_complex(z_real):
    z_complex_recon = torch.complex(
        z_real[:, 0, :, :, :].unsqueeze(1),
        z_real[:, 1, :, :, :].unsqueeze(1)
    )
    return z_complex_recon


def z_complex_to_z_real(z_complex):
    z_real = z_complex.real
    z_imag = z_complex.imag
    z_real_recon = torch.cat([z_real, z_imag], axis=1)
    return z_real_recon

#%%

# parameters for dipole kernel
matrix_size = [64, 64, 64]
voxel_size = [1, 1, 1]

device_id = 0
restore = True

K_unrolling = 3
batch_size = 2
No_samples = 16800
epoch = 0
epsilon = 1e-5
learning_rate = 1e-4
break_amount = 1000
is_data_normalized = True
model = 'WideResNet'
MM_steps = 1

LoRA_rank = 4
LoRA_alpha = 8.0

#%%

data_source = 'given_data'
Training_patient_no = 1
data_source_no = 1
test_orientations = range(1, 6)
test_matrix_size = [176, 176, 160]
save_test_reconstructions = False

if data_source == 'given_data':
    if data_source_no == 1:
        csv_path = '../QSM_data/data_for_experiments/given_data/data_source_1/'
    elif data_source_no == 2:
        csv_path = '../QSM_data/data_for_experiments/given_data/data_source_2/'
    elif data_source_no == 3:
        csv_path = '../QSM_data/data_for_experiments/given_data/data_source_3/'
    elif data_source_no == 4:
        csv_path = '../QSM_data/data_for_experiments/given_data/data_source_4/'
    else:
        raise ValueError('data_source_no must be between 1 and 4')
    data_path = '../QSM_data/data_for_experiments/given_data/data_as_patches/'
    raw_data_path = '../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
else:
    raise ValueError('This second-stage script is configured for given_data')

data_distribution = scipy.io.loadmat(
    csv_path + 'csv_files/data_distribution.mat'
)
test_patients_key = 'test_' + str(data_source_no) + '_files'
if test_patients_key not in data_distribution:
    raise KeyError('Test patient list not found: ' + test_patients_key)
test_patients = [
    int(patient_no)
    for patient_no in data_distribution[test_patients_key].ravel()
]

print('csv_path:', csv_path)
print('data_path:', data_path)
print('test_patients:', test_patients)

#%%

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for this training script')

device = torch.device('cuda:' + str(device_id))
restore_weights_path = (
    '../Spinet/savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/'
    'experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/'
    'Jul_10_12_15_pm_model_K_3_given_data_dw_WideResNet_data_source_1/'
    'Spinet_QSM_model_14_.pth'
)

#%%

loader = mydataloader(csv_path + 'csv_files/train.csv', data_path)
trainloader = DataLoader(
    loader,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=True
)

valdata = mydataloader(csv_path + 'csv_files/val.csv', data_path)
valloader = DataLoader(
    valdata,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=True
)

if len(trainloader) == 0 or len(valloader) == 0:
    raise ValueError('Training and validation data loaders must not be empty')

for patient_no in test_patients:
    for orientation in test_orientations:
        for data_name in ('phs', 'cos', 'msk'):
            test_data_path = (
                raw_data_path
                + 'patient_'
                + str(patient_no)
                + '/'
                + data_name
                + str(orientation)
                + '.mat'
            )
            if not os.path.exists(test_data_path):
                raise FileNotFoundError('Test data not found: ' + test_data_path)

#%%

if not is_data_normalized:
    sus_mean = 0
    sus_std = 1
    print('\n\n data is not normalized..................\n\n ')
else:
    stats = scipy.io.loadmat(csv_path + 'csv_files/tr-stats.mat')
    sus_mean = torch.tensor(stats['out_mean']).to(device)
    sus_std = torch.tensor(stats['out_std']).to(device)
    if torch.any(sus_std == 0):
        raise ValueError('Training susceptibility standard deviation contains zero')
    print(sus_mean, sus_std)

#%%

experiments_folder = (
    'savedModels/Spinet_QSM_denoisers_with_stage_lora/'
    'experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/'
)
experiment_name = (
    datetime.now().strftime('%b_%d_%I_%M_%P_')
    + 'model_K_'
    + str(K_unrolling)
    + '_'
    + data_source
    + '_dw_'
    + model
    + '_stage_LoRA'
)
directory = experiments_folder + experiment_name + '/'

print('*******************************************************')
print('Model will be saved to:', directory)
os.makedirs(directory, exist_ok=False)

logging.basicConfig(
    filename=directory + 'app.log',
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
logging.warning('BATCH SIZE:' + str(batch_size))
logging.warning('K_UNROLLING:' + str(K_unrolling))
logging.warning('restore:' + str(restore))
logging.warning('No training samples:' + str(batch_size * len(trainloader)))
logging.warning('No validation samples:' + str(batch_size * len(valloader)))
logging.warning('learning_rate:' + str(learning_rate))
logging.warning('data_normalized:' + str(is_data_normalized))
logging.warning('data_source:' + str(data_source))
logging.warning('data_path:' + data_path)
logging.warning('csv_path:' + csv_path)
logging.warning('MMsteps: ' + str(MM_steps))
logging.warning('LoRA rank: ' + str(LoRA_rank))
logging.warning('LoRA alpha: ' + str(LoRA_alpha))

#%%

dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = dk.float()
dk = torch.unsqueeze(dk, dim=0)
dk = dk.to(device)
Dk_square = dk * dk
dk_test = dipole_kernel(test_matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk_test = torch.unsqueeze(dk_test, dim=0).float().to(device)
Dk_square_test = dk_test * dk_test

print('dk.shape', dk.shape)
print('Dk_square.shape', Dk_square.shape)
print('dk_test.shape', dk_test.shape)
print('Dk_square_test.shape', Dk_square_test.shape)

#%%

class LoRA_Conv3d(nn.Module):
    def __init__(self, base_conv, rank, alpha):
        super(LoRA_Conv3d, self).__init__()
        if base_conv.groups != 1:
            raise ValueError('LoRA_Conv3d requires Conv3d groups=1')

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
        self.lambda_val = nn.Parameter(initial_lambda.detach().clone())
        self.register_buffer('p', p.detach().clone())

#%%

dw = WideResNet().to(device)

if restore:
    if not os.path.exists(restore_weights_path):
        raise FileNotFoundError('Shared checkpoint not found: ' + restore_weights_path)
    dw.load_state_dict(torch.load(restore_weights_path, map_location=device), strict=True)
    print('we have restored shared weights of the model:', restore_weights_path)
    logging.warning('restore_weights_path:' + restore_weights_path)

for parameter in dw.parameters():
    parameter.requires_grad = False

print('p', dw.p)
print('lambda_val', dw.lambda_val)

#%%

# Create stage-specific LoRA modules. The shared CNN itself is only created once.
model_dic = {}
active_stage = None
convolution_list = [
    module for module in dw.modules() if isinstance(module, nn.Conv3d)
]

if len(convolution_list) == 0:
    raise ValueError('No Conv3d layers found in the shared denoiser')

for i in range(K_unrolling):
    model_dic['model_' + str(i)] = Stage_LoRA(
        convolution_list,
        dw.lambda_val,
        dw.p
    ).to(device)

    for conv_i, conv in enumerate(convolution_list):
        stage_lora = model_dic['model_' + str(i)].lora_layers[conv_i]

        def add_stage_lora(module, inputs, output, stage_no=i, lora=stage_lora):
            if active_stage == stage_no:
                return output + lora(inputs[0])
            return output

        conv.register_forward_hook(add_stage_lora)

print('Restored shared model:', restore)
for i in range(K_unrolling):
    dw_stage = model_dic['model_' + str(i)]
    print('stage:', i, 'lambda_val:', dw_stage.lambda_val)
    print('stage:', i, 'p:', dw_stage.p)

#%%

def dw_with_stage_lora(x_input, stage_no):
    global active_stage
    active_stage = stage_no
    try:
        output = dw(x_input)
    finally:
        active_stage = None
    return output


def save_model_based_deep_learning_model(
    directory,
    epoch_no,
    training_loss,
    validation_loss
):
    MoDL_state = {}
    MoDL_state['K_unrolling'] = K_unrolling
    MoDL_state['epoch_no'] = epoch_no
    MoDL_state['training_loss'] = training_loss
    MoDL_state['validation_loss'] = validation_loss
    MoDL_state['shared_denoiser'] = dw.state_dict()
    MoDL_state['LoRA_rank'] = LoRA_rank
    MoDL_state['LoRA_alpha'] = LoRA_alpha

    for i in range(K_unrolling):
        dw_stage = model_dic['model_' + str(i)]
        MoDL_state['model_' + str(i)] = dw_stage.state_dict()

    output_path = directory + 'Spinet_QSM_model_' + str(epoch_no) + '_.pth'
    torch.save(MoDL_state, output_path)

#%%

print('----------------------------------------------------------------')
print('\n Experiment_name:')
print(directory)
print('----------------------------------------------------------')
print('K_unrolling:', K_unrolling)
print('batch_size:', batch_size)
print('No_samples:', No_samples)
print('learning_rate:', learning_rate)
print('device_id:', device_id)
print('no of training batches:', len(trainloader))
print('no of validation batches:', len(valloader))

ss = sobel_kernel()
ss = ss.float()
ss = ss.to(device)

dw_paramters_dic_list = []
for i in range(K_unrolling):
    dw_stage = model_dic['model_' + str(i)]
    individual_model_param_dic = {'params': dw_stage.parameters()}
    dw_paramters_dic_list.append(individual_model_param_dic)

optimizer = optim.Adam(dw_paramters_dic_list, lr=learning_rate)

loss_Train = []
loss_Val = []
test_SSIM = []
test_PSNR = []
test_RMSE = []
test_HFEN = []
lambda_list = []
p_list = []

#%%

def b_gpu(y, lambda_val, z_k, w_square, dk_kernel=None):
    if dk_kernel is None:
        dk_kernel = dk
    output1 = torch.fft.fftn(y)
    output2 = dk_kernel * output1
    output3 = torch.fft.ifftn(output2)
    output4 = output3 + lambda_val * w_square * z_k
    return output4


def A_gpu(x, lambda_val, w_square, Dk_square_kernel=None):
    if Dk_square_kernel is None:
        Dk_square_kernel = Dk_square
    output1 = Dk_square_kernel * torch.fft.fftn(x)
    output2 = torch.fft.ifftn(output1)
    output3 = output2 + lambda_val * w_square * x
    return output3


def CG_GPU_for_given_unrolling_k(
    local_field_gpu,
    z_k_gpu,
    given_unrolling_k,
    w_square,
    dk_kernel=None,
    Dk_square_kernel=None
):
    dw_stage = model_dic['model_' + str(given_unrolling_k)]
    if dk_kernel is None:
        dk_kernel = dk
    if Dk_square_kernel is None:
        Dk_square_kernel = Dk_square
    x_0 = torch.zeros_like(local_field_gpu, dtype=torch.float64)

    r_0 = b_gpu(
        local_field_gpu,
        dw_stage.lambda_val,
        z_k_gpu,
        w_square,
        dk_kernel
    ) - A_gpu(x_0, dw_stage.lambda_val, w_square, Dk_square_kernel)

    r_old = r_0
    p_old = r_0
    x_old = x_0
    r_stat = [torch.sum(r_old.conj() * r_old).real.item()]

    for i in range(30):
        r_old_T_r_old = torch.sum(r_old.conj() * r_old)

        if r_old_T_r_old.real.item() < 1e-10:
            return x_old

        if r_old_T_r_old.real.item() > r_stat[-1] and r_stat[-1] < 1e-6:
            return x_old

        r_stat.append(torch.sum(r_old.conj() * r_old).real.item())
        Ap_old = A_gpu(
            p_old,
            dw_stage.lambda_val,
            w_square,
            Dk_square_kernel
        )
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


def test_model_epoch(epoch_no):
    test_results = []
    predictions_directory = directory + 'predictions_' + str(epoch_no) + '/'
    if save_test_reconstructions:
        os.makedirs(predictions_directory, exist_ok=True)

    dw.eval()
    for stage_no in range(K_unrolling):
        model_dic['model_' + str(stage_no)].eval()

    with torch.no_grad():
        for patient_no in test_patients:
            print('Testing patient:', patient_no)
            for orientation in test_orientations:
                phs_path = (
                    raw_data_path
                    + 'patient_'
                    + str(patient_no)
                    + '/phs'
                    + str(orientation)
                    + '.mat'
                )
                sus_path = (
                    raw_data_path
                    + 'patient_'
                    + str(patient_no)
                    + '/cos'
                    + str(orientation)
                    + '.mat'
                )
                msk_path = (
                    raw_data_path
                    + 'patient_'
                    + str(patient_no)
                    + '/msk'
                    + str(orientation)
                    + '.mat'
                )

                phs = torch.tensor(scipy.io.loadmat(phs_path)['phs'])
                sus = torch.tensor(scipy.io.loadmat(sus_path)['cos'])
                msk = torch.tensor(scipy.io.loadmat(msk_path)['msk'])

                phs = phs.unsqueeze(0).unsqueeze(0).float().to(device)
                sus = sus.unsqueeze(0).unsqueeze(0).float().to(device)
                msk = msk.unsqueeze(0).unsqueeze(0).float().to(device)

                if list(phs.shape[-3:]) != test_matrix_size:
                    raise ValueError(
                        'Unexpected test volume size for patient '
                        + str(patient_no)
                        + ', orientation '
                        + str(orientation)
                        + ': '
                        + str(list(phs.shape[-3:]))
                    )

                phs_F = torch.fft.fftn(phs, dim=[2, 3, 4])
                x_k_complex = torch.fft.ifftn(
                    dk_test * phs_F,
                    dim=[2, 3, 4]
                )

                for stage_no in range(K_unrolling):
                    x_k_complex = x_k_complex * msk
                    x_k_real = z_complex_to_z_real(x_k_complex).float()
                    x_k_real = (x_k_real - sus_mean) / sus_std
                    z_k_real = dw_with_stage_lora(x_k_real, stage_no)
                    z_k_real = z_k_real * sus_std + sus_mean
                    z_k_complex = z_real_to_z_complex(z_k_real)

                    for mm_steps in range(MM_steps):
                        exponent = (
                            model_dic['model_' + str(stage_no)].p / 2 - 1.0
                        )
                        if torch.all(exponent == 0).item():
                            w_square = torch.ones_like(x_k_complex)
                        else:
                            w = torch.pow(x_k_complex - z_k_complex, exponent)
                            w_square = w.conj() * w

                        x_k_complex = CG_GPU_for_given_unrolling_k(
                            phs,
                            z_k_complex,
                            stage_no,
                            w_square,
                            dk_test,
                            Dk_square_test
                        )
                        x_k_complex = x_k_complex * msk

                sus_cal_cpu = (
                    x_k_complex.real.detach().cpu().numpy()
                    * msk.detach().cpu().numpy()
                )
                sus_given_cpu = sus.detach().cpu().numpy()
                sus_cal_cpu_3d = np.squeeze(sus_cal_cpu)
                sus_given_cpu_3d = np.squeeze(sus_given_cpu)

                test_result = {
                    'epoch': epoch_no,
                    'patient': patient_no,
                    'orientation': orientation,
                    'ssim': float(compute_ssim_numpy_version_2(
                        sus_cal_cpu_3d,
                        sus_given_cpu_3d
                    )),
                    'psnr': float(compute_psnr(sus_cal_cpu, sus_given_cpu)),
                    'rmse': float(compute_rmse(sus_cal_cpu, sus_given_cpu)),
                    'hfen': float(compute_hfen(sus_cal_cpu, sus_given_cpu))
                }
                test_results.append(test_result)
                print(test_result)

                if save_test_reconstructions:
                    filename = (
                        predictions_directory
                        + 'modl-net-'
                        + str(patient_no)
                        + '-'
                        + str(orientation)
                        + '.mat'
                    )
                    scipy.io.savemat(filename, {'modl': sus_cal_cpu})

    test_results_path = directory + 'test_metrics.csv'
    write_header = not os.path.exists(test_results_path)
    with open(test_results_path, 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'epoch',
                'patient',
                'orientation',
                'ssim',
                'psnr',
                'rmse',
                'hfen'
            ]
        )
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerows(test_results)

    mean_metrics = {
        metric_name: float(np.mean([
            result[metric_name] for result in test_results
        ]))
        for metric_name in ('ssim', 'psnr', 'rmse', 'hfen')
    }
    print('Test mean metrics:', mean_metrics)
    return mean_metrics

#%%

epoch = 0
for epoch in range(60):
    runningLoss = 0

    print('\n-------------------------------------------------------------------------------------------------\n')
    print('epoch---', epoch)

    dw.train()
    for module in dw.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()
    for i in range(K_unrolling):
        model_dic['model_' + str(i)].train()

    for i, data in tqdm.tqdm(enumerate(trainloader)):
        phs, msk, sus, file_name = data
        dk_repeat = dk.repeat(batch_size, 1, 1, 1, 1)

        phs = phs.to(device)
        msk = msk.to(device)
        sus = sus.to(device)

        phs_F = torch.fft.fftn(phs, dim=[2, 3, 4])
        phs_F = dk_repeat * phs_F
        x_0_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4])

        x_k_complex = x_0_complex

        for k in range(K_unrolling):
            x_k_complex = x_k_complex * msk
            x_k_real = z_complex_to_z_real(x_k_complex).float()
            x_k_real = (x_k_real - sus_mean) / sus_std

            z_k_real = dw_with_stage_lora(x_k_real, k)
            z_k_real = z_k_real * sus_std + sus_mean
            z_k_complex = z_real_to_z_complex(z_k_real)

            solutions = []
            for j in range(batch_size):
                x_k_j_complex = x_k_complex[j, :, :, :, :].unsqueeze(0)
                z_k_j_complex = z_k_complex[j, :, :, :, :].unsqueeze(0)

                for mm_steps in range(MM_steps):
                    exponent = model_dic['model_' + str(k)].p / 2 - 1.0
                    if torch.all(exponent == 0).item():
                        w_square = torch.ones_like(x_k_j_complex)
                    else:
                        w = torch.pow(x_k_j_complex - z_k_j_complex, exponent)
                        w_square = w.conj() * w

                    x_k_j_complex = CG_GPU_for_given_unrolling_k(
                        phs[j, :, :, :, :].unsqueeze(0),
                        z_k_j_complex,
                        k,
                        w_square
                    )
                    x_k_j_complex = x_k_j_complex * msk[j:j + 1, :, :, :, :]

                solutions.append(x_k_j_complex)

            x_k_complex = torch.cat(solutions, dim=0)
            x_k_complex = x_k_complex * msk

        x_k_complex = x_k_complex * msk

        optimizer.zero_grad()
        loss = total_loss_l1(
            chi=x_k_complex.real,
            y=sus,
            b=phs,
            d=dk,
            m=msk,
            sobel=ss
        )
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()

    training_loss = runningLoss / len(trainloader)
    loss_Train.append(training_loss)
    print('Training_loss:', loss_Train)

    #%%

    running_val_Loss = 0
    dw.eval()
    for i in range(K_unrolling):
        model_dic['model_' + str(i)].eval()

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(valloader)):
            phs, msk, sus, file_name = data
            dk_repeat = dk.repeat(batch_size, 1, 1, 1, 1)

            phs = phs.to(device)
            msk = msk.to(device)
            sus = sus.to(device)

            phs_F = torch.fft.fftn(phs, dim=[2, 3, 4])
            phs_F = dk_repeat * phs_F
            x_0_complex = torch.fft.ifftn(phs_F, dim=[2, 3, 4])
            x_k_complex = x_0_complex

            for k in range(K_unrolling):
                x_k_complex = x_k_complex * msk
                x_k_real = z_complex_to_z_real(x_k_complex).float()
                x_k_real = (x_k_real - sus_mean) / sus_std

                z_k_real = dw_with_stage_lora(x_k_real, k)
                z_k_real = z_k_real * sus_std + sus_mean
                z_k_complex = z_real_to_z_complex(z_k_real)

                solutions = []
                for j in range(batch_size):
                    x_k_j_complex = x_k_complex[j, :, :, :, :].unsqueeze(0)
                    z_k_j_complex = z_k_complex[j, :, :, :, :].unsqueeze(0)

                    for mm_steps in range(MM_steps):
                        exponent = model_dic['model_' + str(k)].p / 2 - 1.0
                        if torch.all(exponent == 0).item():
                            w_square = torch.ones_like(x_k_j_complex)
                        else:
                            w = torch.pow(x_k_j_complex - z_k_j_complex, exponent)
                            w_square = w.conj() * w

                        x_k_j_complex = CG_GPU_for_given_unrolling_k(
                            phs[j, :, :, :, :].unsqueeze(0),
                            z_k_j_complex,
                            k,
                            w_square
                        )
                        x_k_j_complex = x_k_j_complex * msk[j:j + 1, :, :, :, :]

                    solutions.append(x_k_j_complex)

                x_k_complex = torch.cat(solutions, dim=0)
                x_k_complex = x_k_complex * msk

            x_k_complex = x_k_complex * msk
            loss = total_loss_l1(
                chi=x_k_complex.real,
                y=sus,
                b=phs,
                d=dk,
                m=msk,
                sobel=ss
            )
            running_val_Loss += loss.item()

    validation_loss = running_val_Loss / len(valloader)
    loss_Val.append(validation_loss)
    print('Validation_loss:', loss_Val)
    print('Checking what is the K value:', K_unrolling)

    test_metrics = test_model_epoch(epoch)
    test_SSIM.append(test_metrics['ssim'])
    test_PSNR.append(test_metrics['psnr'])
    test_RMSE.append(test_metrics['rmse'])
    test_HFEN.append(test_metrics['hfen'])

    lambda_epoch = []
    p_epoch = []
    for i in range(K_unrolling):
        dw_stage = model_dic['model_' + str(i)]
        lambda_epoch.append(dw_stage.lambda_val.item())
        p_epoch.append(dw_stage.p.item())
    lambda_list.append(lambda_epoch)
    p_list.append(p_epoch)

    model_details = {
        'train_loss': loss_Train,
        'valid_loss': loss_Val,
        'test_ssim': test_SSIM,
        'test_psnr': test_PSNR,
        'test_rmse': test_RMSE,
        'test_hfen': test_HFEN,
        'lambda': lambda_list,
        'p': p_list
    }

    with open(directory + 'model_details.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch',
            'train_loss',
            'valid_loss',
            'test_ssim',
            'test_psnr',
            'test_rmse',
            'test_hfen',
            'lambda_stage_0',
            'lambda_stage_1',
            'lambda_stage_2'
        ])
        for epoch_i in range(len(loss_Train)):
            csv_writer.writerow([
                epoch_i,
                loss_Train[epoch_i],
                loss_Val[epoch_i],
                test_SSIM[epoch_i],
                test_PSNR[epoch_i],
                test_RMSE[epoch_i],
                test_HFEN[epoch_i],
                lambda_list[epoch_i][0],
                lambda_list[epoch_i][1],
                lambda_list[epoch_i][2]
            ])

    scipy.io.savemat(directory + 'model_details.mat', model_details)
    save_model_based_deep_learning_model(
        directory,
        epoch,
        training_loss,
        validation_loss
    )

    print('\n-------------------------------------------------------------------------------------------------\n')

#%%
