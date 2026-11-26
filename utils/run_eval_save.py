import sys, os
import torch
import numpy as np

sys.argv = ['train_target_adapt_configurable.py', '--patient', '2', '--device_id', '0']

import train_target_adapt_configurable as ttac
from train_target_adapt_configurable import (
    dw, model_dic, convolution_list, device,
    test_cache
)

calib_ckpt_path = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/target_adapt_lpcnn_p2_val_p4_r4_a8/best_val_model.pth'
ckpt = torch.load(calib_ckpt_path, map_location=device)
for k in range(3):
    model_dic[f'model_{k}'].load_state_dict(ckpt[f'model_{k}'])
    model_dic[f'model_{k}'].eval()

print("Loaded calibrated weights! Running single volume inference for Patient 6 Orientation 1...")

p6_data = [item for item in test_cache if item['patient'] == 6 and item['orientation'] == 1][0]
phs = p6_data['phs_batch'].to(device)
msk = p6_data['msk_batch'].to(device)
dk = p6_data['dk'].to(device)
dk_sq = p6_data['Dk_sq'].to(device)
cos_raw = p6_data['cos_raw']
msk_raw = p6_data['msk_raw']

with torch.no_grad():
    recon_calib = ttac.forward_unrolled_pass(phs, msk, dk, dk_sq)
    recon_calib = recon_calib.squeeze().cpu().numpy()

print("Calibrated recon shape:", recon_calib.shape)
np.save('/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/recon_p6_calib.npy', recon_calib)
np.save('/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/cos_p6.npy', cos_raw)
np.save('/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/msk_p6.npy', msk_raw)

print("Running baseline zero-shot pass...")
for k in range(3):
    for layer in model_dic[f'model_{k}'].lora_layers:
        layer.scale = 0.0

with torch.no_grad():
    recon_base = ttac.forward_unrolled_pass(phs, msk, dk, dk_sq)
    recon_base = recon_base.squeeze().cpu().numpy()

np.save('/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/recon_p6_base.npy', recon_base)
print("All volumes saved successfully!")
