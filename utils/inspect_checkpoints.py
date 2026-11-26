import os, glob

base_ckpt = '../Spinet/savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Jul_10_12_15_pm_model_K_3_given_data_dw_WideResNet_data_source_1/Spinet_QSM_model_14_.pth'
print('Base exists:', os.path.exists(base_ckpt))

r4_dir = 'savedModels/Spinet_QSM_denoisers_with_stage_lora/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Nov_22_11_18_am_model_K_3_given_data_dw_WideResNet_stage_LoRA/'
r8_dir = 'savedModels/Spinet_QSM_denoisers_with_stage_lora/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Nov_22_11_49_am_model_K_3_given_data_dw_WideResNet_stage_LoRA/'

print('r4 ckpts:', sorted([os.path.basename(f) for f in glob.glob(r4_dir + '*.pth')]))
print('r8 ckpts:', sorted([os.path.basename(f) for f in glob.glob(r8_dir + '*.pth')]))
