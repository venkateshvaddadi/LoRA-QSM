# Pre-trained Checkpoints for LoRA-QSM

This directory outlines the pre-trained checkpoints corresponding to each training phase in the manuscript:

### 1. Stage 1 Foundation Prior (Shared Weights Baseline)
- **Path**: `Spinet_QSM_model_14_.pth`
- **Architecture**: 3D WideResNet (20 Conv layers, 447,234 parameters)
- **Training**: 30 epochs on source SNU 3T multi-orientation benchmark ($K=3$ stages)
- **Learned Regularizer**: $\lambda = 0.1393$, $p = 2.0$

### 2. Stage 2 In-Domain LoRA Adapted Models (Table II)
- **$r=4, \alpha=8.0$**: Model checkpoint evaluated on SNU 30 test volumes (SSIM: 0.9136, PSNR: 41.32 dB)
- **$r=4, \alpha=16.0$**: Epoch 21 peak PSNR checkpoint (PSNR: 41.34 dB)
- **$r=8, \alpha=16.0$**: Epoch 18 peak all-time structural similarity (SSIM: 0.9145, PSNR: 41.35 dB)

### 3. Stage 2 Target-Calibrated Models on LPCNN (Table IV)
- **Subject 2 Calibrated ($r=8, \alpha=8.0$)**: Minimum validation loss checkpoint selected on independent Subject 4 (Holdout RMSE: 51.42%, SSIM: 0.9448)
- **Subject 2 Calibrated ($r=4, \alpha=8.0$)**: Compact rank configuration (Holdout RMSE: 51.48%, SSIM: 0.9446)
- **Subject 1 Calibrated ($r=8, \alpha=8.0$)**: (Holdout RMSE: 53.18%)

*Note: Large `.pth` model binaries are hosted externally to adhere to GitHub file size limits. You can download the pre-trained checkpoints from the Zenodo / Google Drive repository (link to be provided upon paper publication).*
