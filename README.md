# LoRA-QSM: Two-Stage Learning Paradigm for Model-Based QSM Reconstruction

PyTorch implementation for stage-specific low-rank adaptation in unrolled Quantitative Susceptibility Mapping.

## Overview
- **Stage 1 (Foundation)**: Unrolled 3D CNN denoiser with shared weights and scalar regularizer.
- **Stage 2 (LoRA Adaptation)**: Frozen backbone with stage-specific low-rank updates and individual regularizers.

## Code Structure
- `qsm_modules/`: Core denoisers, dipole physics operator, and loss functions.
- `stage1_foundation/`: Stage 1 baseline pre-training.
- `stage2_lora_adaptation/`: Stage 2 in-domain low-rank adaptation (Table II).
- `cross_domain_lpcnn/`: LPCNN zero-shot transfer (Table III) and single-subject calibration (Table IV).
- `benchmarks_and_profiling/`: GPU memory and latency profiling (Table V).
- `figures_generation/`: Scripts for plotting reconstruction figures and loss curves.

## Installation
```bash
git clone https://github.com/venkateshvaddadi/LoRA-QSM.git
cd LoRA-QSM
pip install -r requirements.txt
```
