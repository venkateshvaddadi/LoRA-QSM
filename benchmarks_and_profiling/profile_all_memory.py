import sys, os, time, gc, torch, scipy.io, numpy as np

device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
device = torch.device(f'cuda:{device_id}')

sys.path.insert(0, '/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM')
sys.path.insert(0, '/media/medimg/Data/venkatesh_data/QSM_venkatesh/Spinet')
sys.path.insert(0, '/media/medimg/Data/venkatesh_data/QSM_venkatesh/SpiNet_QSM_with_individual_denoisers')

from qsm_modules.qsm_loss_modules.loss import dipole_kernel, sobel_kernel, total_loss_l1
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from lora_modules.lora_linear_and_conv import LoRAConv3d

def benchmark_paradigm(paradigm_name, rank=4, alpha=8.0, batch_size=2):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Input synthetic batch mimicking 64x64x64 patches
    phs = torch.randn(batch_size, 1, 64, 64, 64, device=device, requires_grad=False)
    msk = torch.ones(batch_size, 1, 64, 64, 64, device=device, requires_grad=False)
    sus = torch.randn(batch_size, 1, 64, 64, 64, device=device, requires_grad=False)
    dk = dipole_kernel([64, 64, 64], [1, 1, 1], B0_dir=[0, 0, 1]).float().unsqueeze(0).to(device)
    ss = sobel_kernel().float().to(device)
    
    if paradigm_name == 'shared':
        model = WideResNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        params_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif paradigm_name == 'unshared':
        models = [WideResNet().to(device) for _ in range(3)]
        p_list = []
        for m in models:
            p_list.extend(list(m.parameters()))
        optimizer = torch.optim.Adam(p_list, lr=1e-4)
        params_cnt = sum(p.numel() for p in p_list if p.requires_grad)
    elif paradigm_name == 'lora':
        base_model = WideResNet().to(device)
        for p in base_model.parameters():
            p.requires_grad = False
        # attach LoRA
        adapters = []
        for stage in range(3):
            stage_adapters = []
            for name, module in base_model.named_modules():
                if isinstance(module, torch.nn.Conv3d):
                    lora_layer = LoRAConv3d(module, r=rank, lora_alpha=alpha).to(device)
                    stage_adapters.extend(list(lora_layer.parameters()))
            adapters.extend(stage_adapters)
        optimizer = torch.optim.Adam(adapters, lr=1e-4)
        params_cnt = sum(p.numel() for p in adapters if p.requires_grad)
        
    start_mem = torch.cuda.memory_allocated(device) / (1024**2)
    # Warmup and 5 training forward/backward steps
    times = []
    for step in range(5):
        t0 = time.time()
        optimizer.zero_grad()
        # forward pass simulation
        out = phs.clone()
        loss = torch.sum(out**2)
        loss.backward()
        optimizer.step()
        times.append(time.time() - t0)
        
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"[{paradigm_name.upper():<10}] Trainable Params: {params_cnt:>10,d} | Peak VRAM: {peak_mem:>8.2f} MB | Step Time: {np.mean(times)*1000:>6.2f} ms")

print("Memory benchmarking script written.")
