import sys, os
import torch
import numpy as np
import scipy.io as sio

# Check if GPU has memory to run 1 volume inference
device_id = 0
device = torch.device(f'cuda:{device_id}')
t = torch.cuda.get_device_properties(device).total_memory / 1e9
r = torch.cuda.memory_reserved(device) / 1e9
a = torch.cuda.memory_allocated(device) / 1e9
print(f'GPU {device_id}: Total={t:.2f}GB, Reserved={r:.2f}GB, Allocated={a:.2f}GB, Free={t-r:.2f}GB')
