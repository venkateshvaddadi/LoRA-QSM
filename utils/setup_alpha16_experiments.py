with open('train_shared_spinet_with_stage_lora.py', 'r') as f:
    code = f.read()

# Create r=4, alpha=16.0 script
code_r4 = code.replace('device_id = 0', 'device_id = 0')
code_r4 = code_r4.replace('LoRA_rank = 4', 'LoRA_rank = 4')
code_r4 = code_r4.replace('LoRA_alpha = 8.0', 'LoRA_alpha = 16.0')
code_r4 = code_r4.replace("'_stage_LoRA'", "'_stage_LoRA_r4_alpha16'")

with open('train_lora_r4_alpha16.py', 'w') as f:
    f.write(code_r4)

# Create r=8, alpha=16.0 script
code_r8 = code.replace('device_id = 0', 'device_id = 1')
code_r8 = code_r8.replace('LoRA_rank = 4', 'LoRA_rank = 8')
code_r8 = code_r8.replace('LoRA_alpha = 8.0', 'LoRA_alpha = 16.0')
code_r8 = code_r8.replace("'_stage_LoRA'", "'_stage_LoRA_r8_alpha16'")

with open('train_lora_r8_alpha16.py', 'w') as f:
    f.write(code_r8)

print('Both scripts created successfully!')
