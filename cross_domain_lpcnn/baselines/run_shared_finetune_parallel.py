import subprocess
import time
import os
import threading

PYTHON_EXEC = "/home/medimg/miniconda3/envs/torch_pip/bin/python"
SCRIPT_PATH = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/train_target_adapt_shared_base.py"
LOG_DIR = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/shared_finetune_logs"

os.makedirs(LOG_DIR, exist_ok=True)

gpu0_jobs = [
    {"patient": 1, "val_patient": 4, "epochs": 10},
    {"patient": 3, "val_patient": 4, "epochs": 10},
]

gpu1_jobs = [
    {"patient": 2, "val_patient": 4, "epochs": 10},
]

def worker(gpu_id, job_queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    for job in job_queue:
        p = job["patient"]
        vp = job["val_patient"]
        epochs = job["epochs"]
        tag = f"shared_finetune_p{p}_val_p{vp}"
        log_file = os.path.join(LOG_DIR, f"{tag}_gpu{gpu_id}.log")
        
        cmd = [
            PYTHON_EXEC,
            SCRIPT_PATH,
            "--patient", str(p),
            "--val_patient", str(vp),
            "--epochs", str(epochs),
            "--device_id", "0"
        ]
        
        print(f"[GPU {gpu_id}] Starting: {tag}", flush=True)
        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            ret = proc.wait()
            
        print(f"[GPU {gpu_id}] Completed: {tag} with exit code {ret}", flush=True)
        time.sleep(5)

if __name__ == "__main__":
    print("Launching Shared Model Fine-Tuning across GPU 0 and GPU 1...", flush=True)
    t0 = threading.Thread(target=worker, args=(0, gpu0_jobs))
    t1 = threading.Thread(target=worker, args=(1, gpu1_jobs))
    
    t0.start()
    t1.start()
    
    t0.join()
    t1.join()
    print("All shared model fine-tuning experiments completed!", flush=True)
