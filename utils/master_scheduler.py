import subprocess
import time
import os
import threading

# Configuration: Remaining 10 experiments out of the 12 grid (10 epochs each)
gpu0_queue = [
    {"patient": 3, "val_patient": 4, "rank": 4, "alpha": 16.0, "epochs": 10},
    {"patient": 1, "val_patient": 4, "rank": 4, "alpha": 8.0, "epochs": 10},
    {"patient": 3, "val_patient": 4, "rank": 4, "alpha": 8.0, "epochs": 10},
    {"patient": 2, "val_patient": 4, "rank": 8, "alpha": 16.0, "epochs": 10},
    {"patient": 1, "val_patient": 4, "rank": 8, "alpha": 8.0, "epochs": 10},
]

gpu1_queue = [
    {"patient": 2, "val_patient": 4, "rank": 4, "alpha": 8.0, "epochs": 10},
    {"patient": 1, "val_patient": 4, "rank": 8, "alpha": 16.0, "epochs": 10},
    {"patient": 3, "val_patient": 4, "rank": 8, "alpha": 16.0, "epochs": 10},
    {"patient": 2, "val_patient": 4, "rank": 8, "alpha": 8.0, "epochs": 10},
    {"patient": 3, "val_patient": 4, "rank": 8, "alpha": 8.0, "epochs": 10},
]

PYTHON_EXEC = "/home/medimg/miniconda3/envs/torch_pip/bin/python"
SCRIPT_PATH = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/train_target_adapt_configurable.py"
LOG_DIR = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/LORA_QSM/savedModels/scheduler_logs"

os.makedirs(LOG_DIR, exist_ok=True)

def worker(gpu_id, job_queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    for job in job_queue:
        p = job["patient"]
        vp = job["val_patient"]
        r = job["rank"]
        a = int(job["alpha"])
        tag = f"p{p}_val_p{vp}_r{r}_a{a}"
        log_file = os.path.join(LOG_DIR, f"{tag}_gpu{gpu_id}.log")
        
        cmd = [
            PYTHON_EXEC,
            SCRIPT_PATH,
            "--patient", str(job["patient"]),
            "--val_patient", str(job["val_patient"]),
            "--rank", str(job["rank"]),
            "--alpha", str(job["alpha"]),
            "--epochs", str(job["epochs"]),
            "--device_id", "0"
        ]
        
        print(f"[GPU {gpu_id}] Starting job: {tag}", flush=True)
        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            ret = proc.wait()
            
        print(f"[GPU {gpu_id}] Finished job: {tag} with exit code {ret}", flush=True)
        time.sleep(5)

if __name__ == "__main__":
    print("Master Scheduler Starting across GPU 0 and GPU 1...", flush=True)
    t0 = threading.Thread(target=worker, args=(0, gpu0_queue))
    t1 = threading.Thread(target=worker, args=(1, gpu1_queue))
    
    t0.start()
    t1.start()
    
    t0.join()
    t1.join()
    print("All queued calibration experiments completed successfully!", flush=True)
