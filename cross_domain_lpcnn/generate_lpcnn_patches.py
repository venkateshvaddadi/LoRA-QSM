#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import time
import os
import sys
import pandas as pd

patient_list = list(range(1, 9))
orientation = [1, 2, 3, 4]

base_data_path = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/"
base_out_path  = "/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/"
out_patches_dir = os.path.join(base_out_path, "data_as_patches/")

os.makedirs(out_patches_dir, exist_ok=True)

PS = 64
patch_num = [6, 8, 7]

start_time = time.time()
print("#### LPCNN PATCH EXTRACTION STARTED ####")
print("Target Patch Directory:", out_patches_dir)

patch_info_list = []

for dataset_num in patient_list:
    t_p = time.time()
    data_path = os.path.join(base_data_path, f"patient_{dataset_num}/")
    print(f"\n--- Processing Patient {dataset_num} ---")

    names = []
    fields = []
    masks = []
    suscs = []

    for orients in orientation:
        patch_number = 1
        phs_file = os.path.join(data_path, f"phs{orients}.mat")
        msk_file = os.path.join(data_path, f"msk{orients}.mat")
        cos_file = os.path.join(data_path, f"cos{orients}.mat")

        field = scipy.io.loadmat(phs_file)["phs"]
        mask  = scipy.io.loadmat(msk_file)["msk"]
        susc  = scipy.io.loadmat(cos_file)["cos"]

        matrix_size = np.shape(field)
        strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]

        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    phs_tmp = field[i*strides[0]:i*strides[0]+PS,
                                    j*strides[1]:j*strides[1]+PS,
                                    k*strides[2]:k*strides[2]+PS]

                    msk_tmp = mask[ i*strides[0]:i*strides[0]+PS,
                                    j*strides[1]:j*strides[1]+PS,
                                    k*strides[2]:k*strides[2]+PS]

                    susc_tmp = susc[i*strides[0]:i*strides[0]+PS,
                                    j*strides[1]:j*strides[1]+PS,
                                    k*strides[2]:k*strides[2]+PS]

                    base_name = f"tr-{dataset_num}-{orients}-{patch_number}.mat"
                    filename = os.path.join(out_patches_dir, base_name)

                    scipy.io.savemat(filename, {
                        "phs": phs_tmp,
                        "msk": msk_tmp,
                        "susc": susc_tmp
                    })

                    meta = {
                        "FileName": base_name,
                        "patient_id": dataset_num,
                        "orientation": orients,
                        "patch_number": patch_number
                    }
                    patch_info_list.append(meta)
                    names.append(base_name)
                    fields.append(phs_tmp)
                    masks.append(msk_tmp)
                    suscs.append(susc_tmp)

                    patch_number += 1

    p_dir = os.path.join(base_out_path, f"patient_{dataset_num}/")
    p_csv_dir = os.path.join(p_dir, "csv_files/")
    os.makedirs(p_csv_dir, exist_ok=True)

    p_f = np.array(fields, dtype="float32")
    p_m = np.array(masks,  dtype="float32")
    p_s = np.array(suscs,  dtype="float32")

    inp_mean = float(np.mean(p_f[p_m > 0]))
    inp_std  = float(np.std(p_f[p_m > 0]))
    out_mean = float(np.mean(p_s[p_m > 0]))
    out_std  = float(np.std(p_s[p_m > 0]))

    scipy.io.savemat(os.path.join(p_dir, "tr-stats.mat"), {
        "inp_mean": inp_mean, "inp_std": inp_std,
        "out_mean": out_mean, "out_std": out_std
    })

    p_df = pd.DataFrame(names, columns=["FileName"])
    n_total = len(p_df)
    n_val = int(0.1 * n_total)
    p_val_df = p_df.iloc[:n_val]
    p_train_df = p_df.iloc[n_val:]

    p_df.to_csv(os.path.join(p_dir, "all_patches.csv"), index=False)
    p_train_df.to_csv(os.path.join(p_csv_dir, "train.csv"), index=False)
    p_val_df.to_csv(os.path.join(p_csv_dir, "val.csv"), index=False)

    print(f"Patient {dataset_num} done in {time.time()-t_p:.1f}s. Patches: {n_total}. Mean inp: {inp_mean:.4f}, out: {out_mean:.4f}")

all_df = pd.DataFrame(patch_info_list)
all_df.to_csv(os.path.join(base_out_path, "all_cohort_patches.csv"), index=False)
print(f"\n#### ALL 8 PATIENTS COMPLETED IN {time.time()-start_time:.1f}s! Total patches: {len(patch_info_list)} ####")
