import scipy.io
import os

raw_dir = '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients/patient_1/'
print('Files in patient_1:', os.listdir(raw_dir))
for f in sorted(os.listdir(raw_dir)):
    if f.endswith('.mat'):
        mat = scipy.io.loadmat(os.path.join(raw_dir, f))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        shapes = {k: mat[k].shape for k in keys}
        print(f, shapes)
