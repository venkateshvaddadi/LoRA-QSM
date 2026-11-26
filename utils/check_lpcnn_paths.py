import os
paths = [
    '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_8_patients',
    '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/LPCNN_data',
    '/media/medimg/Data/venkatesh_data/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training'
]
for p in paths:
    print(p, 'exists:', os.path.exists(p))
    if os.path.exists(p):
        print('  contents:', os.listdir(p)[:10])
