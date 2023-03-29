import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from datetime import date
from download_data import download_and_combine, slice_data


# Handling the masked array problem by
# 1) Filling the masked arrays with nan
# 2) removing first,second and last row to get it by same grid dimension
# 2.1) Need to get the grid dimension on 2^n form, e.g. 128 x 128

# u_nan = np.ma.filled(u.astype(float), np.nan)
# u_nomask = u_nan[:, 4:-4, 4:-3]

# v_nan = np.ma.filled(v.astype(float), np.nan)
# v_nomask = v_nan[:, 4:-4, 4:-3]

# w_nan = np.ma.filled(w.astype(float), np.nan)
# w_nomask = w_nan[:, 4:-4, 4:-3]


# Creating coarse simulation by skipping every alternate grid
def reformat_to_torch(u, v, w, pressure, z, coarseness_factor=4, train_fraction=0.8):
    

    # Transforming the HR data into tensor form
    u, v, w, p, z_tensor = [wind_component[:, np.newaxis, :, :, :]
        for wind_component in [u, v, w, pressure, z]
    ]

    uvw_arr = np.concatenate((u, v, w), axis=1)
    max_comp = np.max(uvw_arr.__abs__())   
    
    uvw_arr_norm_HR = uvw_arr/max_comp

    uvw_arr_norm_LR = uvw_arr_norm_HR[:, :, ::coarseness_factor, ::coarseness_factor,:]


    # Concatenating the tensors together like three RGB channels
    HR_data = torch.from_numpy(uvw_arr_norm_HR)
    LR_data = torch.from_numpy(uvw_arr_norm_LR)


    number_of_train_samples = int(HR_data.size(0) * train_fraction)
    number_of_test_samples = int(HR_data.size(0) * (1 - train_fraction) / 2)
    index_start_end = [(0,number_of_train_samples), (number_of_train_samples, number_of_train_samples + number_of_test_samples), (number_of_train_samples + number_of_test_samples, -1)]
        
    (HR_data_train, LR_data_train), (HR_data_test, LR_data_test), (HR_data_val, LR_data_val) = [(HR_data[start:end].squeeze(), LR_data[start:end].squeeze()) for start, end in index_start_end]

    
    return HR_data_train, LR_data_train, HR_data_test, LR_data_test, HR_data_val, LR_data_val



def preprosess(train_fraction=0.8, X_DICT = {"start": 4, "max": -3, "step": 1}, Z_DICT = {"start": 1, "max": 2, "step": 1}, start_date=date(2018, 4, 1), end_date= date(2018, 4, 3)):
    data_code = "simra_BESSAKER_"

    time, terrain, x, y, z, u, v, w, theta, tke, td, pressure = download_and_combine(
        data_code, start_date, end_date
    )

    terrain, x, y, X, Y, z, u, v, w, pressure = slice_data(
        terrain, x, y, z, u, v, w, pressure, X_DICT, Z_DICT
    )

    COARSENESS_FACTOR = 4

    HR_data_train, LR_data_train, HR_data_test, LR_data_test, HR_data_val, LR_data_val = reformat_to_torch(u, v, w, pressure, z, COARSENESS_FACTOR, train_fraction)

    dataset_train = torch.utils.data.TensorDataset(
        LR_data_train, HR_data_train
    )
    dataset_test = torch.utils.data.TensorDataset(
        LR_data_test, HR_data_test
    )
    dataset_validation = torch.utils.data.TensorDataset(
        LR_data_val, HR_data_val
    )

    return dataset_train, dataset_test, dataset_validation

if __name__ == "__main__":
    preprosess()
