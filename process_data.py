import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from datetime import date
from download_data import (
    download_and_combine,
    slice_data,
    interp_file_name,
    get_interpolated_z_data,
)
from torch.autograd import grad

torch.gradient
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


def calculate_gradient_of_wind_field(HR_data, x, y, Z):
    grad_x, grad_y = torch.gradient(HR_data, dim=(2, 3), spacing=(x, y))
    grad_z = calculate_div_z(HR_data, Z)
    divergence = (
        grad_x[:, 0, 1:-1, 1:-1, 1:-1]
        + grad_y[:, 1, 1:-1, 1:-1, 1:-1]
        + grad_z[:, 2, 1:-1, 1:-1, :]
    )
    xy_divergence = (
        grad_x[:, 0, 1:-1, 1:-1, 1:-1]
        + grad_y[:, 1, 1:-1, 1:-1, 1:-1]
    )

    return (
        torch.cat(
            (
                grad_x[:, :, 1:-1, 1:-1, 1:-1],
                grad_y[:, :, 1:-1, 1:-1, 1:-1],
                grad_z[:, :, 1:-1, 1:-1, :],
            ),
            dim=1,
        ),
        divergence,
        xy_divergence
    )


def calculate_div_z(HR_data, Z):
    dZ = Z[:, :, :, 1:] - Z[:, :, :, :-1]

    derivatives = torch.zeros_like(HR_data)[:, :, :, :, 1:-1]
    for i in range(1, Z.shape[-1] - 1):
        dz1, dz2 = torch.tile(
            dZ[:, None, :, :, i - 1], (1, HR_data.shape[1], 1, 1)
        ), torch.tile(dZ[:, None, :, :, i], (1, HR_data.shape[1], 1, 1))

        derivatives[:, :, :, :, i - 1] = (
            dz1**2 * HR_data[:, :, :, :, i + 1]
            + (dz2**2 - dz1**2) * HR_data[:, :, :, :, i]
            - dz2**2 * HR_data[:, :, :, :, i - 1]
        ) / (dz1 * dz2 * (dz1 + dz2))

    return derivatives


# Creating coarse simulation by skipping every alternate grid
def reformat_to_torch(
    u,
    v,
    w,
    p,
    z,
    coarseness_factor=4,
    train_fraction=0.8,
    include_pressure=False,
    include_z_channel=False,
    include_above_ground_channel = False,
    terrain = np.asarray([])
):
    
    u, v, w, p, z = [
        wind_component[:, np.newaxis, :, :, :]
        for wind_component in [u, v, w, p, z]
    ]

    HR_arr = np.concatenate((u, v, w), axis=1)
    del u, v, w

    if include_pressure:
        HR_arr = np.concatenate((HR_arr / np.max(HR_arr.__abs__()), p / np.max(p)), axis=1)
    else:
        HR_arr = HR_arr / np.max(HR_arr.__abs__())
    
    
    if include_z_channel:
        z_norm = (z - np.min(z))/(np.max(z)-np.min(z))
        arr_norm_LR = np.concatenate((HR_arr, z_norm), axis=1)[:, :, ::coarseness_factor, ::coarseness_factor, :]
        del z_norm
    else:
        arr_norm_LR = HR_arr[:, :, ::coarseness_factor, ::coarseness_factor, :]
    
    if include_above_ground_channel:
        z_above_ground = np.transpose(
                np.transpose(z, ([0, 1, 4, 2, 3])) - terrain, ([0, 1, 3, 4, 2])
            )
        arr_norm_LR = np.concatenate((arr_norm_LR, z_above_ground[:,:,::coarseness_factor, ::coarseness_factor, :]/np.max(z_above_ground)), axis=1)
        del z_above_ground

    HR_data = torch.from_numpy(HR_arr)
    LR_data = torch.from_numpy(arr_norm_LR)
    z = torch.from_numpy(z)

    number_of_train_samples = int(HR_data.size(0) * train_fraction)
    number_of_test_samples = int(HR_data.size(0) * (1 - train_fraction) / 2)
    index_start_end = [
        (0, number_of_train_samples),
        (number_of_train_samples, number_of_train_samples + number_of_test_samples),
        (number_of_train_samples + number_of_test_samples, -1),
    ]

    (
        (HR_data_train, LR_data_train, z_data_train),
        (HR_data_test, LR_data_test, z_data_test),
        (HR_data_val, LR_data_val, z_data_val),
    ) = [
        (HR_data[start:end].squeeze(), LR_data[start:end].squeeze(), z[start:end].squeeze())
        for start, end in index_start_end
    ]

    return (
        HR_data_train,
        LR_data_train,
        z_data_train,
        HR_data_test,
        LR_data_test,
        z_data_test,
        HR_data_val,
        LR_data_val,
        z_data_val
    )


def preprosess(
    train_fraction=0.8,
    X_DICT={"start": 4, "max": -3, "step": 1},
    Z_DICT={"start": 1, "max": 11, "step": 1},
    start_date=date(2018, 4, 1),
    end_date=date(2018, 4, 3),
    include_pressure=False,
    include_z_channel=False,
    interpolate_z=False,
    include_above_ground_channel = False,
):
    data_code = "simra_BESSAKER_"

    terrain, x, y, z, u, v, w, pressure = download_and_combine(
        data_code, start_date, end_date
    )

    terrain, x, y, X, Y, z, u, v, w, pressure = slice_data(
        terrain, x, y, z, u, v, w, pressure, X_DICT, Z_DICT
    )

    COARSENESS_FACTOR = 4

    

    if interpolate_z:
        filename = "./saved_interpolated_z_data/" + interp_file_name(
            X_DICT, Z_DICT, start_date, end_date
        )
        Z_interp_above_ground, u, v, w, pressure = get_interpolated_z_data(
            filename, x, y, z, terrain, u, v, w, pressure
        )
        z = np.tile(
            np.transpose(
                np.transpose(Z_interp_above_ground, ([2, 0, 1])) + terrain, ([1, 2, 0])
            ),
            (z.shape[0], 1, 1, 1),
        )

    (
        HR_data_train,
        LR_data_train,
        z_data_train,
        HR_data_test,
        LR_data_test,
        z_data_test,
        HR_data_val,
        LR_data_val,
        z_data_val
    ) = reformat_to_torch(
        u,
        v,
        w,
        pressure,
        z,
        COARSENESS_FACTOR,
        train_fraction,
        include_pressure=include_pressure,
        include_z_channel=include_z_channel,
        include_above_ground_channel=include_above_ground_channel,
        terrain = terrain,
    )

    dataset_train = torch.utils.data.TensorDataset(LR_data_train, HR_data_train, z_data_train)
    dataset_test = torch.utils.data.TensorDataset(LR_data_test, HR_data_test, z_data_test)
    dataset_validation = torch.utils.data.TensorDataset(LR_data_val, HR_data_val, z_data_val)

    # LR_test, HR_test = dataset_train[:8]
    # Z = HR_test[:, -1, :, :, :]

    # grad_tensor, divergence = calculate_gradient_of_wind_field(HR_test[:,:-1,:,:,:], torch.from_numpy(x), torch.from_numpy(y), Z)

    # # plot_pressure(X[1:-1,1:-1,1:3], Y[1:-1,1:-1,1:3], z[0,1:-1,1:-1,1:3], 3, HR_data_test[0,0,1:-1,1:-1,1:3], terrain, fig=1)

    # plot_pressure(X[1:-1,1:-1,1:3], Y[1:-1,1:-1,1:3], z[0,1:-1,1:-1,1:3], 3, divergence[0,:,:,:2], terrain, fig=2)

    # plot_field(X[1:-1,1:-1,1], Y[1:-1,1:-1,1], z[0,1:-1,1:-1,1], HR_test[0,0,1:-1,1:-1,1], HR_test[0,1,1:-1,1:-1,1], HR_test[0,2,1:-1,1:-1,1], terrain[1:-1,1:-1], z_plot_scale=3, fig=3)
    # plot_field(X, Y, z[time_index], HR_data_train[time_index, 0, :, :, :], HR_data_train[time_index, 1, :,:,:], HR_data_train[time_index, 2, :,:,:], terrain, fig=2)
    # plot_field(X[::COARSENESS_FACTOR, ::COARSENESS_FACTOR,:], Y[::COARSENESS_FACTOR, ::COARSENESS_FACTOR,:], z[time_index, ::COARSENESS_FACTOR, ::COARSENESS_FACTOR,:], LR_data_train[time_index, 0, :, :, :], LR_data_train[time_index, 1, :, :, :], LR_data_train[time_index, 2, :, :, :], terrain[::COARSENESS_FACTOR, ::COARSENESS_FACTOR], fig=3)

    return (
        dataset_train,
        dataset_test,
        dataset_validation,
        torch.from_numpy(x),
        torch.from_numpy(y),
    )


if __name__ == "__main__":
    preprosess(include_above_ground_channel=True)
