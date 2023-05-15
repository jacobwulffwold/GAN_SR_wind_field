import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
import pickle
from download_data import (
    download_and_split,
    slice_only_dim_dicts,
    slice_dict_folder_name,
    get_interpolated_z_data,
    get_static_data,
    filenames_from_start_and_end_dates,
)
from datetime import date
import os

class CustomizedDataset(torch.utils.data.Dataset):
  def __init__(self, 
    filenames,
    subfolder_name, 
    Z_MIN, 
    Z_MAX, 
    UVW_MAX, 
    P_MAX,
    Z_ABOVE_GROUND_MAX,
    x,
    y,
    terrain,
    include_pressure=False,
    include_z_channel=False,
    interpolate_z=False,
    include_above_ground_channel = False,
    COARSENESS_FACTOR = 4,
    data_aug_rot = True,
    data_aug_flip = True,
    enable_slicing = False,
    slice_size = 64,):
        self.filenames = filenames
        self.subfolder_name =subfolder_name
        self.include_pressure = include_pressure
        self.include_z_channel = include_z_channel
        self.interpolate_z = interpolate_z
        self.include_above_ground_channel = include_above_ground_channel
        self.coarseness_factor = COARSENESS_FACTOR
        self.Z_MIN = Z_MIN
        self.Z_MAX = Z_MAX
        self.Z_ABOVE_GROUND_MAX = Z_ABOVE_GROUND_MAX
        self.UVW_MAX = UVW_MAX
        self.P_MAX = P_MAX
        self.x = x
        self.y = y
        self.data_aug_rot = data_aug_rot
        self.data_aug_flip = data_aug_flip
        self.terrain = terrain
        self.enable_slicing = enable_slicing
        self.slice_size = slice_size

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        z, z_above_ground, u, v, w, pressure = pickle.load(open("./full_dataset_files/"+self.subfolder_name+self.filenames[index], "rb"))
        
        if self.interpolate_z:
            z, z_above_ground, u, v, w, pressure = get_interpolated_z_data(
                "./saved_interpolated_z_data/"+self.subfolder_name+self.filenames[index], self.x, self.y, z_above_ground, self.terrain, u, v, w, pressure
            )
        
        if self.enable_slicing:
            x_start = np.random.randint(0, self.x.size - self.slice_size)
            y_start = np.random.randint(0, self.y.size - self.slice_size)
            z, z_above_ground, u, v, w, pressure = slice_only_dim_dicts(z, z_above_ground, u, v, w, pressure, x_dict={"start":x_start, "max":x_start+self.slice_size, "step":1}, y_dict={"start":y_start, "max":y_start+self.slice_size, "step":1}, z_dict={"start":0, "max":z.shape[-1], "step":1})

        LR, HR, Z = reformat_to_torch(
            u,
            v,
            w,
            pressure,
            z,
            z_above_ground,
            self.Z_MIN, 
            self.Z_MAX,
            self.Z_ABOVE_GROUND_MAX,
            self.UVW_MAX, 
            self.P_MAX,
            coarseness_factor=self.coarseness_factor,
            include_pressure=self.include_pressure,
            include_z_channel=self.include_z_channel,
            include_above_ground_channel=self.include_above_ground_channel,
        )

        if self.data_aug_rot:
            amount_of_rotations = np.random.randint(0, 4)
            LR = torch.rot90(LR, amount_of_rotations, [1, 2])
            HR = torch.rot90(HR, amount_of_rotations, [1, 2])
            Z = torch.rot90(Z, amount_of_rotations, [1, 2])

        if self.data_aug_flip:
            if np.random.rand() > 0.5:
                LR = torch.flip(LR, [1])
                HR = torch.flip(HR, [1])
                Z = torch.flip(Z, [1])    
            if np.random.rand() > 0.5:
                LR = torch.flip(LR, [2])
                HR = torch.flip(HR, [2])
                Z = torch.flip(Z, [2])
            
        return LR, HR, Z

def calculate_div_z(HR_data:torch.Tensor, Z:torch.Tensor):
    dZ = Z[:, 0, :, :, 1:] - Z[:, 0, :, :, :-1]

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

@torch.jit.script
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


def download_all_files_and_prepare(start_date:date, end_date:date, x_dict, y_dict, z_dict, terrain, folder:str="./full_dataset_files/", train_eval_test_ratio=0.8):
    
    filenames = filenames_from_start_and_end_dates(start_date, end_date)
    Z_MIN, Z_MAX, UVW_MAX, P_MAX, Z_ABOVE_GROUND_MAX = 10000, 0, 0, 0, 0

    finished = False
    start = -1
    subfolder = slice_dict_folder_name(x_dict, y_dict, z_dict)
    
    if not os.path.exists(folder+subfolder):
        os.makedirs(folder+subfolder)
    
    invalid_samples = set()
    
    while not finished:
        for i in range(len(filenames)):           
            if filenames[i] not in invalid_samples:
                try:
                    with open(folder+subfolder+"max_"+filenames[i], "rb") as f:
                        z_min, z_max, z_above_ground_max, uvw_max, p_max = pickle.load(
                            f
                        )
                    if i < train_eval_test_ratio * len(filenames):
                        Z_MIN = min(Z_MIN, z_min)
                        Z_MAX = max(Z_MAX, z_max)
                        UVW_MAX = max(UVW_MAX, uvw_max)
                        P_MAX = max(P_MAX, p_max)
                        Z_ABOVE_GROUND_MAX = max(Z_ABOVE_GROUND_MAX, z_above_ground_max)
                    
                    if start != -1:
                        print("Downloading new files, from ", filenames[start], " to ", filenames[i])
                        invalid_samples = invalid_samples.union(download_and_split(filenames[start:i], terrain, x_dict, y_dict, z_dict, folder=folder+subfolder))
                        start = -1
                except:
                    if start == -1:
                        start = i
            
            if i == len(filenames)-1:
                if start != -1:
                    print("Downloading new files, from ", filenames[start], " to ", filenames[i])
                    invalid_samples = invalid_samples.union(download_and_split(filenames[start:], terrain, x_dict, y_dict, z_dict, folder=folder+subfolder))
                    start = -1
                else:
                    finished = True
    
    filenames = [item for item in filenames if item not in invalid_samples]
    
    print("Finished downloading all files")

    return filenames, subfolder, Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MAX

# Creating coarse simulation by skipping every alternate grid
def reformat_to_torch(
    u,
    v,
    w,
    p,
    z,
    z_above_ground,
    Z_MIN, 
    Z_MAX, 
    Z_ABOVE_GROUND_MAX,
    UVW_MAX, 
    P_MAX,
    coarseness_factor=4,
    include_pressure=False,
    include_z_channel=False,
    include_above_ground_channel = False,
):
    
    u, v, w, p, z = [
        wind_component[np.newaxis, :, :, :]
        for wind_component in [u, v, w, p, z]
    ]

    HR_arr = np.concatenate((u, v, w), axis=0)
    del u, v, w

    if include_pressure:
        HR_arr = np.concatenate((HR_arr / UVW_MAX, p / P_MAX), axis=0)
    else:
        HR_arr = HR_arr / UVW_MAX
    
    if include_z_channel:
        arr_norm_LR = np.concatenate((HR_arr, (z - Z_MIN)/(Z_MAX-Z_MIN)), axis=0)[:, ::coarseness_factor, ::coarseness_factor, :]
    else:
        arr_norm_LR = HR_arr[:, ::coarseness_factor, ::coarseness_factor, :]
    
    if include_above_ground_channel:
        arr_norm_LR = np.concatenate((arr_norm_LR, z_above_ground[:,::coarseness_factor, ::coarseness_factor, :]/Z_ABOVE_GROUND_MAX), axis=0)
        del z_above_ground

    HR_data = torch.from_numpy(HR_arr).float()
    LR_data = torch.from_numpy(arr_norm_LR).float()
    z = torch.from_numpy(z).float()
    
    return (
        LR_data,
        HR_data,
        z,
    )

def preprosess(
    train_eval_test_ratio=0.8,
    X_DICT={"start": 0, "max": 128, "step": 1},
    Y_DICT={"start": 0, "max": 128, "step": 1},
    Z_DICT={"start": 0, "max": 10, "step": 1},
    start_date=date(2018, 4, 1),
    end_date=date(2018, 4, 3),
    include_pressure=False,
    include_z_channel=False,
    interpolate_z=False,
    enable_slicing = False,
    slice_size = 64,
    include_above_ground_channel = False,
    COARSENESS_FACTOR = 4,
    train_aug_rot = False,
    val_aug_rot = False,
    test_aug_rot = False,
    train_aug_flip = False,
    val_aug_flip = False,
    test_aug_flip = False,
):
    try:
        with open("./full_dataset_files/static_terrain_x_y.pkl", "rb") as f:
            terrain, x, y = slice_only_dim_dicts(*pickle.load(f), x_dict=X_DICT, y_dict=Y_DICT)
    except:
        get_static_data()
        with open("./full_dataset_files/static_terrain_x_y.pkl", "rb") as f:
            terrain, x, y = slice_only_dim_dicts(*pickle.load(f), x_dict=X_DICT, y_dict=Y_DICT)

    filenames, subfolder, Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MAX = download_all_files_and_prepare(start_date, end_date, X_DICT, Y_DICT, Z_DICT, terrain, train_eval_test_ratio=train_eval_test_ratio)
    
    if interpolate_z:
        if not os.path.exists("./saved_interpolated_z_data/"+subfolder):
            os.makedirs("./saved_interpolated_z_data/"+subfolder)
           
    number_of_train_samples = int(len(filenames) * train_eval_test_ratio)
    number_of_test_samples = int(len(filenames) * (1 - train_eval_test_ratio) / 2)
        

    dataset_train = CustomizedDataset(
        filenames[:number_of_train_samples],
        subfolder,
        Z_MIN,
        Z_MAX,
        UVW_MAX,
        P_MAX,
        Z_ABOVE_GROUND_MAX,
        x,
        y,
        terrain,
        include_pressure=include_pressure,
        include_z_channel=include_z_channel,
        interpolate_z=interpolate_z,
        include_above_ground_channel = include_above_ground_channel,
        COARSENESS_FACTOR = COARSENESS_FACTOR,
        data_aug_rot=train_aug_rot,
        data_aug_flip=train_aug_flip,
        enable_slicing = enable_slicing,
        slice_size = slice_size,)

    dataset_test = CustomizedDataset(
        filenames[number_of_train_samples:number_of_train_samples + number_of_test_samples],
        subfolder,
        Z_MIN,
        Z_MAX,
        UVW_MAX,
        P_MAX,
        Z_ABOVE_GROUND_MAX,
        x,
        y,
        terrain,
        include_pressure=include_pressure,
        include_z_channel=include_z_channel,
        interpolate_z=interpolate_z,
        include_above_ground_channel = include_above_ground_channel,
        COARSENESS_FACTOR = COARSENESS_FACTOR,
        data_aug_rot=test_aug_rot,
        data_aug_flip=test_aug_flip,
        enable_slicing = enable_slicing,
        slice_size = slice_size,)

    dataset_validation = CustomizedDataset(
        filenames[number_of_train_samples + number_of_test_samples:],
        subfolder,
        Z_MIN,
        Z_MAX,
        UVW_MAX,
        P_MAX,
        Z_ABOVE_GROUND_MAX,
        x,
        y,
        terrain,
        include_pressure=include_pressure,
        include_z_channel=include_z_channel,
        interpolate_z=interpolate_z,
        include_above_ground_channel = include_above_ground_channel,
        COARSENESS_FACTOR = COARSENESS_FACTOR,
        data_aug_rot=val_aug_rot,
        data_aug_flip=val_aug_flip,
        enable_slicing = enable_slicing,
        slice_size = slice_size,)

    # LR_test, HR_test, Z_test = dataset_train[:8]
    # Z = HR_test[:, -1, :, :, :]

    # grad_tensor, divergence = calculate_gradient_of_wind_field(HR_test[:,:-1,:,:,:], torch.from_numpy(x), torch.from_numpy(y), Z)

    # # plot_pressure(X[1:-1,1:-1,1:3], Y[1:-1,1:-1,1:3], z[0,1:-1,1:-1,1:3], 3, HR_data_test[0,0,1:-1,1:-1,1:3], terrain, fig=1)

    # plot_pressure(X[1:-1,1:-1,1:3], Y[1:-1,1:-1,1:3], z[0,1:-1,1:-1,1:3], 3, divergence[0,:,:,:2], terrain, fig=2)

    # plot_field(X[1:-1,1:-1,1], Y[1:-1,1:-1,1], z[0,1:-1,1:-1,1], HR_test[0,0,1:-1,1:-1,1], HR_test[0,1,1:-1,1:-1,1], HR_test[0,2,1:-1,1:-1,1], terrain[1:-1,1:-1], z_plot_scale=3, fig=3)
    # plot_field(X, Y, z[time_index], HR_data_train[time_index, 0, :, :, :], HR_data_train[time_index, 1, :,:,:], HR_data_train[time_index, 2, :,:,:], terrain, fig=2)
    # plot_field(X[::COARSENESS_FACTOR, ::COARSENESS_FACTOR,:], Y[::COARSENESS_FACTOR, ::COARSENESS_FACTOR,:], z[time_index, ::COARSENESS_FACTOR, ::COARSENESS_FACTOR,:], LR_data_train[time_index, 0, :, :, :], LR_data_train[time_index, 1, :, :, :], LR_data_train[time_index, 2, :, :, :], terrain[::COARSENESS_FACTOR, ::COARSENESS_FACTOR], fig=3)

    if enable_slicing:
        x, y, = x[:slice_size], y[:slice_size]


    return (
        dataset_train,
        dataset_test,
        dataset_validation,
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float(),
    )


if __name__ == "__main__":
    preprosess(include_above_ground_channel=True)
