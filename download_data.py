# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from datetime import datetime, timedelta, date
from urllib import request
import os
import pickle
import netCDF4
from netCDF4 import Dataset
import numpy as np
from mayavi import mlab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

EARTH_RADIUS = 6371009


def plot_images(
    filename,
    terrain,
    X,
    Y,
    Z,
    x_dict={"start": 0, "max": -1, "step": 1},
    z_dict={"start": 0, "max": -1, "step": 5},
    z_plot_scale=1,
):
    with open(filename, "rb") as f:
        images = pickle.load(f)

    i = 0
    for key, value in images.items():
        print(key)
        this_x_dict = x_dict.copy()
        if key == "LR":
            for dict_key, dict_value in x_dict.items():
                this_x_dict[dict_key] = dict_value // 4
                if dict_key == "step" and dict_value // 4 == 0:
                    this_x_dict[dict_key] = 1

            this_X, this_Y, this_Z, this_terrain, u, v, w = slice_only_dim_dicts(
                this_x_dict,
                z_dict,
                X[::4, ::4, :],
                Y[::4, ::4, :],
                Z[::4, ::4, :],
                terrain[::4, ::4],
                value[0],
                value[1],
                value[2],
            )

        else:
            this_X, this_Y, this_Z, this_terrain, u, v, w = slice_only_dim_dicts(
                x_dict, z_dict, X, Y, Z, terrain, value[0], value[1], value[2]
            )

        plot_field(
            np.squeeze(this_X),
            np.squeeze(this_Y),
            np.squeeze(this_Z),
            u,
            v,
            w,
            z_plot_scale=z_plot_scale,
            fig=i,
            terrain=this_terrain,
        )

        i += 1


def check(url):
    try:
        u = request.urlopen(url)
        u.close()
        return True
    except:
        return False

def filenames_from_start_and_end_dates(start_date: date, end_date: date):
    start_time = datetime(start_date.year, start_date.month, start_date.day)
    end_time = datetime(end_date.year, end_date.month, end_date.day)
    delta = end_time - start_time
    names = []
    for i in range((delta.days+1) * 24):
        names.append((str(start_time + timedelta(hours=i))+".pkl").replace(" ","-").replace(":00:00",""))
    
    return names

def download_Bessaker_data(start_date, end_date, destination_folder):
    start_date = start_date
    end_date = end_date
    delta = end_date - start_date
    home_dir = "https://thredds.met.no/thredds/fileServer/opwind/"
    data_code = "simra_BESSAKER_"
    sim_times = ["T00Z.nc", "T12Z.nc"]
    no_data_points = (delta.days + 1) * 2
    counter = 0
    for i in range(delta.days + 1):
        for sim_time in sim_times:
            temp = start_date + timedelta(days=i)
            temp_date = datetime.strptime(str(temp), "%Y-%m-%d")
            filename = data_code + ((str(temp)).replace("-", "")) + sim_time
            local_filename = destination_folder + filename
            if os.path.isfile(local_filename) == True:
                counter = counter + 1
                print("Number of files downloaded ", counter, "/", no_data_points)
            elif os.path.isfile(local_filename) != True:
                print("Attempting to download file ", filename)
                URL = (
                    home_dir
                    + str(temp_date.year)
                    + "/"
                    + str(temp_date.month).zfill(2)
                    + "/"
                    + str(temp_date.day).zfill(2)
                    + "/"
                    + filename
                )
                try:
                    if check(URL):
                        request.urlretrieve(URL, local_filename)
                        print("Downloaded file", filename)
                        counter = counter + 1
                        print(
                            "Number of files downloaded ", counter, "/", no_data_points
                        )
                    else:
                        print("File not found")
                except TypeError as e:
                    print("Error downlowing file {}".format(e))
                    print("Continuing")


def combine_files(data_code, start_date, end_date, outfilename):
    start_date = start_date
    end_date = end_date
    delta = end_date - start_date
    sim_times = ["T00Z.nc", "T12Z.nc"]
    nc_dir = list()
    for i in range(delta.days + 1):
        for sim_time in sim_times:
            temp = start_date + timedelta(days=i)
            filename = data_code + ((str(temp)).replace("-", ""))
            filename = "/downloaded_raw_bessaker_data/" + filename + sim_time
            nc_dir.append(filename)
    # list_of_paths = glob.glob(nc_dir, recursive=True)
    nc_fid = netCDF4.MFDataset(nc_dir)
    latitude = nc_fid["longitude"][:]
    longitude = nc_fid["latitude"][:]
    x = nc_fid["x"][:]
    y = nc_fid["y"][:]
    z = nc_fid["geopotential_height_ml"][1, :, :, :]
    terrain = nc_fid["surface_altitude"][:]
    theta = nc_fid["air_potential_temperature_ml"][:]
    u = nc_fid["x_wind_ml"][:]
    v = nc_fid["y_wind_ml"][:]
    w = nc_fid["upward_air_velocity_ml"][:]
    u10 = nc_fid["x_wind_10m"][:]
    v10 = nc_fid["y_wind_10m"][:]
    tke = nc_fid["turbulence_index_ml"][:]
    td = nc_fid["turbulence_dissipation_ml"][:]
    outfile = open(outfilename, "wb")
    pickle.dump(
        [latitude, longitude, x, y, z, u, v, w, theta, tke, td, u10, v10, terrain],
        outfile,
    )
    outfile.close()


def extract_XY_plane(data_code, start_date, end_date, iz):
    data_code = data_code
    start_date = start_date
    end_date = end_date
    delta = end_date - start_date
    sim_times = ["T00Z.nc", "T12Z.nc"]
    index = 0
    for i in range(delta.days + 1):
        for sim_time in sim_times:
            temp = start_date + timedelta(days=i)
            filename = data_code + ((str(temp)).replace("-", ""))
            filename = "./downloaded_raw_bessaker_data/" + filename + sim_time
            if index == 0:
                nc_fid = Dataset(filename, mode="r")
                time = nc_fid["time"][:]
                latitude = nc_fid["longitude"][:]
                longitude = nc_fid["latitude"][:]
                x = nc_fid["x"][:]
                y = nc_fid["y"][:]
                z = nc_fid["geopotential_height_ml"][:][1, iz, :, :]
                terrain = nc_fid["surface_altitude"][:]
                theta = nc_fid["air_potential_temperature_ml"][:][:, iz, :, :]
                u = nc_fid["x_wind_ml"][:][:, iz, :, :]
                v = nc_fid["y_wind_ml"][:][:, iz, :, :]
                w = nc_fid["upward_air_velocity_ml"][:][:, iz, :, :]
                # u10 = nc_fid['x_wind_10m'][:]
                # v10 = nc_fid['y_wind_10m'][:]
                tke = nc_fid["turbulence_index_ml"][:][:, iz, :, :]
                td = nc_fid["turbulence_dissipation_ml"][:][:, iz, :, :]
                index = index + 1
                nc_fid.close()
            else:
                nc_fid = Dataset(filename, mode="r")
                time = np.ma.append(time, nc_fid["time"][:][1:13], axis=0)
                u = np.ma.append(u, nc_fid["x_wind_ml"][1:13, iz, :, :], axis=0)
                v = np.ma.append(v, nc_fid["y_wind_ml"][:][1:13, iz, :, :], axis=0)
                w = np.ma.append(
                    w, nc_fid["upward_air_velocity_ml"][:][1:13, iz, :, :], axis=0
                )
                theta = np.ma.append(
                    theta,
                    nc_fid["air_potential_temperature_ml"][:][1:13, iz, :, :],
                    axis=0,
                )
                tke = np.ma.append(
                    tke, nc_fid["turbulence_index_ml"][:][1:13, iz, :, :], axis=0
                )
                td = np.ma.append(
                    td,
                    nc_fid["turbulence_dissipation_ml"][:][:][1:13, iz, :, :],
                    axis=0,
                )
                nc_fid.close()
    return time, latitude, longitude, terrain, x, y, z, u, v, w, theta, tke, td


def quick_append(var, key, nc_fid, transpose_indices=[0, 2, 3, 1]):
    return np.ma.append(
        var,
        np.transpose(nc_fid[key][:-1, :, :, :], (transpose_indices))[:, :, :, ::-1],
        axis=0,
    )

def get_static_data():
    filename = os.listdir("./downloaded_raw_bessaker_data/")[0]
    
    nc_fid = Dataset("./downloaded_raw_bessaker_data/" + filename, mode="r")
    
    x = 100000*nc_fid["x"][:]
    y = 100000*nc_fid["y"][:]
    terrain = nc_fid["surface_altitude"][:]
    nc_fid.close()

    terrain = np.ma.filled(terrain.astype(float), np.nan)
    terrain, x, y = slice_only_dim_dicts(terrain, x, y)

    with open("./full_dataset_files/static_terrain_x_y.pkl", "wb") as f:
            pickle.dump([terrain, x, y], f)

def extract_slice_and_filter_3D(data_code, start_date, end_date, transpose_indices=[0, 2, 3, 1]):
    delta = end_date - start_date
    sim_times = ["T00Z.nc", "T12Z.nc"]
    index = 0
    invalid_filenames = []
    for i in range(delta.days + 1):
        for sim_time in sim_times:
            temp = start_date + timedelta(days=i)
            filename = data_code + ((str(temp)).replace("-", ""))
            filename = "./downloaded_raw_bessaker_data/" + filename + sim_time
            try:
                nc_fid = Dataset(filename, mode="r")
                assert nc_fid["time"][:].shape[0] == 13
                if index == 0:
                    # time = nc_fid["time"][:]
                    # latitude = nc_fid["longitude"][:]
                    # longitude = nc_fid["latitude"][:]
                    z = np.transpose(
                        nc_fid["geopotential_height_ml"][:], (transpose_indices)
                    )[:-1, :, :, ::-1]
                    # theta = np.transpose(
                    #     nc_fid["air_potential_temperature_ml"][:], (transpose_indices)
                    # )[:, :, :, ::-1]
                    u = np.transpose(nc_fid["x_wind_ml"][:], (transpose_indices))[
                        :-1, :, :, ::-1
                    ]
                    v = np.transpose(nc_fid["y_wind_ml"][:], (transpose_indices))[
                        :-1, :, :, ::-1
                    ]
                    w = np.transpose(
                        nc_fid["upward_air_velocity_ml"][:], (transpose_indices)
                    )[:-1, :, :, ::-1]
                    pressure = np.transpose(
                        nc_fid["air_pressure_ml"][:], (transpose_indices)
                    )[:-1, :, :, ::-1]
                    # u10 = nc_fid['x_wind_10m'][:]
                    # v10 = nc_fid['y_wind_10m'][:]
                    # tke = np.transpose(
                    #     nc_fid["turbulence_index_ml"][:], (transpose_indices)
                    # )[:, :, :, ::-1]
                    # td = np.transpose(
                    #     nc_fid["turbulence_dissipation_ml"][:], (transpose_indices)
                    # )[:, :, :, ::-1]
                    index = index + 1
                    nc_fid.close()
                else:
                    u = quick_append(u, "x_wind_ml", nc_fid, transpose_indices)
                    v = quick_append(v, "y_wind_ml", nc_fid, transpose_indices)
                    w = quick_append(w, "upward_air_velocity_ml", nc_fid, transpose_indices)
                    pressure = quick_append(
                        pressure, "air_pressure_ml", nc_fid, transpose_indices
                    )
                    z = quick_append(z, "geopotential_height_ml", nc_fid, transpose_indices)
                    nc_fid.close()
            except:
                invalid_filenames.append((temp,sim_time))
    
    if "pressure" in locals():
        u, v, w, pressure, z = [np.ma.filled(wind_field.astype(float), np.nan)
            for wind_field in [u, v, w, pressure, z]
        ]
        z, u, v, w, pressure = slice_only_dim_dicts(z, u, v, w, pressure) 
    else:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), invalid_filenames

    
    return (
        # time,
        # latitude,
        # longitude,
        # terrain,
        # x,
        # y,
        z,
        u,
        v,
        w,
        # theta,
        # tke,
        # td,
        pressure,
        invalid_filenames,
    )


def slice_only_dim_dicts(
    *args,
    x_dict={"start": 4, "max": -4, "step": 1},
    y_dict={"start": 4, "max": -3, "step": 1},
    z_dict={"start": 1, "max": 41, "step": 1},
):
    value_list = []
    for val in args:
        if val.ndim == 3:
            value_list.append(
                val[
                    x_dict["start"] : x_dict["max"] : x_dict["step"],
                    y_dict["start"] : y_dict["max"] : x_dict["step"],
                    z_dict["start"] : z_dict["max"] : z_dict["step"],
                ]
            )
        elif val.ndim == 2:
            value_list.append(
                val[
                    x_dict["start"] : x_dict["max"] : x_dict["step"],
                    y_dict["start"] : y_dict["max"] : y_dict["step"],
                ]
            )
        elif val.ndim == 4:
            value_list.append(
                val[
                    :,
                    x_dict["start"] : x_dict["max"] : x_dict["step"],
                    y_dict["start"] : y_dict["max"] : y_dict["step"],
                    z_dict["start"] : z_dict["max"] : z_dict["step"],
                ]
            )
        elif val.ndim == 1:
            if val.size == 136:
                value_list.append(val[x_dict["start"] : x_dict["max"] : x_dict["step"]])
            else:
                value_list.append(val[y_dict["start"] : y_dict["max"] : y_dict["step"]])

    return value_list


def slice_data(
    terrain,
    x,
    y,
    z,
    u,
    v,
    w,
    pressure,
    x_dict={"start": 4, "max": -3, "step": 1},
    z_dict={"start": 1, "max": 41, "step": 1},
):
    X, Y, _ = np.meshgrid(x, y, z[0, 0, :])
    X, Y = (
        X[
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ],
        Y[
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ],
    )
    u, v, w, pressure, z = [
        wind_field[
            :,
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ]
        for wind_field in [u, v, w, pressure, z]
    ]

    terrain = terrain[
        x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
        x_dict["start"] : x_dict["max"] : x_dict["step"],
    ]

    return (
        terrain,
        x[x_dict["start"] : x_dict["max"] : x_dict["step"]],
        y[x_dict["start"] : x_dict["max"] - 1 : x_dict["step"]],
        X,
        Y,
        z,
        u,
        v,
        w,
        pressure,
    )


def interpolate_z_axis(
    x,
    y,
    z_above_ground,
    u,
    v,
    w,
    pressure,
):
    new_1D_z_above_ground = np.linspace(
        np.mean(z_above_ground[:, :, 0]),
        np.mean(z_above_ground[:, :, -1]),
        num=z_above_ground[0, 0, :].size,
    )
    _, _, new_3D_z_above_ground = np.meshgrid(x, y, new_1D_z_above_ground)

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
                   
            u[i, j, :] = np.interp(
                new_1D_z_above_ground, z_above_ground[i, j, :], u[i, j, :]
            )
            v[i, j, :] = np.interp(
                new_1D_z_above_ground, z_above_ground[i, j, :], v[i, j, :]
            )
            u[i, j, :] = np.interp(
                new_1D_z_above_ground, z_above_ground[i, j, :], w[i, j, :]
            )
            pressure[i, j, :] = np.interp(
                new_1D_z_above_ground,
                z_above_ground[i, j, :],
                pressure[i, j, :],
            )

    return new_3D_z_above_ground, u, v, w, pressure


def get_interpolated_z_data(
    filename,
    x,
    y,
    z_above_ground,
    terrain,
    u,
    v,
    w,
    pressure,
):
    try:
        with open(filename, "rb") as f:
            (
                z,
                Z_interp_above_ground,
                u,
                v,
                w,
                pressure,
            ) = pickle.load(f)
        # print("Loaded interpolated (z_above_ground) data from file " + filename)
    except:
        # print("Interpolating z axis...")
        (
            Z_interp_above_ground,
            u,
            v,
            w,
            pressure,
        ) = interpolate_z_axis(x, y, z_above_ground, u, v, w, pressure)

        z = np.transpose(np.transpose(Z_interp_above_ground, ([2,0,1])) + terrain, ([1,2,0]))

        with open(filename, "wb") as f:
            pickle.dump(
                [z, Z_interp_above_ground, u, v, w, pressure],
                f,
            )
        # print("Saved data to file " + filename)

    return z, Z_interp_above_ground, u, v, w, pressure


def interpolate_cartesian_3D(
    X_input,
    Y_input,
    Z_input,
    u_input,
    v_input,
    w_input,
    x,
    y,
    time_index,
):
    XYZ_sigma_flat = np.vstack(
        (X_input.flatten(), Y_input.flatten(), Z_input[time_index].flatten())
    ).T
    y_wind_field = np.vstack(
        (
            u_input[time_index].flatten(),
            v_input[time_index].flatten(),
            w_input[time_index].flatten(),
        )
    ).T

    kernel = RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)
    )

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)

    gpr.fit(XYZ_sigma_flat, y_wind_field)

    X_cartesian, Y_cartesian, Z_cartesian = np.meshgrid(
        x,
        y,
        np.arange(
            np.average(Z_input[time_index, :, :, 0]),
            (x.size) * (x[1] - x[0]) + np.average(Z_input[:, :, 0]),
            x[1] - x[0],
        ),
    )

    XYZ_flat_cartesian = np.vstack(
        (X_cartesian.flatten(), Y_cartesian.flatten(), Z_cartesian.flatten())
    ).T

    pred_wind_field_cartesian = gpr.predict(XYZ_flat_cartesian)

    u_cart = pred_wind_field_cartesian[:, 0].reshape(x.size, x.size, x.size)
    v_cart = pred_wind_field_cartesian[:, 1].reshape(x.size, x.size, x.size)
    w_cart = pred_wind_field_cartesian[:, 2].reshape(x.size, x.size, x.size)

    return X_cartesian, Y_cartesian, Z_cartesian, u_cart, v_cart, w_cart

def plot_field(X, Y, Z, u, v, w, terrain=np.asarray([]), z_plot_scale=1, fig=1):
    mlab.figure(fig)
    mlab.quiver3d(
        X,
        Y,
        z_plot_scale * Z,
        u,
        v,
        w,
    )

    if terrain.any():
        try:
            mlab.surf(
                X[:, :, 0].T,
                Y[:, :, 0].T,
                z_plot_scale * terrain.T,
                colormap="black-white",
            )
        except:
            mlab.surf(X.T, Y.T, z_plot_scale * terrain.T, colormap="black-white")

    mlab.show()

def plot_pressure(
    X,
    Y,
    Z,
    z_plot_scale,
    pressure,
    terrain=np.asarray([]),
    fig=2,
    surface=True,
    z_step=5,
):
    # for i in range(pressure[0, 0, :].size):
    #     pressure[:, :, i] = (pressure[:, :, i] - pressure[:, :, -1]) / (
    #         pressure[:, :, 0] - pressure[:, :, -1]
    #     )

    mlab.figure(fig)

    if surface == True:
        for i in range(0, X[0, 0, :].size, z_step):
            mlab.mesh(
                X[:, :, 0].T,
                Y[:, :, 0].T,
                z_plot_scale * Z[:, :, i].T,
                scalars=pressure[:, :, i].T,
            )

    else:
        mlab.points3d(X, Y, z_plot_scale * Z, pressure)
        if terrain.any():
            mlab.surf(
                X[:, :, 0].T,
                Y[:, :, 0].T,
                z_plot_scale * terrain.T,
                colormap="black-white",
            )

    mlab.show()


def preprosess_and_plot(
    terrain,
    x,
    y,
    z,
    u,
    v,
    w,
    pressure,
    x_dict,
    z_dict,
    z_plot_scale,
    time_index,
    interpolate=False,
    cartesian=False,
    interpolated_file_name="",
):
    terrain, x, y, X, Y, z, u, v, w, pressure = slice_data(
        terrain, x, y, z, u, v, w, pressure, x_dict, z_dict
    )

    # plot_images("./runs/3D_full_conv_test/images/val_imgs__it_44.pkl", terrain, X, Y, z[time_index], X_DICT, Z_DICT, z_plot_scale)

    if interpolate:
        # [print(x.shape) for x in [terrain, x, y, X, Y, z, u, v, w]]
        if cartesian:
            filename = "./saved_interpolated_cartesian_data/" + interpolated_file_name
            try:
                with open(filename, "rb") as f:
                    (
                        X_cartesian,
                        Y_cartesian,
                        Z_cartesian,
                        u_cart,
                        v_cart,
                        w_cart,
                    ) = pickle.load(f)
                print("Loaded cartesian data from file " + filename)
            except:
                (
                    X_cartesian,
                    Y_cartesian,
                    Z_cartesian,
                    u_cart,
                    v_cart,
                    w_cart,
                ) = interpolate_cartesian_3D(X, Y, z, u, v, w, x, y, time_index)

                with open(filename, "wb") as f:
                    pickle.dump(
                        [X_cartesian, Y_cartesian, Z_cartesian, u_cart, v_cart, w_cart],
                        f,
                    )
                print("Saved data to file " + filename)

                plot_field(
                    X_cartesian,
                    Y_cartesian,
                    Z_cartesian,
                    u_cart,
                    v_cart,
                    w_cart,
                    terrain,
                    z_plot_scale,
                    fig=3,
                )
        else:
            filename = "./saved_interpolated_z_data/" + interpolated_file_name

            (
                Z_interp_above_ground,
                u_interp,
                v_interp,
                w_interp,
                pressure_interp,
            ) = get_interpolated_z_data(filename, x, y, terrain, z, u, v, w, pressure)

            Z_interp = np.transpose(
                np.transpose(Z_interp_above_ground, ([2, 0, 1])) + terrain, ([1, 2, 0])
            )

            plot_field(
                X,
                Y,
                Z_interp,
                u_interp[time_index],
                v_interp[time_index],
                w_interp[time_index],
                terrain,
                z_plot_scale,
                fig=3,
            )

    plot_field(
        X,
        Y,
        z[time_index],
        u[time_index],
        v[time_index],
        w[time_index],
        terrain,
        z_plot_scale,
        fig=1,
    )

    plot_pressure(X, Y, z, z_plot_scale, pressure[time_index], fig=2)

def split_into_separate_files(z,
            u,
            v,
            w,
            pressure,
            filenames, 
            terrain,
            invalid_samples:set,
            folder = "./full_dataset_files/"):
    
    z_above_ground = np.transpose(np.transpose(z, ([0,3,1,2])) - terrain, ([0,2,3,1]))
    index=0
    for i in range(len(filenames)):
        if filenames[i] not in invalid_samples:
            if os.path.isfile(folder+filenames[i]):
                continue
            
            if np.isnan(np.concatenate((z[index],z_above_ground[index],u[index],v[index],w[index],pressure[index]))).any() or np.isinf(np.concatenate((z[index],z_above_ground[index],u[index],v[index],w[index],pressure[index]))).any() or u[index][u[index]>100].any() or v[index][v[index]>100].any() or w[index][w[index]>100].any() or pressure[index][pressure[index]>200000].any():
                invalid_samples.add(filenames[i])
                continue

            with open(folder+filenames[i], "wb") as f:
                pickle.dump(
                    [
                        z[index],
                        z_above_ground[index],
                        u[index],
                        v[index],
                        w[index],
                        pressure[index],
                    ],
                    f,
                )
            with open(folder+"max_"+filenames[i], "wb") as f:
                pickle.dump(
                    [
                        np.min(z[index]),
                        np.max(z[index]),
                        np.max(z_above_ground[index]),
                        np.max(np.concatenate((u[index],v[index],w[index]))),
                        np.max(pressure[index]),
                    ],
                    f,
                )
            index+=1
    return invalid_samples

def download_and_split(filenames, terrain, x_dict, y_dict, z_dict, folder = "./full_dataset_files/"):
    data_code = "simra_BESSAKER_"
    start_time = datetime.strptime(filenames[0][:-7], "%Y-%m-%d")
    end_time = datetime.strptime(filenames[-1][:-7], "%Y-%m-%d")
    days = (end_time - start_time).days+1
    transpose_indices = [0, 2, 3, 1]
    invalid_samples = set()
    for i in range(0, days, 5):
        start = i
        end = min(i+5, days)
        start_date = (start_time + timedelta(days=start)).date()
        end_date = (start_time + timedelta(days=end-1)).date()
        
        download_Bessaker_data(start_date, end_date, "./downloaded_raw_bessaker_data/")
        
        z, u, v, w, pressure, invalid_download_files = extract_slice_and_filter_3D(data_code, start_date, end_date, transpose_indices)

        for date, sim_time in invalid_download_files:
            new_names = filenames_from_start_and_end_dates(date, date)[:12] if sim_time == "T00Z.nc" else filenames_from_start_and_end_dates(date, date)[12:]
            [invalid_samples.add(name) for name in new_names]
        
        if u.any():
            z, u, v, w, pressure = slice_only_dim_dicts(z, u, v, w, pressure, x_dict=x_dict, y_dict=y_dict, z_dict=z_dict)
            invalid_samples = split_into_separate_files(z, u, v, w, pressure, filenames[24*start : 24*end], terrain, invalid_samples, folder=folder)
    
    return invalid_samples

def slice_dict_folder_name(x_dict, y_dict, z_dict):
    return (
        "x_"
        + str(x_dict["start"])
        + "_"
        + str(x_dict["max"])
        + "_"
        + str(x_dict["step"])
        + "___y_"
        + str(y_dict["start"])
        + "_"
        + str(y_dict["max"])
        + "_"
        + str(y_dict["step"])
        + "___z_"
        + str(z_dict["start"])
        + "_"
        + str(z_dict["max"])
        + "_"
        + str(z_dict["step"])
        + "/"
    )


if __name__ == "__main__":
    # start_date = date(2018, 4, 1)  # 1,2
    # end_date = date(2018, 4, 2)  #

    # # z, u, v, w, pressure, z_min, z_max, uvw_max, p_max, filename = download_and_combine(
    # #     start_date, end_date
    # # )

    z_plot_scale = 5
    time_index = 3
    # X_DICT = {"start": 4, "max": -3, "step": 1}
    # Z_DICT = {"start": 1, "max": 41, "step": 10}

    # file_name = interp_file_name(X_DICT, Z_DICT, start_date, end_date)




    # terrain, x, y, X, Y, z, u, v, w, pressure = slice_data(
    #     terrain, x, y, z, u, v, w, pressure, X_DICT, Z_DICT
    # )

    # plot_images("./runs/3D_full_conv_five_channels/images/val_imgs__it_40.pkl", terrain, X, Y, z[time_index], z_plot_scale=z_plot_scale)

    # z = (z.transpose(2,0,1)-terrain).transpose((1,2,0))
    # preprosess_and_plot(
    #     terrain,
    #     x,
    #     y,
    #     z,
    #     u,
    #     v,
    #     w,
    #     pressure,
    #     X_DICT,
    #     Z_DICT,
    #     z_plot_scale,
    #     time_index,
    #     interpolate=False,
    #     interpolated_file_name=file_name,
    # )

    # # mlab.quiver3d(X[:,:, z_start:z_lim:z_step],Y[:,:, z_start:z_lim:z_step],z_scale*z[:,:, z_start:z_lim:z_step],u_nomask[time_index][:, :, z_start:z_lim:z_step], v_nomask[time_index][:, :, z_start:z_lim:z_step], w_nomask[time_index][:,:, z_start:z_lim:z_step], mask_points=1)
    # # mlab.surf(X_2D, Y_2D, z_scale*terrain_nomask.transpose(), colormap="black-white")

    # """
    # delta_x = (X[135, 134] - X[0, 0])*10**5
    # delta_y = (Y[135, 134] - Y[0, 0])*10**5
    # X_res = delta_x/136
    # Y_res = delta_y/135
    # """
    # # saving u, v, w
    # # with open("2018_apr.pkl", "wb") as f:
    # #     pickle.dump([u, v, w], f)

    # """
    # with open('2018_january.pkl', 'rb') as f:
    # u_jan, v_jan, w_jan = pickle.load(f)
    # with open('2018_february.pkl', 'rb') as f:
    # u_feb, v_feb, w_feb = pickle.load(f)
    # with open('2018_march.pkl', 'rb') as f:
    # u_mar, v_mar, w_mar = pickle.load(f)
    # with open('2018_apr.pkl', 'rb') as f:
    # u_apr, v_apr, w_apr = pickle.load(f)

    # u = np.concatenate((u_jan,u_feb,u_mar, u_apr), axis = 0)
    # v = np.concatenate((v_jan,v_feb,v_mar, v_apr), axis = 0)
    # w = np.concatenate((w_jan,w_feb,w_mar, w_apr), axis = 0)
    # with open('download_data.pkl', 'wb') as f:
    #     pickle.dump([u, v, w], f)
    # """
