# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from datetime import timedelta, date
import datetime
from urllib import request
import os
import pickle
import glob
import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import num2date
import matplotlib.animation as animation
import matplotlib
import cftime
from tqdm import tqdm
from mayavi import mlab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

EARTH_RADIUS = 6371009

def plot_images(filename, 
    terrain,
    X,
    Y,
    Z,
    x_dict={"start": 4, "max": -3, "step": 1},
    z_dict={"start": 1, "max": 10, "step": 5},
    z_plot_scale=1,):

    with open(filename, "rb") as f:
        images = pickle.load(f)
    
    i=0
    for key, value in images.items():
        print(key)
        u, v, w = slice_only_dim_dicts(x_dict, z_dict, value[0], value[1], value[2])
        
        plot_field(np.squeeze(X), np.squeeze(Y), np.squeeze(Z), u,v,w, z_plot_scale=z_plot_scale, fig=i, terrain=terrain)
        
        i+=1


def check(url):
    try:
        u = request.urlopen(url)
        u.close()
        return True
    except:
        return False


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
            temp_date = datetime.datetime.strptime(str(temp), "%Y-%m-%d")
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
        np.transpose(nc_fid[key][1:13, :, :, :], (transpose_indices))[:, :, :, ::-1],
        axis=0,
    )


def extract_3D(data_code, start_date, end_date, transpose_indices=[0, 2, 3, 1]):
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
                z = np.transpose(
                    nc_fid["geopotential_height_ml"][:], (transpose_indices)
                )[:, :, :, ::-1]
                terrain = nc_fid["surface_altitude"][:]
                theta = np.transpose(
                    nc_fid["air_potential_temperature_ml"][:], (transpose_indices)
                )[:, :, :, ::-1]
                u = np.transpose(nc_fid["x_wind_ml"][:], (transpose_indices))[
                    :, :, :, ::-1
                ]
                v = np.transpose(nc_fid["y_wind_ml"][:], (transpose_indices))[
                    :, :, :, ::-1
                ]
                w = np.transpose(
                    nc_fid["upward_air_velocity_ml"][:], (transpose_indices)
                )[:, :, :, ::-1]
                pressure = np.transpose(
                    nc_fid["air_pressure_ml"][:], (transpose_indices)
                )[:, :, :, ::-1]
                # u10 = nc_fid['x_wind_10m'][:]
                # v10 = nc_fid['y_wind_10m'][:]
                tke = np.transpose(
                    nc_fid["turbulence_index_ml"][:], (transpose_indices)
                )[:, :, :, ::-1]
                td = np.transpose(
                    nc_fid["turbulence_dissipation_ml"][:], (transpose_indices)
                )[:, :, :, ::-1]
                index = index + 1
                nc_fid.close()
            else:
                nc_fid = Dataset(filename, mode="r")
                time = np.ma.append(time, nc_fid["time"][:][1:13], axis=0)
                u = quick_append(u, "x_wind_ml", nc_fid, transpose_indices)
                v = quick_append(v, "y_wind_ml", nc_fid, transpose_indices)
                w = quick_append(w, "upward_air_velocity_ml", nc_fid, transpose_indices)
                pressure = quick_append(
                    pressure, "air_pressure_ml", nc_fid, transpose_indices
                )
                theta = quick_append(
                    theta, "air_potential_temperature_ml", nc_fid, transpose_indices
                )
                tke = quick_append(
                    tke, "turbulence_index_ml", nc_fid, transpose_indices
                )
                td = quick_append(
                    td, "turbulence_dissipation_ml", nc_fid, transpose_indices
                )
                z = quick_append(z, "geopotential_height_ml", nc_fid, transpose_indices)
                nc_fid.close()
    return (
        time,
        latitude,
        longitude,
        terrain,
        x,
        y,
        z,
        u,
        v,
        w,
        theta,
        tke,
        td,
        pressure,
    )

def slice_only_dim_dicts(
    x_dict={"start": 4, "max": -3, "step": 1},
    z_dict={"start": 1, "max": 10, "step": 5},
*args):
    value_list = []
    for val in args:
        if val.ndim == 3:
            value_list.append(val[
                x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
                x_dict["start"] : x_dict["max"] : x_dict["step"],
                z_dict["start"] : z_dict["max"] : z_dict["step"],
            ])
        elif val.ndim == 2:
            value_list.append(val[
                x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
                x_dict["start"] : x_dict["max"] : x_dict["step"],
            ])
        elif val.ndim == 4:
            value_list.append(val[
                :,
                x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
                x_dict["start"] : x_dict["max"] : x_dict["step"],
                z_dict["start"] : z_dict["max"] : z_dict["step"],
            ])
    
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
    x_dict={"start": 4, "max": -90, "step": 1},
    z_dict={"start": -1, "max": -30, "step": -1},
):
    X, Y, _ = np.meshgrid(x, y, z[0, 0, :])
    X, Y = (
        100000
        * X[
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ],
        100000
        * Y[
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ],
    )
    u, v, w, pressure, z = [
        np.ma.filled(wind_field.astype(float), np.nan)[
            :,
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ]
        for wind_field in [u, v, w, pressure, z]
    ]

    terrain = np.ma.filled(terrain.astype(float), np.nan)
    terrain = terrain[
        x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
        x_dict["start"] : x_dict["max"] : x_dict["step"],
    ]

    return (
        terrain,
        100000 * x[x_dict["start"] : x_dict["max"] : x_dict["step"]],
        100000 * y[x_dict["start"] : x_dict["max"] - 1 : x_dict["step"]],
        X,
        Y,
        z,
        u,
        v,
        w,
        pressure,
    )


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
                X[:, :, 0].T, Y[:, :, 0].T, z_plot_scale * terrain.T, colormap="black-white"
            )
        except:
            mlab.surf(
                X.T, Y.T, z_plot_scale * terrain.T, colormap="black-white"
            )


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
    for i in range(pressure[0, 0, :].size):
        pressure[:, :, i] = (pressure[:, :, i] - pressure[:, :, -1]) / (
            pressure[:, :, 0] - pressure[:, :, -1]
        )

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
):
    terrain, x, y, X, Y, z, u, v, w, pressure = slice_data(
        terrain, x, y, z, u, v, w, pressure, x_dict, z_dict
    )

    plot_images("./runs/3D_full_conv_test/images/val_imgs__it_44.pkl", terrain, X, Y, z[time_index], X_DICT, Z_DICT, z_plot_scale)

    if interpolate:
        filename = (
            "./saved_interpolated_data/2018_apr_xy"
            + str(x_dict["start"])
            + "_"
            + str(x_dict["max"])
            + "_"
            + str(x_dict["step"])
            + "___z"
            + str(z_dict["start"])
            + "_"
            + str(z_dict["max"])
            + "_"
            + str(z_dict["step"])
            + "__time_"
            + str(time_index)
            + ".pickle"
        )
        [print(x.shape) for x in [terrain, x, y, X, Y, z, u, v, w]]
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
                    [X_cartesian, Y_cartesian, Z_cartesian, u_cart, v_cart, w_cart], f
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
    

    # plot_pressure(X, Y, z, z_plot_scale, pressure[time_index], fig=2)


def download_and_combine(data_code, start_date, end_date):
    filename = (
        "./combined_downloaded_bessaker_data/start_"
        + str(start_date)
        + "___end_"
        + str(end_date)
        + ".pickle"
    )

    try:
        with open(filename, "rb") as f:
            time, terrain, x, y, z, u, v, w, theta, tke, td, pressure = pickle.load(f)
        print("Already downloaded. Loaded data from file " + filename)

    except:
        download_Bessaker_data(start_date, end_date, "./downloaded_raw_bessaker_data/")
        transpose_indices = [0, 2, 3, 1]

        (
            time,
            latitude,
            longitude,
            terrain,
            x,
            y,
            z,
            u,
            v,
            w,
            theta,
            tke,
            td,
            pressure,
        ) = extract_3D(data_code, start_date, end_date, transpose_indices)

        with open(filename, "wb") as f:
            pickle.dump([time, terrain, x, y, z, u, v, w, theta, tke, td, pressure], f)
        print("Saved data to file " + filename)

    return time, terrain, x, y, z, u, v, w, theta, tke, td, pressure


if __name__ == "__main__":
    data_code = "simra_BESSAKER_"
    start_date = date(2018, 4, 1)  # 1,2
    end_date = date(2018, 4, 2)  #

    time, terrain, x, y, z, u, v, w, theta, tke, td, pressure = download_and_combine(
        data_code, start_date, end_date
    )

    z_plot_scale = 1
    time_index = 3
    X_DICT = {"start": 4, "max": 30, "step": 1}
    Z_DICT = {"start": 1, "max": 2, "step": 1}

    terrain, x, y, X, Y, z, u, v, w, pressure = slice_data(
        terrain, x, y, z, u, v, w, pressure, X_DICT, Z_DICT
    )

    plot_images("./runs/larger_pix_loss/images/val_imgs__it_200.pkl", terrain, X, Y, z[time_index], X_DICT, Z_DICT, z_plot_scale)
    
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
    # )

    # mlab.quiver3d(X[:,:, z_start:z_lim:z_step],Y[:,:, z_start:z_lim:z_step],z_scale*z[:,:, z_start:z_lim:z_step],u_nomask[time_index][:, :, z_start:z_lim:z_step], v_nomask[time_index][:, :, z_start:z_lim:z_step], w_nomask[time_index][:,:, z_start:z_lim:z_step], mask_points=1)
    # mlab.surf(X_2D, Y_2D, z_scale*terrain_nomask.transpose(), colormap="black-white")

    """
    delta_x = (X[135, 134] - X[0, 0])*10**5
    delta_y = (Y[135, 134] - Y[0, 0])*10**5
    X_res = delta_x/136
    Y_res = delta_y/135
    """
    # saving u, v, w
    # with open("2018_apr.pickle", "wb") as f:
    #     pickle.dump([u, v, w], f)

    """
    with open('2018_january.pickle', 'rb') as f:
    u_jan, v_jan, w_jan = pickle.load(f)
    with open('2018_february.pickle', 'rb') as f:
    u_feb, v_feb, w_feb = pickle.load(f)
    with open('2018_march.pickle', 'rb') as f:
    u_mar, v_mar, w_mar = pickle.load(f)
    with open('2018_apr.pickle', 'rb') as f:
    u_apr, v_apr, w_apr = pickle.load(f)

    u = np.concatenate((u_jan,u_feb,u_mar, u_apr), axis = 0)
    v = np.concatenate((v_jan,v_feb,v_mar, v_apr), axis = 0)
    w = np.concatenate((w_jan,w_feb,w_mar, w_apr), axis = 0)
    with open('download_data.pickle', 'wb') as f:
        pickle.dump([u, v, w], f)
    """
