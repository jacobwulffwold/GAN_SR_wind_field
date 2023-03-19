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
    destination_folder = "./"
    no_data_points = (delta.days + 1) * 2
    counter = 0
    for i in range(delta.days + 1):
        for sim_time in sim_times:
            temp = start_date + timedelta(days=i)
            temp_date = datetime.datetime.strptime(str(temp), "%Y-%m-%d")
            filename = data_code + ((str(temp)).replace("-", ""))
            filename = destination_folder + filename + sim_time
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
            if os.path.isfile(filename) == True:
                counter = counter + 1
                print("Number of files downloaded ", counter, "/", no_data_points)
            elif os.path.isfile(filename) != True:
                print("Attempting to download file ", filename)
                try:
                    if check(URL):
                        request.urlretrieve(URL, "./" + filename)
                        print("Downloading file", filename)
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
            filename = filename + sim_time
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
            filename = filename + sim_time
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


def extract_3D(data_code, start_date, end_date):
    delta = end_date - start_date
    sim_times = ["T00Z.nc", "T12Z.nc"]
    index = 0
    for i in range(delta.days + 1):
        for sim_time in sim_times:
            temp = start_date + timedelta(days=i)
            filename = data_code + ((str(temp)).replace("-", ""))
            filename = filename + sim_time
            if index == 0:
                nc_fid = Dataset(filename, mode="r")
                time = nc_fid["time"][:]
                latitude = nc_fid["longitude"][:]
                longitude = nc_fid["latitude"][:]
                x = nc_fid["x"][:]
                y = nc_fid["y"][:]
                z = np.transpose(nc_fid["geopotential_height_ml"][1], (1, 2, 0))
                terrain = nc_fid["surface_altitude"][:]
                theta = np.transpose(
                    nc_fid["air_potential_temperature_ml"][:], (0, 2, 3, 1)
                )
                u = np.transpose(nc_fid["x_wind_ml"][:], (0, 2, 3, 1))
                v = np.transpose(nc_fid["y_wind_ml"][:], (0, 2, 3, 1))
                w = np.transpose(nc_fid["upward_air_velocity_ml"][:], (0, 2, 3, 1))
                # u10 = nc_fid['x_wind_10m'][:]
                # v10 = nc_fid['y_wind_10m'][:]
                tke = np.transpose(nc_fid["turbulence_index_ml"][:], (0, 2, 3, 1))
                td = np.transpose(nc_fid["turbulence_dissipation_ml"][:], (0, 2, 3, 1))
                index = index + 1
                nc_fid.close()
            else:
                nc_fid = Dataset(filename, mode="r")
                time = np.ma.append(time, nc_fid["time"][:][1:13], axis=0)
                u = np.ma.append(
                    u,
                    np.transpose(nc_fid["x_wind_ml"][1:13, :, :, :], (0, 2, 3, 1)),
                    axis=0,
                )
                v = np.ma.append(
                    v,
                    np.transpose(nc_fid["y_wind_ml"][:][1:13, :, :, :], (0, 2, 3, 1)),
                    axis=0,
                )
                w = np.ma.append(
                    w,
                    np.transpose(
                        nc_fid["upward_air_velocity_ml"][:][1:13, :, :, :], (0, 2, 3, 1)
                    ),
                    axis=0,
                )
                theta = np.ma.append(
                    theta,
                    np.transpose(
                        nc_fid["air_potential_temperature_ml"][:][1:13, :, :, :],
                        (0, 2, 3, 1),
                    ),
                    axis=0,
                )
                tke = np.ma.append(
                    tke,
                    np.transpose(
                        nc_fid["turbulence_index_ml"][:][1:13, :, :, :], (0, 2, 3, 1)
                    ),
                    axis=0,
                )
                td = np.ma.append(
                    td,
                    np.transpose(
                        nc_fid["turbulence_dissipation_ml"][:][:][1:13, :, :, :],
                        (0, 2, 3, 1),
                    ),
                    axis=0,
                )
                # z     = np.ma.append(z,nc_fid['geopotential_height_ml'][:][:][1:13,:,:,:],axis=0)
                nc_fid.close()
    return time, latitude, longitude, terrain, x, y, z, u, v, w, theta, tke, td



x_dict={"start": 4, "max": -3, "step": 1},

z_dict={"start": -1, "max": -41, "step": -1},


def slice_data(
    terrain,
    x,
    y,
    z,
    u,
    v,
    w,
    x_dict={"start": 4, "max": -3, "step": 1},
    z_dict={"start": -1, "max": -41, "step": -1},
):
    X, Y, _ = np.meshgrid(x, y, z[0, 0, :])
    X, Y, z = (
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
        z[
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ],
    )
    u, v, w = [
        np.ma.filled(wind_field.astype(float), np.nan)[
            :,
            x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
            x_dict["start"] : x_dict["max"] : x_dict["step"],
            z_dict["start"] : z_dict["max"] : z_dict["step"],
        ]
        for wind_field in [u, v, w]
    ]

    terrain = np.ma.filled(terrain.astype(float), np.nan)
    terrain = terrain[
        x_dict["start"] : x_dict["max"] - 1 : x_dict["step"],
        x_dict["start"] : x_dict["max"] : x_dict["step"],
    ]

    return terrain, X, Y, z, u, v, w


def interpolate_cartesian_3D(X_input, Y_input, Z_input, output_dimension=128):

    XYZsigma_flat = np.vstack((X_input.flatten(), Y_input.flatten(), Z_input.flatten())).T
    y_wind_field = np.vstack((u[0].flatten(), v[0].flatten(), w[0].flatten())).T


    kernel = RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)
    )


    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)

    gpr.fit(XYZsigma_flat, y_wind_field)

    X_cartesian, Y_cartesian, Z_cartesian = np.meshgrid(
        100000 * x[4:-3], 100000 * y[4:-4], np.arange(0, 2000, 128)
    )

    X_flat_cartesian = np.vstack(
        (X_cartesian.flatten(), Y_cartesian.flatten(), Z_cartesian.flatten())
    ).T

    u_cart, v_cart, w_cart = gpr.predict(X_flat_cartesian)



data_code = "simra_BESSAKER_"
start_date = date(2018, 4, 1)  # 1,2
end_date = date(2018, 4, 3)  #
download_Bessaker_data(start_date, end_date, "./")
# combine_files(data_code,start_date,end_date,outfilename='temp.pkl')

time, latitude, longitude, terrain, x, y, z, u, v, w, theta, tke, td = extract_3D(
    data_code, start_date, end_date
)

[print(x.shape) for x in [terrain, x, y, z, u, v, w]]

z_max = z[0, 0, :].size
z_lim = z_max
z_step = 35
z_start = 1
z_plot_scale = 3


terrain, X, Y, z, u, v, w = slice_data(terrain, x, y, z, u, v, w)


[print(x.shape) for x in [terrain, X, Y, z, u, v, w]]


# X_2D, Y_2D = np.meshgrid(x,y)

# ax = plt.figure().add_subplot(projection='3d')

# ax.quiver(X[::10, ::10, ::10],Y[::10, ::10, ::10],z[::10, ::10, ::10],u[0][::10, ::10, ::10], v[0][::10, ::10, ::10], w[0][::10, ::10, ::10])

# mlab.quiver3d(X,Y,z, u[0], v[0], w[0])


# v = np.ma.filled(v.astype(float), np.nan)
# v = v[:, 4:-4, 4:-3, ::-1]

# w = np.ma.filled(w.astype(float), np.nan)
# w = w[:, 4:-4, 4:-3, ::-1]

# terrain = np.ma.filled(terrain.astype(float), np.nan)
# terrain = terrain[4:-4, 4:-3]

# z = z[4:-4, 4:-3, ::-1]

# X, Y = 100000 * X[4:-4, 4:-3, :], 100000 * Y[4:-4, 4:-3, :]
# X_2D, Y_2D = 100000*X_2D[4:-4, 4:-3].transpose(), 100000*Y_2D[4:-4, 4:-3].transpose()


# mlab.quiver3d(X[:,:, z_start:z_lim:z_step],Y[:,:, z_start:z_lim:z_step],z_scale*z[:,:, z_start:z_lim:z_step],u_nomask[0][:, :, z_start:z_lim:z_step], v_nomask[0][:, :, z_start:z_lim:z_step], w_nomask[0][:,:, z_start:z_lim:z_step], mask_points=1)
# # mlab.surf(X_2D, Y_2D, z_scale*terrain_nomask.transpose(), colormap="black-white")

# # for i in range(z_start,z_max,z_step):
# #     mlab.mesh(X_2D, Y_2D, z_scale*z[:, :, i].transpose(), scalars=np.sqrt(u_nomask[0][:,:,i]**2+v_nomask[0][:,:,i]**2+w_nomask[0][:,:,i]**2).transpose())
# mlab.show()


# X, Y = np.meshgrid(x, y)
# plt.contour(z)
# plt.contourf(u[1,:,:],cmap='hsv')
# plt.quiver(u[10,:,:],v[10,:,:],scale=200)
# plt.show()
# print(np.max(u),np.max(v),np.max(w),np.max(theta))
# for i in tqdm(range(135)):
#     for j in range(136):
#         for k in range(41):
#             vars[:,k,j,i] = np.var(z[:,k,j,i])

# print(vars)
# print(np.average(vars))
# print(np.median(vars))
# print(np.max(vars))




"""
delta_x = (X[135, 134] - X[0, 0])*10**5
delta_y = (Y[135, 134] - Y[0, 0])*10**5
X_res = delta_x/136
Y_res = delta_y/135
"""
# saving u, v, w
with open("2018_apr.pickle", "wb") as f:
    pickle.dump([u, v, w], f)


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
