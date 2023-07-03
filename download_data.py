"""
download_data.py
Written by Jacob Wulff Wold 2023
Apache License

Downloads, preprocesses and saves the data.
"""


from datetime import datetime, timedelta, date
from urllib import request
import os
import pickle
import netCDF4
from netCDF4 import Dataset
import numpy as np
import torch


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
    for i in range((delta.days + 1) * 24):
        names.append(
            (str(start_time + timedelta(hours=i)) + ".pkl")
            .replace(" ", "-")
            .replace(":00:00", "")
        )

    return names


def download_Bessaker_data(start_date, end_date, destination_folder, invalid_urls):
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
                if filename not in invalid_urls:
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
                                "Number of files downloaded ",
                                counter,
                                "/",
                                no_data_points,
                            )
                        else:
                            print("File not found")
                            with open(
                                "./data/downloaded_raw_bessaker_data/invalid_files.txt",
                                "a",
                            ) as f:
                                f.write(filename + "\n")

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


def quick_append(var, key, nc_fid, transpose_indices=[0, 2, 3, 1]):
    return np.ma.append(
        var,
        np.transpose(nc_fid[key][:-1, :, :, :], (transpose_indices))[:, :, :, ::-1],
        axis=0,
    )


def get_static_data():
    filename = os.listdir("./data/downloaded_raw_bessaker_data/")[0]

    nc_fid = Dataset("./data/downloaded_raw_bessaker_data/" + filename, mode="r")

    x = 100000 * nc_fid["x"][:]
    y = 100000 * nc_fid["y"][:]
    terrain = nc_fid["surface_altitude"][:]
    nc_fid.close()

    terrain = np.ma.filled(terrain.astype(float), np.nan)
    terrain, x, y = slice_only_dim_dicts(terrain, x, y)

    with open("./data/full_dataset_files/static_terrain_x_y.pkl", "wb") as f:
        pickle.dump([terrain, x, y], f)


def extract_slice_and_filter_3D(
    data_code, start_date, end_date, transpose_indices=[0, 2, 3, 1]
):
    delta = end_date - start_date
    sim_times = ["T00Z.nc", "T12Z.nc"]
    index = 0
    invalid_filenames = []
    for i in range(delta.days + 1):
        for sim_time in sim_times:
            temp = start_date + timedelta(days=i)
            filename = data_code + ((str(temp)).replace("-", ""))
            filename = "./data/downloaded_raw_bessaker_data/" + filename + sim_time
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
                    w = quick_append(
                        w, "upward_air_velocity_ml", nc_fid, transpose_indices
                    )
                    pressure = quick_append(
                        pressure, "air_pressure_ml", nc_fid, transpose_indices
                    )
                    z = quick_append(
                        z, "geopotential_height_ml", nc_fid, transpose_indices
                    )
                    nc_fid.close()
            except:
                invalid_filenames.append((temp, sim_time))

    if "pressure" in locals():
        u, v, w, pressure, z = [
            np.ma.filled(wind_field.astype(float), np.nan)
            for wind_field in [u, v, w, pressure, z]
        ]
        z, u, v, w, pressure = slice_only_dim_dicts(z, u, v, w, pressure)
    else:
        return (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            invalid_filenames,
        )

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
    xy_count = 0
    for val in args:
        if val.ndim == 3:
            value_list.append(
                val[
                    x_dict["start"] : x_dict["max"] : x_dict["step"],
                    y_dict["start"] : y_dict["max"] : y_dict["step"],
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
            if xy_count == 0:
                value_list.append(val[x_dict["start"] : x_dict["max"] : x_dict["step"]])
                xy_count += 1
            else:
                value_list.append(val[y_dict["start"] : y_dict["max"] : y_dict["step"]])

    return value_list


def reverse_interpolate_z_axis(
    HR_interp,
    Z_raw,
    Z_interp,
):
    HR_no_interp = np.zeros_like(HR_interp)
    for x in range(HR_interp.shape[0]):
        for z in range(HR_interp.shape[1]):
            for i in range(HR_interp.shape[2]):
                for j in range(HR_interp.shape[3]):
                    HR_no_interp[x, z, i, j, :] = np.interp(
                        Z_raw[x, 0, i, j, :],
                        Z_interp[x, 0, i, j, :],
                        HR_interp[x, z, i, j, :],
                    )

    return torch.from_numpy(HR_no_interp)


def interpolate_z_axis(
    x,
    y,
    z_above_ground,
    u,
    v,
    w,
    pressure,
    terrain,
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
            w[i, j, :] = np.interp(
                new_1D_z_above_ground, z_above_ground[i, j, :], w[i, j, :]
            )
            pressure[i, j, :] = np.interp(
                new_1D_z_above_ground,
                z_above_ground[i, j, :],
                pressure[i, j, :],
            )
    z = np.transpose(
        np.transpose(new_3D_z_above_ground, ([2, 0, 1])) + terrain, ([1, 2, 0])
    )

    return z, new_3D_z_above_ground, u, v, w, pressure


def get_interpolated_z_data(
    filename,
    x,
    y,
    z_above_ground,
    u,
    v,
    w,
    pressure,
    terrain,
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
            z,
            Z_interp_above_ground,
            u,
            v,
            w,
            pressure,
        ) = interpolate_z_axis(x, y, z_above_ground, u, v, w, pressure, terrain)

        with open(filename, "wb") as f:
            pickle.dump(
                [z, Z_interp_above_ground, u, v, w, pressure],
                f,
            )
        # print("Saved data to file " + filename)

    return z, Z_interp_above_ground, u, v, w, pressure


def split_into_separate_files(
    z,
    u,
    v,
    w,
    pressure,
    filenames,
    terrain,
    invalid_samples: set,
    folder="./data/full_dataset_files/",
):
    z_above_ground = np.transpose(
        np.transpose(z, ([0, 3, 1, 2])) - terrain, ([0, 2, 3, 1])
    )
    index = 0
    for i in range(len(filenames)):
        if filenames[i] not in invalid_samples:
            if os.path.isfile(folder + "max/max_" + filenames[i]):
                continue

            if (
                np.isnan(
                    np.concatenate(
                        (
                            z[index],
                            z_above_ground[index],
                            u[index],
                            v[index],
                            w[index],
                            pressure[index],
                        )
                    )
                ).any()
                or np.isinf(
                    np.concatenate(
                        (
                            z[index],
                            z_above_ground[index],
                            u[index],
                            v[index],
                            w[index],
                            pressure[index],
                        )
                    )
                ).any()
                or u[index][u[index] > 100].any()
                or v[index][v[index] > 100].any()
                or w[index][w[index] > 100].any()
                or pressure[index][pressure[index] > 200000].any()
            ):
                invalid_samples.add(filenames[i])
                continue

            with open(folder + filenames[i], "wb") as f:
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
            with open(folder + "max/max_" + filenames[i], "wb") as f:
                pickle.dump(
                    [
                        np.min(z),
                        np.max(z),
                        np.max(z_above_ground),
                        np.max(np.concatenate((u, v, w))),
                        np.min(pressure),
                        np.max(pressure),
                    ],
                    f,
                )
            index += 1
    return invalid_samples


def download_and_split(
    filenames,
    terrain,
    x_dict,
    y_dict,
    z_dict,
    invalid_urls,
    folder="./data/full_dataset_files/",
):
    data_code = "simra_BESSAKER_"
    start_time = datetime.strptime(filenames[0][:-7], "%Y-%m-%d")
    end_time = datetime.strptime(filenames[-1][:-7], "%Y-%m-%d")
    days = (end_time - start_time).days + 1
    transpose_indices = [0, 2, 3, 1]
    invalid_samples = set()
    for i in range(0, days, 5):
        start = i
        end = min(i + 5, days)
        start_date = (start_time + timedelta(days=start)).date()
        end_date = (start_time + timedelta(days=end - 1)).date()

        download_Bessaker_data(
            start_date,
            end_date,
            "./data/downloaded_raw_bessaker_data/",
            invalid_urls,
        )

        z, u, v, w, pressure, invalid_download_files = extract_slice_and_filter_3D(
            data_code, start_date, end_date, transpose_indices
        )

        for date, sim_time in invalid_download_files:
            new_names = (
                filenames_from_start_and_end_dates(date, date)[:12]
                if sim_time == "T00Z.nc"
                else filenames_from_start_and_end_dates(date, date)[12:]
            )
            [invalid_samples.add(name) for name in new_names]

        if u.any():
            z, u, v, w, pressure = slice_only_dim_dicts(
                z, u, v, w, pressure, x_dict=x_dict, y_dict=y_dict, z_dict=z_dict
            )
            invalid_samples = split_into_separate_files(
                z,
                u,
                v,
                w,
                pressure,
                filenames[24 * start : 24 * end],
                terrain,
                invalid_samples,
                folder=folder,
            )

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
