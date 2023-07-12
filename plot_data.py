"""
plot_data.py
Written by Jacob Wulff Wold 2023
Apache License

Plotting functions employed in the master thesis
"""

from mayavi import mlab
from mayavi.mlab import pipeline as pipe
from tvtk.api import tvtk
import numpy as np
import torch.nn as nn
import pickle as pkl
from process_data import preprosess, get_static_data
from datetime import date
from download_data import slice_only_dim_dicts, reverse_interpolate_z_axis
from GAN_models.wind_field_GAN_3D import compute_PSNR_for_SR_and_trilinear
from tqdm import tqdm
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import os
from cycler import cycler
from CNN_models.Generator_3D_Resnet_ESRGAN import Generator_3D
import pandas as pd
import pickle as pkl
import torch
from config.config import Config
from GAN_models.wind_field_GAN_3D import wind_field_GAN_3D
import copy

Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MIN, P_MAX = [
    -2.7072315216064453,
    550.43798828125,
    68.45956420898438,
    32.32976150512695,
    0.0,
    105182.828125,
]


name_dict = {
    "Z_handling90only_wind": r"only_wind",
    "Z_handling90wind_rawZ": r"$z$_channel",
    "Z_handling90wind_pressure": r"$p$_channel",
    "SCH100_schedule2_larger_grad": r"$p\_z$_channels",
    "Z_handling90wind_rawZ_pressure": r"$p\_z$_channels",
    "Z_handling90wind_Zground": r"$z_{ground}$_channels",
    "Z_handling90wind_Zground_pressure": r"$p\_z_{ground}$_channels",
    "Z90_interponly_wind": r"only_wind_interp",
    "Z90_interpwind_interpZ": r"$z$_channel_interp",
    "Z90_interpwind_pressure": r"$p$_channel_interp",
    "Z90_interpwind_interpZ_pressure": r"$z\_p$_channels_interp",
    "Z_handling90_seedonly_wind": r"only_wind",
    "Z_handling90_seedwind_rawZ": r"$z$_channel",
    "Z_handling90_seedwind_pressure": r"$p$_channel",
    "Z_handling90_seed_wind_rawZ_pressure": r"$p\_z$_channels",
    "Z_handling90_seedwind_rawZ_pressure": r"$p\_z$_channels",
    "Z100_seed_wind_Zground": r"$z_{ground}$_channels",
    "Z100_seed_wind_Zground_pressure": r"$p\_z_{ground}$_channels",
    "Z_handling90_seedwind_Zground": r"$z_{ground}$_channels",
    "Z_handling90_seedwind_Zground_pressure": r"$p\_z_{ground}$_channels",
    "Z90_interp_seedonly_wind": r"only_wind_interp",
    "Z90_interp_seedwind_interpZ": r"$z$_channel_interp",
    "Z90_interp_seedwind_pressure": r"$p$_channel_interp",
    "Z90_interp_seedwind_interpZ_pressure": r"$z\_p$_channels_interp",
    "C100_only_pix": r"only_pix_cost",
    "C100_grad": r"grad_cost",
    "C100_div": r"div_cost",
    "C100_xy": r"$xy$_cost",
    "C100_grad_large": r"large_grad_cost",
    "C100_div_large": r"large_div_cost",
    "C100_xy_large": r"large_$xy$_cost",
    "C100_seed_only_pix": r"only_pix_cost",
    "C100_seed_grad": r"grad_cost",
    "C100_seed_div": r"div_cost",
    "C100_seed_xy": r"$xy$_cost",
    "C100_seed_grad_large": r"large_grad_cost",
    "C100_seed_div_large": r"large_div_cost",
    "C100_seed_xy_large": r"large_$xy$_cost",
    "STD": "std_cost",
    "STD_seed": "std_cost",
}

loss_name_dict = {
    "xy_gradient": r"$L_G^{\nabla xy}$",
    "xy_divergence": r"$L_G^{div_{xy}}$",
    "z_gradient": r"$L_G^{\nabla z}$",
    "divergence": r"$L_G^{div}$",
    "pix": r"$L_G^{pix}$",
    "adversarial": r"$L_G^{adversarial}$",
}


def plot_field(
    X,
    Y,
    Z,
    u,
    v,
    w,
    terrain=np.asarray([]),
    z_plot_scale=1,
    fig=1,
    colormap="viridis",
    terrainX=np.asarray([]),
    terrainY=np.asarray([]),
    max_value=None,
    title="",
):
    mlab.figure(fig, bgcolor=(1, 1, 1))
    field = mlab.quiver3d(
        X,
        Y,
        z_plot_scale * Z,
        u,
        v,
        w,
        colormap=colormap,
    )

    if terrain.any():
        try:
            if terrainX.any():
                mlab.surf(
                    terrainX,
                    terrainY,
                    z_plot_scale * terrain,
                    colormap="black-white",
                    opacity=0.5,
                )
            else:
                mlab.surf(
                    X[:, :, 0],
                    Y[:, :, 0],
                    z_plot_scale * terrain,
                    colormap="black-white",
                    opacity=0.5,
                )
        except:
            mlab.surf(X, Y, z_plot_scale * terrain, colormap="black-white", opacity=0.5)

    if max_value is not None:
        lut_manager = field.module_manager.vector_lut_manager
        lut_manager.data_range = (0, max_value)

    if title:
        # mlab.title(title, color=(0.2, 0.2, 0.2))
        mlab.vectorbar(
            field, title=title + " [m/s]", orientation="vertical", nb_labels=5
        )
    else:
        mlab.vectorbar(
            field, title="Wind speed [m/s]", orientation="vertical", nb_labels=5
        )

    mlab.show()


def create_error_figure(
    wind_height_index,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_TL,
    average_SR_error,
    average_TL_error,
    average_SR_error_relative,
    average_TL_error_relative,
):
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("viridis"))
    vmin, vmax = np.min(wind_comp_HR[:, :, wind_height_index]), np.max(
        wind_comp_HR[:, :, wind_height_index]
    )
    vmin_wind_field, vmax_wind_field = np.min(
        np.concatenate(
            (
                wind_comp_TL[:, :, wind_height_index],
                wind_comp_SR[:, :, wind_height_index],
            ),
            axis=(0),
        )
    ), np.max(
        np.concatenate(
            (
                wind_comp_TL[:, :, wind_height_index],
                wind_comp_SR[:, :, wind_height_index],
            ),
            axis=(0),
        )
    )
    vmin_error, vmax_error = np.min(
        np.concatenate(
            (
                wind_comp_TL[:, :, wind_height_index]
                - wind_comp_HR[:, :, wind_height_index],
                wind_comp_SR[:, :, wind_height_index]
                - wind_comp_HR[:, :, wind_height_index],
            ),
            axis=(0),
        )
    ), np.max(
        np.concatenate(
            (
                wind_comp_TL[:, :, wind_height_index]
                - wind_comp_HR[:, :, wind_height_index],
                wind_comp_SR[:, :, wind_height_index]
                - wind_comp_HR[:, :, wind_height_index],
            ),
            axis=(0),
        )
    )
    vmin_abs_error, vmax_abs_error = 0.0, max(abs(vmax_error), abs(vmin_error))
    sm.set_clim(vmin=vmin, vmax=vmax)
    sm_error = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("coolwarm"))
    sm_error.set_clim(vmin=vmin_error, vmax=vmax_error)
    sm_abs_error = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("jet"))
    sm_abs_error.set_clim(vmin=vmin_abs_error, vmax=vmax_abs_error)

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex=True)
    axes2[0, 1].pcolor(
        wind_comp_SR[:, :, wind_height_index],
        vmin=vmin_wind_field,
        vmax=vmax_wind_field,
        cmap="viridis",
        edgecolor="none",
    )
    axes2[0, 1].set_title(
        f"SR, avg error: {round(average_SR_error,3)} m/s ({round(100*average_SR_error_relative,1)}% of average)"
    )
    axes2[0, 0].pcolor(
        wind_comp_SR[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index],
        vmin=vmin_error,
        vmax=vmax_error,
        cmap="coolwarm",
    )
    axes2[0, 0].set_title("Error SR-HR (m/s)")
    axes2[0, 2].pcolor(
        abs(
            wind_comp_HR[:, :, wind_height_index]
            - wind_comp_SR[:, :, wind_height_index]
        ),
        vmin=vmin_abs_error,
        vmax=vmax_abs_error,
        cmap="jet",
        edgecolor="none",
    )
    axes2[0, 2].set_title("SR Absolute Error (m/s)")
    fig2.colorbar(sm, ax=axes2[0, 1])
    fig2.colorbar(
        sm_error,
        ax=axes2[0, 0],
    )

    fig2.colorbar(
        sm_abs_error,
        ax=axes2[0, 2],
    )
    axes2[1, 1].pcolor(wind_comp_TL[:, :, wind_height_index])
    axes2[1, 1].set_title(
        f"TL, avg error: {round(average_TL_error,3)} m/s ({round(100*average_TL_error_relative,1)}% of average)"
    )
    axes2[1, 0].pcolor(
        wind_comp_TL[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index],
        cmap="coolwarm",
    )
    axes2[1, 0].set_title("Error TL-HR (m/s)")
    axes2[1, 2].pcolor(
        abs(
            wind_comp_HR[:, :, wind_height_index]
            - wind_comp_TL[:, :, wind_height_index]
        ),
        cmap="jet",
        edgecolor="none",
    )
    axes2[1, 2].set_title("TL Absolute Error (m/s)")
    fig2.colorbar(sm, ax=axes2[1, 1])
    fig2.colorbar(
        sm_error,
        ax=axes2[1, 0],
    )

    fig2.colorbar(
        sm_abs_error,
        ax=axes2[1, 2],
    )
    fig2.subplots_adjust(hspace=0.2)
    return fig2


def create_comparison_figure(
    wind_height_index,
    wind_comp_LR,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_TL,
):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    vmin, vmax = np.min(wind_comp_HR[:, :, wind_height_index]), np.max(
        wind_comp_HR[:, :, wind_height_index]
    )
    axes[0, 0].pcolor(
        wind_comp_LR[:, :, wind_height_index],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        edgecolor="none",
    )
    axes[0, 0].set_title("LR")
    axes[0, 1].pcolor(
        wind_comp_HR[:, :, wind_height_index],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        edgecolor="none",
    )
    axes[0, 1].set_title("HR")
    axes[1, 1].pcolor(
        wind_comp_SR[:, :, wind_height_index],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        edgecolor="none",
    )
    axes[1, 1].set_title("SR")
    axes[1, 0].pcolor(
        wind_comp_TL[:, :, wind_height_index],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        edgecolor="none",
    )
    axes[1, 0].set_title("TL")
    fig.subplots_adjust(hspace=0.3)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("viridis"))
    sm.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(sm, ax=axes)
    return fig


def plot_metrics1(
    tb_folder,
    metric,
    metric_folder,
    ax: plt.Axes,
    df=pd.DataFrame([]),
    title=None,
    ylabel=None,
    xlabel=None,
):
    folders = os.listdir(tb_folder)
    folders.sort()
    if title:
        ax.set_title(title)
    # else:
    #     fig.title(tb_folder[tb_folder.rfind("/")+1:])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if df.empty:
        df = SummaryReader(tb_folder, pivot=True, extra_columns={"dir_name"}).scalars

    color_cycle = cycler(
        color=[
            "#E24A33",
            "#348ABD",
            "#988ED5",
            "#777777",
            "#FBC15E",
            "#8EBA42",
            "#FFB5B8",
            "#56B4E9",
            "#7FCE6C",
            "#B37CAB",
        ]
    )
    ax.set_prop_cycle(color_cycle)

    for folder in folders:
        if folder[0] == ".":
            continue
        this_df = df[
            df["dir_name"].str.contains(folder + "/")
            & df["dir_name"].str.contains(metric)
        ]
        ax.plot(
            this_df[this_df["step"] < 90001]["step"],
            this_df[this_df["step"] < 90001][metric_folder].apply(
                lambda x: x[0] if isinstance(x, list) else x
            ),
            label=name_dict[folder],
        )

    if metric == "val_PSNR":
        trilinear_metric = "Trilinear_PSNR"
    elif metric == "pix_loss_unscaled":
        trilinear_metric = "trilinear_pix_loss"

    this_df = df[
        df["dir_name"].str.contains(folder + "/")
        & df["dir_name"].str.contains(trilinear_metric)
    ]
    ax.plot(
        this_df[this_df["step"] < 90001]["step"],
        this_df[this_df["step"] < 90001][metric_folder].apply(
            lambda x: x[0] if isinstance(x, list) else x
        ),
        label="Trilinear interpolation",
        color="#CCCCCC",
        linestyle="--",
    )


def plot_metrics2(
    tb_folder,
    metric,
    metric_folder,
    ax: plt.Axes,
    df=pd.DataFrame([]),
    title=None,
    ylabel=None,
    xlabel=None,
):
    folders = os.listdir(tb_folder)
    folders.sort()
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if df.empty:
        df = SummaryReader(tb_folder, pivot=True, extra_columns={"dir_name"}).scalars

    color_cycle = cycler(
        color=[
            "#E24A33",
            "#348ABD",
            "#988ED5",
            "#777777",
            "#FBC15E",
            "#8EBA42",
            "#FFB5B8",
            "#56B4E9",
            "#7FCE6C",
            "#B37CAB",
        ]
    )
    ax.set_prop_cycle(color_cycle)

    for folder in folders:
        if folder[0] == ".":
            continue
        this_df = df[
            df["dir_name"].str.contains(folder + "/")
            & df["dir_name"].str.contains(metric)
        ]
        ax.plot(
            this_df["step"],
            this_df[metric_folder].apply(lambda x: x[0] if isinstance(x, list) else x),
            label=name_dict[folder],
        )

    if metric == "val_PSNR":
        trilinear_metric = "Trilinear_PSNR"
    elif metric == "pix_loss_unscaled":
        trilinear_metric = "trilinear_pix_loss"

    this_df = df[
        df["dir_name"].str.contains(folder + "/")
        & df["dir_name"].str.contains(trilinear_metric)
    ]
    ax.plot(
        this_df[this_df["step"] < 90001]["step"],
        this_df[this_df["step"] < 90001][metric_folder].apply(
            lambda x: x[0] if isinstance(x, list) else x
        ),
        label="Trilinear interpolation",
        color="#CCCCCC",
        linestyle="--",
    )


def create_best_exp25_plot():
    plt.style.use("ggplot")
    plt.rcParams.update({"font.family": "Helvetica"})
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    tb_folder = "/Volumes/jawold/GAN_SR_wind_field/tensorboard_log/35kGAN_cost_param_search/dict_values([30.63578914390694, 1.85622829372303, 7.21417216452547, 3.664861906371173, 0.33632239425077304])"
    G_val_folder = "G_loss/validation"
    losses = [
        "xy_gradient",
        "xy_divergence",
        "z_gradient",
        "divergence",
        "pix",
        "adversarial",
    ]
    df = SummaryReader(tb_folder, pivot=True, extra_columns={"dir_name"}).scalars

    for loss in losses:
        this_df = df[
            df["dir_name"].str.contains(G_val_folder)
            & df["dir_name"].str.contains("/" + loss)
        ]
        ax.plot(
            this_df["step"],
            this_df[G_val_folder].apply(lambda x: x[0] if isinstance(x, list) else x),
            label=loss_name_dict[loss],
        )
    ax.legend()
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Validation loss")
    fig.suptitle(
        r"Decomposed validation loss for $L_G^*$", fontweight="bold", fontsize=20
    )
    ax.set_ylim(0.0, 0.06)
    fig.savefig("./figures/exp25Best.pdf", format="pdf", bbox_inches="tight")


def create_norm_plot():
    plt.style.use("ggplot")
    plt.rcParams.update({"font.family": "Helvetica"})
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    with open("./norm.csv", "r") as f:
        norm = f.readlines()
    norm = [float(x) for x in norm]
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Norm")
    fig.suptitle(r"Norm during training with $L_G^*$", fontweight="bold", fontsize=20)
    ax.set_ylim(0.0, 5.0)
    ax.plot(2 * range(len(norm)), norm)
    fig.savefig("./figures/norm.pdf", format="pdf", bbox_inches="tight")


def plot_metrics25(
    tb_folder,
    metric,
    metric2,
    metric_folder,
    metric_folder2,
    df=pd.DataFrame([]),
    title=None,
    ylabel=None,
    xlabel=None,
):
    plt.style.use("ggplot")
    plt.rcParams.update({"font.family": "Helvetica"})
    fig, ax = plt.subplots(2, 1, sharex=True)
    folders = os.listdir(tb_folder)
    folders.sort()
    color_cycle = cycler(
        color=[
            "#E24A33",
            "#348ABD",
            "#988ED5",
            "#777777",
            "#FBC15E",
            "#8EBA42",
            "#FFB5B8",
            "#56B4E9",
            "#7FCE6C",
            "#B37CAB",
        ]
    )
    ax[0].set_ylim(33, 41)
    ax[0].set_ylabel("PSNR")
    ax[1].set_ylim(0.01, 0.13)
    ax[1].set_ylabel("Total validation loss")
    ax[1].set_xlabel("Training iteration")
    if title:
        ax.set_title(title)
    # else:
    #     fig.title(tb_folder[tb_folder.rfind("/")+1:])
    if df.empty:
        df = SummaryReader(tb_folder, pivot=True, extra_columns={"dir_name"}).scalars

    for folder in folders:
        if folder[0] == ".":
            continue
        if (folder + "/" + metric_folder + "/" + metric) not in {
            "dict_values([30.63578914390694, 1.85622829372303, 7.21417216452547, 3.664861906371173, 0.33632239425077304])/metrics/PSNR/val_PSNR",
            "dict_values([20.91300827774578, 0.5645893276116984, 0.3982808839783455, 1.7620882095663186, 0.23122622000767656])/metrics/PSNR/val_PSNR",
            "dict_values([26.44932054951696, 0.6426414839705488, 1.1846421662378028, 0.6195959029034585, 0.4462440454874421])/metrics/PSNR/val_PSNR",
            "dict_values([22.627515146672238, 3.0490220080601826, 11.238059327924026, 4.4547009536819475, 0.7566444181206711])/metrics/PSNR/val_PSNR",
            "dict_values([25.805853036899766, 0.5323644093171086, 4.55909757730143, 0.45456084359547966, 0.37842160521818535])/metrics/PSNR/val_PSNR",
            "dict_values([15.680657860827832, 0.3446202881342237, 2.6726819011613303, 0.42865107311793044, 0.3507716997518868])/metrics/PSNR/val_PSNR",
            "dict_values([11.762344114861072, 0.7807444987928767, 1.9376401130837047, 0.7233429775951351, 0.39397716234055485])/metrics/PSNR/val_PSNR",
        }:
            this_df1 = df[df["dir_name"] == folder + "/" + metric_folder + "/" + metric]
            this_df2 = df[
                df["dir_name"] == folder + "/" + metric_folder2 + "/" + metric2
            ]
            ax[0].plot(
                this_df1["step"],
                this_df1[metric_folder].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                ),
                color="#CCCCCC",
            )
            ax[1].plot(
                this_df2["step"],
                this_df2[metric_folder2].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                ),
                color="#CCCCCC",
            )

    ax[0].set_prop_cycle(color_cycle)
    ax[1].set_prop_cycle(copy.deepcopy(color_cycle))
    for folder in folders:
        if folder[0] == ".":
            continue
        if (folder + "/" + metric_folder + "/" + metric) in {
            "dict_values([30.63578914390694, 1.85622829372303, 7.21417216452547, 3.664861906371173, 0.33632239425077304])/metrics/PSNR/val_PSNR",
            "dict_values([20.91300827774578, 0.5645893276116984, 0.3982808839783455, 1.7620882095663186, 0.23122622000767656])/metrics/PSNR/val_PSNR",
            "dict_values([26.44932054951696, 0.6426414839705488, 1.1846421662378028, 0.6195959029034585, 0.4462440454874421])/metrics/PSNR/val_PSNR",
            "dict_values([22.627515146672238, 3.0490220080601826, 11.238059327924026, 4.4547009536819475, 0.7566444181206711])/metrics/PSNR/val_PSNR",
            "dict_values([25.805853036899766, 0.5323644093171086, 4.55909757730143, 0.45456084359547966, 0.37842160521818535])/metrics/PSNR/val_PSNR",
            "dict_values([15.680657860827832, 0.3446202881342237, 2.6726819011613303, 0.42865107311793044, 0.3507716997518868])/metrics/PSNR/val_PSNR",
            "dict_values([11.762344114861072, 0.7807444987928767, 1.9376401130837047, 0.7233429775951351, 0.39397716234055485])/metrics/PSNR/val_PSNR",
        }:
            this_df1 = df[df["dir_name"] == folder + "/" + metric_folder + "/" + metric]
            this_df2 = df[
                df["dir_name"] == folder + "/" + metric_folder2 + "/" + metric2
            ]
            ax[0].plot(
                this_df1["step"],
                this_df1[metric_folder].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                ),
            )
            ax[1].plot(
                this_df2["step"],
                this_df2[metric_folder2].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                ),
            )
    fig.savefig("./figures/exp25.pdf")


def create_exp1_plot():
    df = SummaryReader(
        "./tensorboard_log_cluster/Z_handling_no/seed1/",
        pivot=True,
        extra_columns={"dir_name"},
    ).scalars
    df["metrics/pix"] = (
        df["metrics/pix"]
        .apply(lambda x: x[0] if isinstance(x, list) else x)
        .apply(lambda x: UVW_MAX * x)
    )
    plt.style.use("ggplot")
    plt.rcParams.update({"font.family": "Helvetica"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey="row")
    axes[0, 0].set(ylim=(33, 41))
    axes[1, 0].set(ylim=(UVW_MAX * 0.007, UVW_MAX * 0.014))

    fig.suptitle("Experiment 1", fontweight="bold", fontsize=20)
    plot_metrics1(
        "./tensorboard_log_cluster/Z_handling_no/seed1/",
        "val_PSNR",
        "metrics/PSNR",
        axes[0, 0],
        df=df,
        ylabel="PSNR",
        title="Seed one",
    )
    plot_metrics1(
        "./tensorboard_log_cluster/Z_handling_no/seed1/",
        "pix_loss_unscaled",
        "metrics/pix",
        axes[1, 0],
        df=df,
        ylabel="average absolute error (m/s)",
        xlabel="Training iteration",
    )
    # fig.legend(handles, labels, loc='center right')

    df = SummaryReader(
        "./tensorboard_log_cluster/Z_handling_no/seed2/",
        pivot=True,
        extra_columns={"dir_name"},
    ).scalars
    df["metrics/pix"] = (
        df["metrics/pix"]
        .apply(lambda x: x[0] if isinstance(x, list) else x)
        .apply(lambda x: UVW_MAX * x)
    )
    plot_metrics1(
        "./tensorboard_log_cluster/Z_handling_no/seed2/",
        "val_PSNR",
        "metrics/PSNR",
        axes[0, 1],
        df=df,
        title="Seed two",
    )
    plot_metrics1(
        "./tensorboard_log_cluster/Z_handling_no/seed2/",
        "pix_loss_unscaled",
        "metrics/pix",
        axes[1, 1],
        df=df,
        xlabel="Training iteration",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles, labels, loc="lower center", ncol=4, fancybox=True, shadow=True)
    fig.savefig("./figures/Exp1.pdf", format="pdf", bbox_inches="tight")


def create_exp2_plot():
    df = SummaryReader(
        "./tensorboard_log_cluster/C100/", pivot=True, extra_columns={"dir_name"}
    ).scalars
    df["metrics/pix"] = df["metrics/pix"].apply(lambda x: UVW_MAX * x)
    plt.style.use("ggplot")
    plt.rcParams.update({"font.family": "Helvetica"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey="row")
    axes[0, 0].set(ylim=(33, 41))
    axes[1, 0].set(ylim=(UVW_MAX * 0.007, UVW_MAX * 0.014))

    fig.suptitle("Experiment 2", fontweight="bold", fontsize=20)
    plot_metrics2(
        "./tensorboard_log_cluster/C100/",
        "val_PSNR",
        "metrics/PSNR",
        axes[0, 0],
        df=df,
        ylabel="PSNR",
        title="Seed one",
    )
    plot_metrics2(
        "./tensorboard_log_cluster/C100/",
        "pix_loss_unscaled",
        "metrics/pix",
        axes[1, 0],
        df=df,
        ylabel="average absolute error (m/s)",
        xlabel="Training iteration",
    )
    # fig.legend(handles, labels, loc='center right')

    df = SummaryReader(
        "./tensorboard_log_cluster/C100_seed/", pivot=True, extra_columns={"dir_name"}
    ).scalars
    df["metrics/pix"] = df["metrics/pix"].apply(lambda x: UVW_MAX * x)
    plot_metrics2(
        "./tensorboard_log_cluster/C100_seed/",
        "val_PSNR",
        "metrics/PSNR",
        axes[0, 1],
        df=df,
        title="Seed two",
    )
    plot_metrics2(
        "./tensorboard_log_cluster/C100_seed/",
        "pix_loss_unscaled",
        "metrics/pix",
        axes[1, 1],
        df=df,
        xlabel="Training iteration",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles, labels, loc="lower center", ncol=4, fancybox=True, shadow=True)
    fig.savefig("./figures/Exp2.pdf", format="pdf", bbox_inches="tight")


def get_feature_maps(generator: Generator_3D, LR, Z):
    generator.eval()
    with torch.no_grad():
        LR_features = generator.model[:2](LR)
        LR_upscaled_features = generator.model(LR)
        HR_terrain_features = generator.terrain_convs(Z)
        pre_HR_conv = torch.cat((LR_upscaled_features, HR_terrain_features), dim=1)
        last_features_activation = generator.hr_convs[:-2](pre_HR_conv)
        last_features_before_activation = generator.hr_convs[:-3](pre_HR_conv)
        sum_last_features_after_activation = torch.sum(last_features_activation, 1)
        sum_last_features_before_activation = torch.sum(last_features_activation, 1)
        sum_LR_features = torch.sum(LR_features, 1)
        SR = generator(LR, Z)
    return (
        LR_features.squeeze().numpy(),
        LR_upscaled_features.squeeze().numpy(),
        HR_terrain_features.squeeze().numpy(),
        last_features_activation.squeeze().numpy(),
        last_features_before_activation.squeeze().numpy(),
        sum_last_features_after_activation.squeeze().numpy(),
        sum_last_features_before_activation.squeeze().numpy(),
        sum_LR_features.squeeze().numpy(),
        SR.squeeze().numpy(),
    )


def plot_scalar(
    X,
    Y,
    Z,
    scalar_field,
    z_plot_scale=1,
    terrain=np.asarray([]),
    fig=2,
    surface=True,
    z_step=5,
    colormap="jet",
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
                scalars=scalar_field[:, :, i].T,
            )

    else:
        mlab.points3d(X, Y, z_plot_scale * Z, scalar_field)
        if terrain.any():
            mlab.surf(
                X[:, :, 0].T,
                Y[:, :, 0].T,
                z_plot_scale * terrain.T,
                colormap="black-white",
            )

    mlab.show()


def generate_plots(X, Y, Z, u, v, w, terrain, colormap="viridis"):
    X1, Y1, Z1, u1, v1, w1, terrain1 = slice_only_dim_dicts(
        X,
        Y,
        Z,
        u,
        v,
        w,
        terrain,
        x_dict={"start": 0, "max": 128, "step": 1},
        y_dict={"start": 0, "max": 128, "step": 1},
        z_dict={"start": 0, "max": 41, "step": 5},
    )
    plot_field(
        X1, Y1, Z1, u1, v1, w1, terrain1, z_plot_scale=5, fig=1, colormap=colormap
    )
    X1, Y1, Z1, u1, v1, w1, terrain1 = slice_only_dim_dicts(
        X,
        Y,
        Z,
        u,
        v,
        w,
        terrain,
        x_dict={"start": 5, "max": 37, "step": 1},
        y_dict={"start": 10, "max": 42, "step": 1},
        z_dict={"start": 0, "max": 20, "step": 1},
    )
    plot_field(
        X1, Y1, Z1, u1, v1, w1, terrain1, z_plot_scale=1, fig=2, colormap=colormap
    )
    plot_field(
        X1[::4, ::4, :],
        Y1[::4, ::4, :],
        Z1[::4, ::4, :],
        u1[::4, ::4, :],
        v1[::4, ::4, :],
        w1[::4, ::4, :],
        terrain1,
        z_plot_scale=1,
        fig=3,
        colormap=colormap,
        terrainX=X1,
        terrainY=Y1,
    )


def generate_dist(dim, num_samples, dist="beta", alpha=0.35, beta=0.35):
    samples = np.zeros((dim, dim))
    for i in tqdm(range(num_samples)):
        if dist == "beta":
            x_start = round(np.random.beta(alpha, beta) * (dim / 2))
            y_start = round(np.random.beta(alpha, beta) * (dim / 2))
        elif dist == "uniform":
            x_start = np.random.randint(0, dim // 2 + 1)
            y_start = np.random.randint(0, dim // 2 + 1)
        samples[x_start : x_start + dim // 2, y_start : y_start + dim // 2] += 1
    return samples


def distribution_plots():
    x_2D, y_2D = np.meshgrid(x.numpy(), y.numpy())
    num_samples = 100000
    beta_dist = generate_dist(128, num_samples, dist="beta", alpha=0.25, beta=0.25)
    uniform_dist = generate_dist(128, num_samples, dist="uniform")
    mlab.figure(1)
    mlab.surf(
        beta_dist / num_samples,
        warp_scale="auto",
        color=(105 / 255, 165 / 255, 131 / 255),
    )
    mlab.show()
    mlab.figure(2)
    mlab.surf(
        uniform_dist / num_samples,
        warp_scale="auto",
        color=(236 / 255, 121 / 255, 154 / 255),
    )
    mlab.show()
    mlab.figure(3)
    mlab.surf(
        beta_dist / num_samples,
        warp_scale="auto",
        color=(105 / 255, 165 / 255, 131 / 255),
        opacity=0.5,
    )
    mlab.surf(
        uniform_dist / num_samples,
        warp_scale="auto",
        color=(236 / 255, 121 / 255, 154 / 255),
        opacity=0.5,
    )
    mlab.show()


def plot_feature_map_on_grid(feature_map, X, Y, Z, colormap="jet"):
    src = mlab.pipeline.scalar_scatter(X, Y, Z, feature_map)
    mlab.pipeline.glyph(
        src,
        mode="cube",
        scale_factor=150,
        opacity=0.02,
        scale_mode="none",
        colormap=colormap,
    )
    # field = mlab.pipeline.delaunay3d(src)


def plot_feature_field(feature_map, X, Y, Z, colormap="jet"):
    src = mlab.pipeline.scalar_scatter(X, Y, Z, feature_map)
    field = mlab.pipeline.delaunay3d(src)
    mlab.pipeline.volume(field)


def create_structured_grid(X, Y, Z):
    pts = np.zeros(X.shape + (3,), dtype=float)
    pts[..., 0] = X
    pts[..., 1] = Y
    pts[..., 2] = Z

    pts = pts.transpose(2, 1, 0, 3).reshape(pts.size // 3, 3)
    return tvtk.StructuredGrid(dimensions=X.shape, points=pts)


def plot_scalar_on_grid(sgrid, scalar, name="scalars", colormap="jet"):
    sgrid.point_data.scalars = scalar.T.ravel()
    sgrid.point_data.scalars.name = name
    mlab.pipeline.surface(sgrid, colormap=colormap, opacity=0.5)


def plot_vectors_on_grid(sgrid, vectors, name="vectors", colormap="jet"):
    sgrid.point_data.vectors = vectors.transpose(3, 2, 1, 0).reshape(
        vectors.size // 3, 3
    )
    sgrid.point_data.vectors.name = name
    mlab.pipeline.glyph(sgrid, colormap=colormap, opacity=0.5)


def plot_feature_map(feature_map, x=0, y=0, z=0, fig=1, vmin=None, vmax=None):
    mlab.figure(fig, bgcolor=(1, 1, 1))
    if x:
        field = mlab.pipeline.scalar_field(x, y, z, feature_map)
    else:
        field = mlab.pipeline.scalar_field(feature_map)
    if vmin and vmax:
        mlab.pipeline.volume(field, vmin=vmin, vmax=vmax)
    else:
        mlab.pipeline.volume(field)
    mlab.show()


def create_2D_plots(
    z1,
    z2,
    u_LR,
    u_HR,
    u_SR,
    u_TL,
    w_LR,
    w_HR,
    w_SR,
    w_TL,
    scale=4,
    x_dict={"start": 64, "max": 128, "step": 1},
    y_dict={"start": 64, "max": 128, "step": 1},
    z_dict={"start": 0, "max": 10, "step": 1},
):
    # plt.style.use("ggplot")
    plt.rcParams.update({"font.family": "Helvetica"})
    pix_criterion = nn.L1Loss()
    u_SR_loss_z1 = pix_criterion(
        torch.from_numpy(u_HR[:, :, z1]),
        torch.from_numpy(u_SR[:, :, z1]),
    ).item()
    u_TL_loss_z1 = pix_criterion(
        torch.from_numpy(u_HR[:, :, z1]),
        torch.from_numpy(u_TL[:, :, z1]),
    ).item()
    u_SR_loss_z2 = pix_criterion(
        torch.from_numpy(u_HR[:, :, z2]),
        torch.from_numpy(u_SR[:, :, z2]),
    ).item()
    u_TL_loss_z2 = pix_criterion(
        torch.from_numpy(u_HR[:, :, z2]),
        torch.from_numpy(u_TL[:, :, z2]),
    ).item()

    w_SR_loss_z1 = pix_criterion(
        torch.from_numpy(w_HR[:, :, z1]),
        torch.from_numpy(w_SR[:, :, z1]),
    ).item()
    w_TL_loss_z1 = pix_criterion(
        torch.from_numpy(w_HR[:, :, z1]),
        torch.from_numpy(w_TL[:, :, z1]),
    ).item()
    w_SR_loss_z2 = pix_criterion(
        torch.from_numpy(w_HR[:, :, z2]),
        torch.from_numpy(w_SR[:, :, z2]),
    ).item()
    w_TL_loss_z2 = pix_criterion(
        torch.from_numpy(w_HR[:, :, z2]),
        torch.from_numpy(w_TL[:, :, z2]),
    ).item()
    u_SR_loss_z1_relative = u_SR_loss_z1 / np.average(np.abs(u_HR[:, :, z1]))
    u_TL_loss_z1_relative = u_TL_loss_z1 / np.average(np.abs(u_HR[:, :, z1]))
    u_SR_loss_z2_relative = u_SR_loss_z2 / np.average(np.abs(u_HR[:, :, z2]))
    u_TL_loss_z2_relative = u_TL_loss_z2 / np.average(np.abs(u_HR[:, :, z2]))
    w_SR_loss_z1_relative = w_SR_loss_z1 / np.average(np.abs(w_HR[:, :, z1]))
    w_TL_loss_z1_relative = w_TL_loss_z1 / np.average(np.abs(w_HR[:, :, z1]))
    w_SR_loss_z2_relative = w_SR_loss_z2 / np.average(np.abs(w_HR[:, :, z2]))
    w_TL_loss_z2_relative = w_TL_loss_z2 / np.average(np.abs(w_HR[:, :, z2]))

    fig1 = create_comparison_figure(z1, u_LR, u_HR, u_SR, u_TL)
    fig2 = create_comparison_figure(z1, w_LR, w_HR, w_SR, w_TL)
    fig5 = create_error_figure(
        z1,
        u_HR,
        u_SR,
        u_TL,
        u_SR_loss_z1,
        u_TL_loss_z1,
        u_SR_loss_z1_relative,
        u_TL_loss_z1_relative,
    )
    fig6 = create_error_figure(
        z1,
        w_HR,
        w_SR,
        w_TL,
        w_SR_loss_z1,
        w_TL_loss_z1,
        w_SR_loss_z1_relative,
        w_TL_loss_z1_relative,
    )
    fig1.savefig(
        f"./figures/u_{z1}_scale{scale}.png", bbox_inches="tight", dpi=fig1.dpi
    )
    fig2.savefig(
        f"./figures/w_{z1}_scale{scale}.png", bbox_inches="tight", dpi=fig2.dpi
    )
    fig5.savefig(
        f"./figures/u_{z1}_scale{scale}_error.png", bbox_inches="tight", dpi=fig5.dpi
    )
    fig6.savefig(
        f"./figures/w_{z1}_scale{scale}_error.png", bbox_inches="tight", dpi=fig6.dpi
    )
    fig3 = create_comparison_figure(z2, u_LR, u_HR, u_SR, u_TL)
    fig4 = create_comparison_figure(z2, w_LR, w_HR, w_SR, w_TL)
    fig7 = create_error_figure(
        z2,
        u_HR,
        u_SR,
        u_TL,
        u_SR_loss_z2,
        u_TL_loss_z2,
        u_SR_loss_z2_relative,
        u_TL_loss_z2_relative,
    )
    fig8 = create_error_figure(
        z2,
        w_HR,
        w_SR,
        w_TL,
        w_SR_loss_z2,
        w_TL_loss_z2,
        w_SR_loss_z2_relative,
        w_TL_loss_z2_relative,
    )
    fig3.savefig(
        f"./figures/u_{z2}_scale{scale}.png", bbox_inches="tight", dpi=fig3.dpi
    )
    fig4.savefig(
        f"./figures/w_{z2}_scale{scale}.png", bbox_inches="tight", dpi=fig4.dpi
    )
    fig7.savefig(
        f"./figures/u_{z2}_scale{scale}_error.png", bbox_inches="tight", dpi=fig7.dpi
    )
    fig8.savefig(
        f"./figures/w_{z2}_scale{scale}_error.png", bbox_inches="tight", dpi=fig8.dpi
    )
    plt.show()


if __name__ == "__main__":
    
    folder = "./pretrained_models/upscale16_pix4_no_adv_no_slicing/"

    cfg = Config(folder + "config.ini")
    cfg.is_train = False
    cfg.is_download = False
    cfg.is_param_search = False
    cfg.is_test = True
    cfg.device = torch.device("cpu")
    cfg.env.generator_load_path = folder + "/G_150000.pth"

    gan = wind_field_GAN_3D(cfg)
    # (
    #     dataset_train,
    #     dataset_test,
    #     dataset_validation,
    #     x,
    #     y,
    # ) = preprosess(
    #     train_eval_test_ratio=0.8,
    #     X_DICT={"start": 0, "max": 128, "step": 1},
    #     Y_DICT={"start": 0, "max": 128, "step": 1},
    #     Z_DICT = {"start": 0, "max": cfg.gan_config.number_of_z_layers, "step": 1},
    #     start_date=date(2018, 3, 1),
    #     end_date=date(2018, 3, 2),
    #     include_pressure=True,
    #     include_z_channel=False,
    #     interpolate_z=False,
    #     enable_slicing=False,
    #     slice_size=64,
    #     include_above_ground_channel=False,
    #     COARSENESS_FACTOR=4,
    #     train_aug_rot=True,
    #     val_aug_rot=False,
    #     test_aug_rot=False,
    #     train_aug_flip=True,
    #     val_aug_flip=False,
    #     test_aug_flip=False,
    # )
    
    # LR, HR, Z, = dataset_train[1]

    filename = "test_fields_2020-03-01-14.pkl"

    full_filename = folder + "fields/" + filename
    fields = pkl.load(open(full_filename, "rb"))
    HR = fields["HR"]
    SR = fields["SR"]
    TL = fields["TL"]
    LR = fields["LR"]
    Z = fields["Z"]
    HR_orig = fields.get("HR_orig", None)
    SR_orig = fields.get("SR_orig", None)
    Z_orig = fields.get("Z_orig", None)
    try:
        with open("./data/full_dataset_files/static_terrain_x_y.pkl", "rb") as f:
            terrain, x, y = pkl.load(f)
    except:
        get_static_data()
        with open("./data/full_dataset_files/static_terrain_x_y.pkl", "rb") as f:
            terrain, x, y = pkl.load(f)

    amount_of_rotations = 1
    LR = torch.rot90(torch.from_numpy(LR), amount_of_rotations, [1, 2])
    HR = torch.rot90(torch.from_numpy(HR), amount_of_rotations, [1, 2])
    Z = torch.rot90(torch.from_numpy(Z)[None, :, :, :], amount_of_rotations, [1, 2])
    terrain = torch.rot90(torch.from_numpy(terrain), amount_of_rotations, [0, 1])
    if amount_of_rotations == 1:
        HR[:2] = torch.concatenate((-torch.index_select(HR, 0, torch.tensor(1)), torch.index_select(HR, 0, torch.tensor(0))), 0)
        LR[:2] = torch.concatenate((-torch.index_select(LR, 0, torch.tensor(1)), torch.index_select(LR, 0, torch.tensor(0))), 0)
    if amount_of_rotations == 2:
        HR[:2] = torch.concatenate((-torch.index_select(HR, 0, torch.tensor(0)), -torch.index_select(HR, 0, torch.tensor(1))), 0)
        LR[:2] = torch.concatenate((-torch.index_select(LR, 0, torch.tensor(0)), -torch.index_select(LR, 0, torch.tensor(1))), 0)
    if amount_of_rotations == 3:
        HR[:2] = torch.concatenate((torch.index_select(HR, 0, torch.tensor(1)), -torch.index_select(HR, 0, torch.tensor(0))), 0)
        LR[:2] = torch.concatenate((torch.index_select(LR, 0, torch.tensor(1)), -torch.index_select(LR, 0, torch.tensor(0))), 0)

    flip_index = 1
    LR = torch.flip(LR, [flip_index])
    HR = torch.flip(HR, [flip_index])
    Z = torch.flip(Z, [flip_index])
    terrain = torch.flip(terrain, [flip_index-1])
    if flip_index == 1:
        LR[0] = -LR[0]
        HR[0] = -HR[0]
    if flip_index == 2:
        LR[1] = -LR[1]
        HR[1] = -HR[1]

    LR, HR, Z = (
        LR.squeeze().numpy(),
        HR.squeeze().numpy(),
        Z.squeeze().numpy(),
    )

    _, _ = gan.load_model(
        generator_load_path=cfg.env.generator_load_path,
        discriminator_load_path=None,
        state_load_path=None,
    )
    (
        LR_features,
        LR_upscaled_features,
        HR_terrain_features,
        last_features_activation,
        last_features_before_activation,
        sum_last_features_after_activation,
        sum_last_features_before_activation,
        sum_LR_features,
        SR,
    ) = get_feature_maps(
        gan.G,
        torch.from_numpy(LR)[None, :, :, :],
        torch.from_numpy(Z)[None, None, :, :, :],
    )

    X_DICT = {"start": 0, "max": 128, "step": 1}
    Y_DICT = {"start": 0, "max": 128, "step": 1}

    TL = nn.functional.interpolate(
        torch.from_numpy(LR)[None, :3, :, :, :],
        scale_factor=(cfg.scale, cfg.scale, 1),
        align_corners=True,
        mode="trilinear",
    )

    # LR, HR, Z, TL, SR = LR.squeeze().numpy(), HR.squeeze().numpy(), Z.squeeze().numpy(), TL.squeeze().numpy(), SR.squeeze().numpy()
    TL = TL.squeeze().numpy()

    (
        u_norm,
        v_norm,
        w_norm,
    ) = (
        HR[0],
        HR[1],
        HR[2],
    )
    u_LR_norm, v_LR_norm, w_LR_norm = LR[0], LR[1], LR[2]
    u_SR_norm, v_SR_norm, w_SR_norm = SR[0], SR[1], SR[2]
    u_TL_norm, v_TL_norm, w_TL_norm = TL[0], TL[1], TL[2]

    u_HR, v_HR, w_HR, u_SR, v_SR, w_SR, u_TL, v_TL, w_TL, u_LR, v_LR, w_LR = (
        u_norm * UVW_MAX,
        v_norm * UVW_MAX,
        w_norm * UVW_MAX,
        # HR[3] * (dataset_train.P_MAX - dataset_train.P_MIN) + dataset_train.P_MIN,
        u_SR_norm * UVW_MAX,
        v_SR_norm * UVW_MAX,
        w_SR_norm * UVW_MAX,
        u_TL_norm * UVW_MAX,
        v_TL_norm * UVW_MAX,
        w_TL_norm * UVW_MAX,
        u_LR_norm * UVW_MAX,
        v_LR_norm * UVW_MAX,
        w_LR_norm * UVW_MAX,
    )
    SR_ERR_vec_length = np.sqrt(
        (u_HR - u_SR) ** 2 + (v_HR - v_SR) ** 2 + (w_HR - w_SR) ** 2
    )
    SR_ERR_vec_length_mean = np.average(SR_ERR_vec_length)
    TL_ERR_vec_length_mean = np.average(
        np.sqrt((u_HR - u_TL) ** 2 + (v_HR - v_TL) ** 2 + (w_HR - w_TL) ** 2)
    )
    HR_vec_length = np.sqrt(u_HR**2 + v_HR**2 + w_HR**2)

    print("SR_L1_loss: ", SR_ERR_vec_length_mean)
    print("TL_L1_loss: ", TL_ERR_vec_length_mean)
    print("SR_L1_loss_relative: ", SR_ERR_vec_length_mean / np.mean(HR_vec_length))
    print("TL_L1_loss_relative: ", TL_ERR_vec_length_mean / np.mean(HR_vec_length))

    X, Y, z_reg_max = np.mgrid[
        np.min(x) : np.max(x) : x.size * 1j,
        np.min(y) : np.max(y) : y.size * 1j,
        np.min(Z) : np.max(Z) : Z[0, 0].size * 1j,
    ]
    create_2D_plots(
        1,
        8,
        u_LR,
        u_HR,
        u_SR,
        u_TL,
        w_LR,
        w_HR,
        w_SR,
        w_TL,
        scale=cfg.scale,
        x_dict={"start": 64, "max": 128, "step": 1},
        y_dict={"start": 64, "max": 128, "step": 1},
        z_dict={"start": 0, "max": 10, "step": 1},
    )

    full_grid = create_structured_grid(X, Y, Z)
    LR_grid = create_structured_grid(X[::4, ::4, :], Y[::4, ::4, :], Z[::4, ::4, :])

    # x_dict={"start": 64, "max": 128, "step": 1}
    # y_dict={"start": 48, "max": 112, "step": 1}
    # z_dict={"start": 0, "max": 10, "step": 1}
    x_dict = {"start": 0, "max": 128, "step": 1}
    y_dict = {"start": 0, "max": 128, "step": 1}
    z_dict = {"start": 0, "max": 10, "step": 1}

    x2_dict = {
        "start": x_dict["start"] // 4,
        "max": x_dict["max"] // 4,
        "step": x_dict["step"] // 4 + 1,
    }
    y2_dict = {
        "start": y_dict["start"] // 4,
        "max": y_dict["max"] // 4,
        "step": y_dict["step"] // 4 + 1,
    }

    max_wind = np.max(HR_vec_length)
    print(
        "Average Wind speed: ", np.average(np.sqrt(u_HR**2 + v_HR**2 + w_HR**2))
    )

    max_error = np.max(SR_ERR_vec_length)
    # lut_manager_wind_field = mlab.colorbar(orientation='vertical', title="Wind speed (m/s)")
    # lut_manager_wind_field.data_range = (0, np.max(wind_vectors_length))

    (
        X1,
        Y1,
        Z1,
        u1,
        v1,
        w1,
        terrain1,
        u_SR1,
        v_SR1,
        w_SR1,
        u_TL1,
        v_TL1,
        w_TL1,
    ) = slice_only_dim_dicts(
        X,
        Y,
        Z,
        u_HR,
        v_HR,
        w_HR,
        terrain,
        u_SR,
        v_SR,
        w_SR,
        u_TL,
        v_TL,
        w_TL,
        x_dict=x_dict,
        y_dict=y_dict,
        z_dict=z_dict,
    )
    X_LR1, Y_LR1, Z_LR1, u_LR, v_LR, w_LR = slice_only_dim_dicts(
        X[::4, ::4, :],
        Y[::4, ::4, :],
        Z[::4, ::4, :],
        u_LR,
        v_LR,
        w_LR,
        x_dict=x2_dict,
        y_dict=y2_dict,
        z_dict=z_dict,
    )
    plot_field(
        X1,
        Y1,
        Z1,
        u1,
        v1,
        w1,
        terrain=terrain1,
        z_plot_scale=1,
        fig=1,
        colormap="viridis",
        max_value=max_wind,
        title="HR wind field",
    )
    plot_field(
        X_LR1,
        Y_LR1,
        Z_LR1,
        u_LR,
        v_LR,
        w_LR,
        terrain=terrain1,
        terrainX=X1[:, :, 0],
        terrainY=Y1[:, :, 0],
        z_plot_scale=1,
        fig=2,
        colormap="viridis",
        max_value=max_wind,
        title="LR wind field",
    )
    plot_field(
        X1,
        Y1,
        Z1,
        u_SR1,
        v_SR1,
        w_SR1,
        terrain=terrain1,
        z_plot_scale=1,
        fig=3,
        colormap="viridis",
        max_value=max_wind,
        title="SR wind field",
    )
    plot_field(
        X1,
        Y1,
        Z1,
        u_TL1,
        v_TL1,
        w_TL1,
        terrain=terrain1,
        z_plot_scale=1,
        fig=4,
        colormap="viridis",
        max_value=max_wind,
        title="TL wind field",
    )
    plot_field(
        X1,
        Y1,
        Z1,
        u1 - u_SR1,
        v1 - v_SR1,
        w1 - w_SR1,
        terrain=terrain1,
        z_plot_scale=1,
        fig=5,
        colormap="coolwarm",
        max_value=max_error,
        title="HR-SR wind field error",
    )
    plot_field(
        X1,
        Y1,
        Z1,
        u1 - u_TL1,
        v1 - v_TL1,
        w1 - w_TL1,
        terrain=terrain1,
        z_plot_scale=1,
        fig=6,
        colormap="coolwarm",
        max_value=max_error,
        title="HR-TL wind field error",
    )

    plot_feature_map(sum_last_features_after_activation, vmin=-3.05, vmax=1, fig=1)
    plot_feature_map(sum_last_features_before_activation, vmin=-3.05, vmax=1, fig=2)
    plot_feature_map(last_features_before_activation[2], fig=3)
    plot_feature_map(last_features_before_activation[3], fig=4)
    plot_feature_map(last_features_before_activation[4], fig=5)
    plot_feature_map(last_features_before_activation[5], fig=6)
    plot_feature_map(last_features_activation[2], fig=7)
    plot_feature_map(last_features_activation[3], fig=8)
    plot_feature_map(last_features_activation[4], fig=9)
    plot_feature_map(last_features_activation[5], fig=10)
