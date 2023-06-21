from mayavi import mlab
from mayavi.mlab import pipeline as pipe
from tvtk.api import tvtk
import numpy as np
import torch.nn as nn
import pickle as pkl
from process_data import preprosess
from datetime import date
from download_data import slice_only_dim_dicts, reverse_interpolate_z_axis
from GAN_models.wind_field_GAN_3D import compute_PSNR_for_SR_and_trilinear
from tqdm import tqdm
import matplotlib.pyplot as plt
from train import create_comparison_figure, create_error_figure
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

Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MIN,P_MAX = [-2.7072315216064453, 550.43798828125, 68.45956420898438, 32.32976150512695, 0.0, 105182.828125]


name_dict = {
   'Z_handling90only_wind': r'only_wind',
    'Z_handling90wind_rawZ': r'$z$_channel',
    'Z_handling90wind_pressure': r'$p$_channel',
    'SCH100_schedule2_larger_grad': r'$p\_z$_channels',
    'Z_handling90wind_rawZ_pressure': r'$p\_z$_channels',
    'Z_handling90wind_Zground': r'$z_{ground}$_channels',
    'Z_handling90wind_Zground_pressure': r'$p\_z_{ground}$_channels',
    'Z90_interponly_wind': r'only_wind_interp',
    'Z90_interpwind_interpZ': r'$z$_channel_interp',
    'Z90_interpwind_pressure': r'$p$_channel_interp',
    'Z90_interpwind_interpZ_pressure': r'$z\_p$_channels_interp',
    'Z_handling90_seedonly_wind': r'only_wind',
    'Z_handling90_seedwind_rawZ': r'$z$_channel',
    'Z_handling90_seedwind_pressure': r'$p$_channel',
    'Z_handling90_seed_wind_rawZ_pressure': r'$p\_z$_channels',
    'Z_handling90_seedwind_rawZ_pressure': r'$p\_z$_channels',
    'Z100_seed_wind_Zground': r'$z_{ground}$_channels',
    'Z100_seed_wind_Zground_pressure': r'$p\_z_{ground}$_channels',
    'Z_handling90_seedwind_Zground': r'$z_{ground}$_channels',
    'Z_handling90_seedwind_Zground_pressure': r'$p\_z_{ground}$_channels',
    'Z90_interp_seedonly_wind': r'only_wind_interp',
    'Z90_interp_seedwind_interpZ': r'$z$_channel_interp',
    'Z90_interp_seedwind_pressure': r'$p$_channel_interp',
    'Z90_interp_seedwind_interpZ_pressure': r'$z\_p$_channels_interp',
    'C100_only_pix': r'only_pix_cost',
    'C100_grad': r'grad_cost',
    'C100_div': r'div_cost',
    'C100_xy': r'$xy$_cost',
    'C100_grad_large': r'large_grad_cost',
    'C100_div_large': r'large_div_cost',
    'C100_xy_large': r'large_$xy$_cost',
    'C100_seed_only_pix': r'only_pix_cost',
    'C100_seed_grad': r'grad_cost',
    'C100_seed_div': r'div_cost',
    'C100_seed_xy': r'$xy$_cost',
    'C100_seed_grad_large': r'large_grad_cost',
    'C100_seed_div_large': r'large_div_cost',
    'C100_seed_xy_large': r'large_$xy$_cost',
    "STD": "STD",
    "STD_seed": "STD",
}

def plot_field(X, Y, Z, u, v, w, terrain=np.asarray([]), z_plot_scale=1, fig=1, colormap="viridis", terrainX=np.asarray([]), terrainY=np.asarray([])):
    mlab.figure(fig)
    mlab.quiver3d(
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
                    terrainX[:, :, 0].T,
                    terrainY[:, :, 0].T,
                    z_plot_scale * terrain.T,
                    colormap="black-white",
                )
            else:
                mlab.surf(
                    X[:, :, 0].T,
                    Y[:, :, 0].T,
                    z_plot_scale * terrain.T,
                    colormap="black-white",
                )
        except:
            mlab.surf(X.T, Y.T, z_plot_scale * terrain.T, colormap="black-white")
    mlab.show()

def plot_metrics1(tb_folder, metric, metric_folder, ax:plt.Axes, df = pd.DataFrame([]), title=None, ylabel=None, xlabel=None):
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
        df = SummaryReader(tb_folder, pivot=True, extra_columns={'dir_name'}).scalars
        
    color_cycle = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8','#56B4E9', '#7FCE6C', '#B37CAB'])
    ax.set_prop_cycle(color_cycle)

    for folder in folders:
        if folder[0] == ".":
            continue
        this_df = df[df["dir_name"].str.contains(folder+"/") & df["dir_name"].str.contains(metric)]
        ax.plot(this_df[this_df["step"]<90001]["step"], this_df[this_df["step"]<90001][metric_folder].apply(lambda x: x[0] if isinstance(x,list) else x), label=name_dict[folder])
    
    if metric == "val_PSNR":
        trilinear_metric = "Trilinear_PSNR"
    elif metric == "pix_loss_unscaled":
        trilinear_metric = "trilinear_pix_loss"

    this_df = df[df["dir_name"].str.contains(folder+"/") & df["dir_name"].str.contains(trilinear_metric)]
    ax.plot(this_df[this_df["step"]<90001]["step"], this_df[this_df["step"]<90001][metric_folder].apply(lambda x: x[0] if isinstance(x,list) else x), label="Trilinear interpolation", color='#CCCCCC', linestyle="--")

def plot_metrics2(tb_folder, metric, metric_folder, ax:plt.Axes, df = pd.DataFrame([]), title=None, ylabel=None, xlabel=None):
    folders = os.listdir(tb_folder)
    folders.sort()
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if df.empty:
        df = SummaryReader(tb_folder, pivot=True, extra_columns={'dir_name'}).scalars
        
    color_cycle = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8','#56B4E9', '#7FCE6C', '#B37CAB'])
    ax.set_prop_cycle(color_cycle)

    for folder in folders:
        if folder[0] == ".":
            continue
        this_df = df[df["dir_name"].str.contains(folder+"/") & df["dir_name"].str.contains(metric)]
        ax.plot(this_df["step"], this_df[metric_folder].apply(lambda x: x[0] if isinstance(x,list) else x), label=name_dict[folder])
    
    if metric == "val_PSNR":
        trilinear_metric = "Trilinear_PSNR"
    elif metric == "pix_loss_unscaled":
        trilinear_metric = "trilinear_pix_loss"

    this_df = df[df["dir_name"].str.contains(folder+"/") & df["dir_name"].str.contains(trilinear_metric)]
    ax.plot(this_df[this_df["step"]<90001]["step"], this_df[this_df["step"]<90001][metric_folder].apply(lambda x: x[0] if isinstance(x,list) else x), label="Trilinear interpolation", color='#CCCCCC', linestyle="--")



def plot_metrics25(tb_folder, metric, metric2, metric_folder, metric_folder2, df = pd.DataFrame([]), title=None, ylabel=None, xlabel=None):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(2,1, sharex=True)
    folders = os.listdir(tb_folder)
    folders.sort()
    color_cycle = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8','#56B4E9', '#7FCE6C', '#B37CAB'])
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
        df = SummaryReader(tb_folder, pivot=True, extra_columns={'dir_name'}).scalars
        
    for folder in folders:
        if folder[0] == ".":
            continue
        if (folder+"/"+metric_folder+"/"+metric) not in {'dict_values([30.63578914390694, 1.85622829372303, 7.21417216452547, 3.664861906371173, 0.33632239425077304])/metrics/PSNR/val_PSNR','dict_values([20.91300827774578, 0.5645893276116984, 0.3982808839783455, 1.7620882095663186, 0.23122622000767656])/metrics/PSNR/val_PSNR','dict_values([26.44932054951696, 0.6426414839705488, 1.1846421662378028, 0.6195959029034585, 0.4462440454874421])/metrics/PSNR/val_PSNR','dict_values([22.627515146672238, 3.0490220080601826, 11.238059327924026, 4.4547009536819475, 0.7566444181206711])/metrics/PSNR/val_PSNR','dict_values([25.805853036899766, 0.5323644093171086, 4.55909757730143, 0.45456084359547966, 0.37842160521818535])/metrics/PSNR/val_PSNR','dict_values([15.680657860827832, 0.3446202881342237, 2.6726819011613303, 0.42865107311793044, 0.3507716997518868])/metrics/PSNR/val_PSNR','dict_values([11.762344114861072, 0.7807444987928767, 1.9376401130837047, 0.7233429775951351, 0.39397716234055485])/metrics/PSNR/val_PSNR'}:
            this_df1 = df[df["dir_name"]==folder+"/"+metric_folder+"/"+metric]
            this_df2 = df[df["dir_name"]==folder+"/"+metric_folder2+"/"+metric2]
            ax[0].plot(this_df1["step"], this_df1[metric_folder].apply(lambda x: x[0] if isinstance(x,list) else x), color='#CCCCCC')
            ax[1].plot(this_df2["step"], this_df2[metric_folder2].apply(lambda x: x[0] if isinstance(x,list) else x), color='#CCCCCC')
    
    ax[0].set_prop_cycle(color_cycle)
    ax[1].set_prop_cycle(copy.deepcopy(color_cycle))
    for folder in folders:
        if folder[0] == ".":
            continue    
        if (folder+"/"+metric_folder+"/"+metric) in {'dict_values([30.63578914390694, 1.85622829372303, 7.21417216452547, 3.664861906371173, 0.33632239425077304])/metrics/PSNR/val_PSNR','dict_values([20.91300827774578, 0.5645893276116984, 0.3982808839783455, 1.7620882095663186, 0.23122622000767656])/metrics/PSNR/val_PSNR','dict_values([26.44932054951696, 0.6426414839705488, 1.1846421662378028, 0.6195959029034585, 0.4462440454874421])/metrics/PSNR/val_PSNR','dict_values([22.627515146672238, 3.0490220080601826, 11.238059327924026, 4.4547009536819475, 0.7566444181206711])/metrics/PSNR/val_PSNR','dict_values([25.805853036899766, 0.5323644093171086, 4.55909757730143, 0.45456084359547966, 0.37842160521818535])/metrics/PSNR/val_PSNR','dict_values([15.680657860827832, 0.3446202881342237, 2.6726819011613303, 0.42865107311793044, 0.3507716997518868])/metrics/PSNR/val_PSNR','dict_values([11.762344114861072, 0.7807444987928767, 1.9376401130837047, 0.7233429775951351, 0.39397716234055485])/metrics/PSNR/val_PSNR'}:
            this_df1 = df[df["dir_name"]==folder+"/"+metric_folder+"/"+metric]
            this_df2 = df[df["dir_name"]==folder+"/"+metric_folder2+"/"+metric2]
            ax[0].plot(this_df1["step"], this_df1[metric_folder].apply(lambda x: x[0] if isinstance(x,list) else x))
            ax[1].plot(this_df2["step"], this_df2[metric_folder2].apply(lambda x: x[0] if isinstance(x,list) else x))
    fig.savefig("./figures/exp25.pdf")

def create_exp1_plot():
    df = SummaryReader("./tensorboard_log_cluster/Z_handling_no/seed1/", pivot=True, extra_columns={'dir_name'}).scalars
    df["metrics/pix"] = df["metrics/pix"].apply(lambda x: x[0] if isinstance(x,list) else x).apply(lambda x: UVW_MAX*x)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2,2, figsize=(10, 8), sharex=True, sharey="row")
    axes[0,0].set(ylim=(33, 41))
    axes[1,0].set(ylim=(UVW_MAX*0.007, UVW_MAX*0.014))
    
    fig.suptitle("Experiment 1", fontweight="bold", fontsize=20)
    plot_metrics1("./tensorboard_log_cluster/Z_handling_no/seed1/", "val_PSNR", "metrics/PSNR", axes[0,0], df=df, ylabel="PSNR", title="Seed one")
    plot_metrics1("./tensorboard_log_cluster/Z_handling_no/seed1/", "pix_loss_unscaled", "metrics/pix", axes[1,0], df=df, ylabel="average absolute error (m/s)", xlabel="Training iteration")
    # fig.legend(handles, labels, loc='center right')

    df = SummaryReader("./tensorboard_log_cluster/Z_handling_no/seed2/", pivot=True, extra_columns={'dir_name'}).scalars
    df["metrics/pix"] = df["metrics/pix"].apply(lambda x: x[0] if isinstance(x,list) else x).apply(lambda x: UVW_MAX*x)
    plot_metrics1("./tensorboard_log_cluster/Z_handling_no/seed2/", "val_PSNR", "metrics/PSNR", axes[0,1], df=df, title="Seed two")
    plot_metrics1("./tensorboard_log_cluster/Z_handling_no/seed2/", "pix_loss_unscaled", "metrics/pix", axes[1,1], df=df, xlabel="Training iteration")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles, labels, loc='lower center', ncol=4, fancybox=True, shadow=True)
    fig.savefig("./figures/Exp1.pdf", format="pdf", bbox_inches="tight")

def create_exp2_plot():
    df = SummaryReader("./tensorboard_log_cluster/C100/", pivot=True, extra_columns={'dir_name'}).scalars
    df["metrics/pix"] = df["metrics/pix"].apply(lambda x: UVW_MAX*x)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2,2, figsize=(10, 8), sharex=True, sharey="row")
    axes[0,0].set(ylim=(33, 41))
    axes[1,0].set(ylim=(UVW_MAX*0.007, UVW_MAX*0.014))
    
    fig.suptitle("Experiment 2", fontweight="bold", fontsize=20)
    plot_metrics2("./tensorboard_log_cluster/C100/", "val_PSNR", "metrics/PSNR", axes[0,0], df=df, ylabel="PSNR", title="Seed one")
    plot_metrics2("./tensorboard_log_cluster/C100/", "pix_loss_unscaled", "metrics/pix", axes[1,0], df=df, ylabel="average absolute error (m/s)", xlabel="Training iteration")
    # fig.legend(handles, labels, loc='center right')

    df = SummaryReader("./tensorboard_log_cluster/C100_seed/", pivot=True, extra_columns={'dir_name'}).scalars
    df["metrics/pix"] = df["metrics/pix"].apply(lambda x: UVW_MAX*x)
    plot_metrics2("./tensorboard_log_cluster/C100_seed/", "val_PSNR", "metrics/PSNR", axes[0,1], df=df, title="Seed two")
    plot_metrics2("./tensorboard_log_cluster/C100_seed/", "pix_loss_unscaled", "metrics/pix", axes[1,1], df=df, xlabel="Training iteration")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles, labels, loc='lower center', ncol=4, fancybox=True, shadow=True)
    fig.savefig("./figures/Exp2.pdf", format="pdf", bbox_inches="tight")


def get_feature_maps(generator: Generator_3D, LR, Z):
    generator.eval()
    with torch.no_grad():
        LR_features = generator.model[:2](LR)
        LR_upscaled_features = generator.model(LR)
        HR_terrain_features = generator.terrain_conv(Z)
        pre_HR_conv = torch.cat((LR_upscaled_features, HR_terrain_features), dim=1)
        last_features_activation = generator.hr_convs[:-2](pre_HR_conv)
        last_features_before_activation = generator.hr_convs[:-3](pre_HR_conv)
        sum_last_features_after_activation = torch.sum(last_features_activation,1)
        sum_last_features_before_activation = torch.sum(last_features_activation,1)
        sum_LR_features = torch.sum(LR_features,1)
    return LR_features.squeeze().numpy(), LR_upscaled_features.squeeze().numpy(), HR_terrain_features.squeeze().numpy(), last_features_activation.squeeze().numpy(), last_features_before_activation.squeeze().numpy(), sum_last_features_after_activation.squeeze().numpy(), sum_last_features_before_activation.squeeze().numpy(), sum_LR_features.squeeze().numpy()




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

    X1, Y1, Z1, u1, v1, w1, terrain1 = slice_only_dim_dicts(X, Y, Z, u, v, w, terrain, x_dict={"start": 0, "max": 128, "step": 1}, y_dict={"start": 0, "max": 128, "step": 1},z_dict={"start": 0, "max": 41, "step": 5},)
    plot_field(X1,Y1,Z1,u1,v1,w1, terrain1, z_plot_scale=5, fig=1, colormap=colormap)
    X1, Y1, Z1, u1, v1, w1, terrain1  = slice_only_dim_dicts(X, Y, Z, u, v, w, terrain, x_dict={"start": 5, "max": 37, "step": 1}, y_dict={"start": 10, "max": 42, "step": 1},z_dict={"start": 0, "max": 11, "step": 1},)
    plot_field(X1,Y1,Z1,u1,v1,w1, terrain1, z_plot_scale=1, fig=2, colormap=colormap)
    plot_field(X1[::4,::4,:],Y1[::4,::4,:],Z1[::4,::4,:],u1[::4,::4,:],v1[::4,::4,:],w1[::4,::4,:], terrain1, z_plot_scale=1, fig=3, colormap=colormap, terrainX=X1, terrainY=Y1)

def generate_dist(dim, num_samples, dist="beta", alpha=0.35, beta=0.35):
    samples = np.zeros((dim, dim))
    for i in tqdm(range(num_samples)):
        if dist == "beta":
            x_start = round(np.random.beta(alpha, beta) * (dim/2))
            y_start = round(np.random.beta(alpha, beta) * (dim/2))
        elif dist == "uniform":
            x_start = np.random.randint(0, dim//2+1)
            y_start = np.random.randint(0, dim//2+1)
        samples[x_start:x_start + dim//2, y_start:y_start + dim//2] += 1
    return samples

def distribution_plots():
    x_2D, y_2D = np.meshgrid(x.numpy(), y.numpy())
    num_samples = 100000
    beta_dist = generate_dist(128, num_samples, dist="beta", alpha=0.25, beta=0.25)
    uniform_dist = generate_dist(128, num_samples, dist="uniform")
    mlab.figure(1)
    mlab.surf(beta_dist/num_samples, warp_scale="auto", color=(105/255,165/255,131/255))
    mlab.show()
    mlab.figure(2)
    mlab.surf(uniform_dist/num_samples, warp_scale="auto", color=(236/255,121/255,154/255))
    mlab.show()
    mlab.figure(3)
    mlab.surf(beta_dist/num_samples, warp_scale="auto", color=(105/255,165/255,131/255), opacity=0.5)
    mlab.surf(uniform_dist/num_samples, warp_scale="auto", color=(236/255,121/255,154/255), opacity=0.5)
    mlab.show()

def plot_feature_map_on_grid(feature_map, X, Y, Z, colormap="jet"):
    src = mlab.pipeline.scalar_scatter(X, Y, Z, feature_map)
    mlab.pipeline.glyph(src, mode="cube", scale_factor=150, opacity=0.02, scale_mode="none", colormap=colormap)
    # field = mlab.pipeline.delaunay3d(src)

def plot_feature_field(feature_map, X, Y, Z, colormap="jet"):
    src = mlab.pipeline.scalar_scatter(X, Y, Z, feature_map)
    field = mlab.pipeline.delaunay3d(src)
    mlab.pipeline.volume(field)

def create_structured_grid(X,Y,Z):
    pts = np.zeros(X.shape + (3,), dtype=float)
    pts[..., 0] = X
    pts[..., 1] = Y
    pts[..., 2] = Z

    pts = pts.transpose(2, 1, 0, 3).reshape(pts.size // 3, 3)
    return tvtk.StructuredGrid(dimensions=X.shape, points=pts)

def plot_scalar_on_grid(sgrid,scalar, name="scalars", colormap="jet"):
    sgrid.point_data.scalars = scalar.T.ravel()
    sgrid.point_data.scalars.name = name
    mlab.pipeline.surface(sgrid, colormap=colormap, opacity=0.5)

def plot_vectors_on_grid(sgrid, vectors, name="vectors", colormap="jet"):
    sgrid.point_data.vectors = vectors.transpose(3, 2, 1, 0).reshape(vectors.size // 3, 3)
    sgrid.point_data.vectors.name = name
    mlab.pipeline.glyph(sgrid, colormap=colormap, opacity=0.5)



if __name__ == "__main__":

    
    # plot_metrics("./tensorboard_log_cluster/Z_handling_no/seed1", "val_PSNR", "metrics/PSNR")
    # create_exp1_plot()
    plot_metrics25("/Volumes/jawold/GAN_SR_wind_field/tensorboard_log/35kGAN_cost_param_search/", "val_PSNR", "total", "metrics/PSNR", "G_loss/validation")
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
    #     Z_DICT={"start": 0, "max": 10, "step": 1},
    #     start_date=date(2018, 10, 1),
    #     end_date=date(2018, 10, 2),
    #     include_pressure=True,
    #     include_z_channel=False,
    #     interpolate_z=False,
    #     enable_slicing=True,
    #     slice_size=64,
    #     include_above_ground_channel=False,
    #     COARSENESS_FACTOR=4,
    #     train_aug_rot=False,
    #     val_aug_rot=False,
    #     test_aug_rot=False,
    #     train_aug_flip=False,
    #     val_aug_flip=False,
    #     test_aug_flip=False,
    #     for_plotting=True,
    # )

    # dataloader_normal = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=False)

    # LR, HR, Z, filename, terrain, x, y = dataset_test[0]
    # LR, HR, Z = LR.squeeze().numpy(), HR.squeeze().numpy(), Z.squeeze().numpy()


    folder = "./runs/C100_xy_large/fields/"
    filename = "test_fields_2018-03-15-21_0.pkl"
    full_filename = folder + filename
    fields = pkl.load(open(full_filename, "rb")) 
    HR = fields["HR"]
    SR = fields["SR"] 
    TL = fields["TL"] 
    LR = fields["LR"]
    Z = fields["Z"]
    terrain = fields["terrain"]
    x = fields["x"].numpy()
    y = fields["y"].numpy()

    cfg = Config("./runs/C100_xy_large/config.ini")
    cfg.is_train = False
    cfg.is_download = False
    cfg.is_param_search = False
    cfg.is_test = True
    cfg.device = torch.device("cpu")

    gan = wind_field_GAN_3D(cfg)
    _, _ = gan.load_model(
        generator_load_path=cfg.env.generator_load_path,
        discriminator_load_path=None,
        state_load_path=None,
    )
    LR_features, LR_upscaled_features, HR_terrain_features, last_features_activation, last_features_before_activation, sum_last_features_after_activation, sum_last_features_before_activation, sum_LR_features = get_feature_maps(gan.G, torch.from_numpy(LR)[None, :,:,:,:], torch.from_numpy(Z)[None, None, :,:,:])
    
    X_DICT={"start": 0, "max": 128, "step": 1}
    Y_DICT={"start": 0, "max": 128, "step": 1}

    
    # u_norm, v_norm, w_norm, pressure_norm = HR[0], HR[1], HR[2], HR[3]
    # u, v, w, pressure = u_norm*dataset_train.UVW_MAX, v_norm*dataset_train.UVW_MAX, w_norm*dataset_train.UVW_MAX, HR[3]*(dataset_train.P_MAX-dataset_train.P_MIN)+dataset_train.P_MIN
    # X,Y, _ = np.meshgrid(x,y, Z[0,0,:])
    X,Y, _ = np.mgrid[np.min(x):np.max(x):x.size*1j, np.min(y):np.max(y):y.size*1j, np.min(Z[0,0]):np.max(Z[0,0]):Z[0,0].size*1j]
    
    full_grid = create_structured_grid(X,Y,Z) 
    LR_grid = create_structured_grid(X[::4,::4,:],Y[::4,::4,:],Z[::4,::4,:])

    # plot_vectors_on_grid(full_grid, HR, name="HR", colormap="Blues")

    
    plot_scalar_on_grid(LR_grid, LR_features[2], name="LR_feature")
    plot_field(X[::4,::4,:],Y[::4,::4,:],Z[::4,::4,:],LR[0],LR[1],LR[2],colormap="viridis")
    plot_feature_map_on_grid(sum_last_features_after_activation, X, Y, Z)
    plot_field(X, Y, Z, HR[0], HR[1], HR[2], colormap="Blues", fig=1)
    plot_field(X, Y, Z, SR[0], SR[1], SR[2], colormap="viridis", fig=2)
    
    plot_feature_map_on_grid(last_features_before_activation[1], X, Y, Z)
    plot_field(X, Y, Z, HR[0]-SR[0], HR[1]-SR[1], HR[2]-SR[2], colormap="coolwarm", fig=3)
    plot_field(X, Y, Z, HR[0]-TL[0], HR[1]-TL[1], HR[2]-SR[2], colormap="coolwarm", fig=4)




    # generate_plots(X, Y, Z, u, v, w, pressure, dataset_train.terrain, colormap="viridis")
    # X1, Y1, Z1, u1, v1, w1, pressure1, terrain1  = slice_only_dim_dicts(X, Y, Z, u, v, w, pressure, dataset_train.terrain, x_dict={"start": 5, "max": 37, "step": 1}, y_dict={"start": 10, "max": 42, "step": 1},z_dict={"start": 0, "max": 11, "step": 1},)
    # # X1, Y1, Z1, u1, v1, w1, pressure1, terrain1 = slice_only_dim_dicts(X, Y, Z, u, v, w, pressure, dataset_train.terrain, x_dict={"start": 0, "max": 128, "step": 1}, y_dict={"start": 0, "max": 128, "step": 1},z_dict={"start": 0, "max": 10, "step": 10},)
    # # plot_field(X1,Y1,Z1,u1,v1,w1, terrain1, z_plot_scale=2, fig=1)
    # # X1, Y1, Z1, u1, v1, w1, pressure1, terrain1 = slice_only_dim_dicts(X, Y, Z, u, v, w, pressure, dataset_train.terrain, x_dict={"start": 0, "max": 64, "step": 1}, y_dict={"start": 0, "max": 128, "step": 1},z_dict={"start": 0, "max": 10, "step": 10},)
    # # plot_field(X1,Y1,Z1,u1,v1,w1, terrain1, z_plot_scale=2, fig=2)
    # # X1, Y1, Z1, u1, v1, w1, pressure1, terrain1 = slice_only_dim_dicts(X, Y, Z, u, v, w, pressure, dataset_train.terrain, x_dict={"start": 64, "max": 128, "step": 1}, y_dict={"start": 0, "max": 128, "step": 1},z_dict={"start": 0, "max": 10, "step": 10},)
    # plot_field(X1,Y1,Z1,u1,v1,w1, terrain1, z_plot_scale=1, fig=1)
    