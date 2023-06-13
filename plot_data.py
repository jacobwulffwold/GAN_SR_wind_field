from mayavi import mlab
import numpy as np
import torch.nn as nn
import torch
from process_data import preprosess
from datetime import date
from download_data import reverse_interpolate_z_axis
from GAN_models.wind_field_GAN_3D import compute_psnr_for_SR_and_trilinear

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


if __name__ == "__main__":
    (
        dataset_train,
        dataset_test,
        dataset_validation,
        x,
        y,
    ) = preprosess(
        train_eval_test_ratio=0.8,
        X_DICT={"start": 0, "max": 128, "step": 1},
        Y_DICT={"start": 0, "max": 128, "step": 1},
        Z_DICT={"start": 0, "max": 10, "step": 1},
        start_date=date(2018, 10, 1),
        end_date=date(2018, 10, 10),
        include_pressure=True,
        include_z_channel=False,
        interpolate_z=False,
        enable_slicing=False,
        slice_size=64,
        include_above_ground_channel=False,
        COARSENESS_FACTOR=4,
        train_aug_rot=False,
        val_aug_rot=False,
        test_aug_rot=False,
        train_aug_flip=False,
        val_aug_flip=False,
        test_aug_flip=False,
    )
    (
        dataset_train_interp,
        dataset_test,
        dataset_validation,
        x,
        y,
    ) = preprosess(
        train_eval_test_ratio=0.8,
        X_DICT={"start": 0, "max": 128, "step": 1},
        Y_DICT={"start": 0, "max": 128, "step": 1},
        Z_DICT={"start": 0, "max": 10, "step": 1},
        start_date=date(2018, 10, 1),
        end_date=date(2018, 10, 10),
        include_pressure=True,
        include_z_channel=False,
        interpolate_z=True,
        enable_slicing=False,
        slice_size=64,
        include_above_ground_channel=False,
        COARSENESS_FACTOR=4,
        train_aug_rot=False,
        val_aug_rot=False,
        test_aug_rot=False,
        train_aug_flip=False,
        val_aug_flip=False,
        test_aug_flip=False,
    )

    dataloader_normal = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=False)
    dataloader_interp = torch.utils.data.DataLoader(dataset_train_interp, batch_size=32, shuffle=False)

    total_STD_PSNR=0.0
    total_reversed_interpolated_PSNR=0.0
    total_STD_PSNR_interp=0.0

    interp_iter = iter(dataloader_interp)

    for i, (LR, HR, Z) in enumerate(dataloader_normal):
        LR_interp, HR_interp, Z_interp = next(interp_iter)
        reversed_interpolated_HR = reverse_interpolate_z_axis(
            nn.functional.interpolate(
                LR_interp[:, :3, :, :, :],
                scale_factor=(4, 4, 1),
                mode="trilinear",
                align_corners=True,
            ),
            Z,
            Z_interp,
        )
        reversed_interpolated_PSNR, STD_PSNR = compute_psnr_for_SR_and_trilinear(LR, HR, reversed_interpolated_HR, 4, 1e-8, interpolate=True)
        _, STD_PSNR_interp = compute_psnr_for_SR_and_trilinear(LR_interp, HR_interp, reversed_interpolated_HR, 4, 1e-8, interpolate=True)

        total_reversed_interpolated_PSNR += reversed_interpolated_PSNR
        total_STD_PSNR += STD_PSNR
        total_STD_PSNR_interp += STD_PSNR_interp
    
    total_reversed_interpolated_PSNR /= i
    total_STD_PSNR /= i
    total_STD_PSNR_interp /= i

    print(STD_PSNR, STD_PSNR_interp, reversed_interpolated_PSNR)
    print(total_reversed_interpolated_PSNR, total_STD_PSNR, total_STD_PSNR_interp)
    # LR, HR, Z = LR.squeeze().numpy(), HR.squeeze().numpy(), Z.squeeze().numpy()
    # u_norm, v_norm, w_norm, pressure_norm = HR[0], HR[1], HR[2], HR[3]
    # u, v, w, pressure = HR[0]/dataset_train.UVW_MAX, HR[1]/dataset_train.UVW_MAX, HR[2]/dataset_train.UVW_MAX, HR[3]/dataset_train.P_MAX

    # X,Y, _ = np.meshgrid(x.numpy(), y.numpy(), Z[0,0,:])
    # plot_field(X,Y,Z,u,v,w,dataset_train.terrain, z_plot_scale=5)
    