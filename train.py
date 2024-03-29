"""
train.py
Originally written by Eirik Vesterkjær 2019, edited by Jacob Wulff Wold 2023
Apache License

Implements a GAN training loop
Use run.py to run.
"""
import logging
import os
import pickle as pkl
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import tensorboardX
import config.config as config
import matplotlib.pyplot as plt

from GAN_models.wind_field_GAN_3D import wind_field_GAN_3D
import iocomponents.displaybar as displaybar


def train(cfg: config.Config, dataset_train, dataset_validation, x, y):
    cfg_t = cfg.training
    status_logger = logging.getLogger("status")
    train_logger = logging.getLogger("train")
    tb_writer = tensorboardX.SummaryWriter(
        log_dir=cfg.env.this_runs_tensorboard_log_folder
    )

    torch.backends.cudnn.benckmark = True

    dataloader_train, dataloader_val = None, None
    if cfg.dataset_train:
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=cfg.dataset_train.batch_size,
            shuffle=True,
            num_workers=cfg.dataset_train.num_workers,
            pin_memory=True,
        )
        status_logger.info("finished creating training dataloader and dataset")
    else:
        raise ValueError("can't train without a training dataset - adjust the config")
    if cfg.dataset_val:
        dataloader_val = torch.utils.data.DataLoader(
            dataset_validation,
            batch_size=cfg.dataset_val.batch_size,
            shuffle=False,
            num_workers=cfg.dataset_val.num_workers,
            pin_memory=True,
        )
        status_logger.info("finished creating validation dataloader and dataset")
    else:
        status_logger.warning(
            "no validation dataset supplied! consider adjusting the config"
        )

    if cfg.model.lower() == "wind_field_gan_3d":
        gan = wind_field_GAN_3D(cfg)
        status_logger.info(f"Making model wind_field_GAN_3D from config {cfg.name}")
    else:
        status_logger.info(
            f"only wind_field_GAN_2D (wind_field_GAN_2D) and wind_field_GAN_3D(wind_field_gan_3d) is supported at this time - not {cfg.name}"
        )

    status_logger.debug(f"GAN:\n{str(gan)}\n")
    log_status_logs(status_logger, gan.get_new_status_logs())

    start_epoch = 0
    it = 0
    it_per_epoch = len(dataloader_train)
    count_train_epochs = 1 + cfg_t.niter // it_per_epoch
    loaded_it = 0
    wind_comp_dict = {0: "u", 1: "v", 2: "w"}

    if cfg.load_model_from_save:
        status_logger.info(
            f"loading model from from saves. G: {cfg.env.generator_load_path}, D: {cfg.env.discriminator_load_path}"
        )
        _, __ = gan.load_model(
            generator_load_path=cfg.env.generator_load_path,
            discriminator_load_path=cfg.env.discriminator_load_path
            if cfg.env.discriminator_load_path
            else None,
            state_load_path=None,
        )

        if cfg_t.resume_training_from_save:
            status_logger.info(
                f"resuming training from save. state: {cfg.env.state_load_path}"
            )
            loaded_epoch, loaded_it = gan.load_model(
                generator_load_path=None,
                discriminator_load_path=None,
                state_load_path=cfg.env.state_load_path,
            )
            status_logger.info(f"loaded epoch {loaded_epoch}, it {loaded_it}")
            if loaded_it:
                start_epoch = loaded_epoch
                it = loaded_it

    bar = displaybar.DisplayBar(
        max_value=len(dataloader_train),
        start_epoch=start_epoch,
        start_it=it,
        niter=cfg_t.niter,
    )

    status_logger.info(f"beginning run from epoch {start_epoch}, it {it}")
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            cfg.env.this_runs_tensorboard_log_folder
        ),
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        for epoch in range(start_epoch, count_train_epochs):
            status_logger.debug("epoch {epoch}")

            # dataloader -> (LR, HR, HR_img_name)

            for i, (LR, HR, Z) in enumerate(dataloader_train):
                if it > cfg_t.niter:
                    break

                it += 1
                bar.update(i, epoch, it)

                LR = LR.to(cfg.device, non_blocking=True)
                HR = HR.to(cfg.device, non_blocking=True)
                Z = Z.to(cfg.device, non_blocking=True)

                if it == loaded_it + 1:
                    x = x.to(cfg.device, non_blocking=True)
                    y = y.to(cfg.device, non_blocking=True)
                    gan.feed_xy_niter(
                        x,
                        y,
                        torch.tensor(cfg_t.niter, device=cfg.device),
                        cfg_t.d_g_train_ratio,
                        cfg_t.d_g_train_period,
                    )

                gan.optimize_parameters(LR, HR, Z, it)

                profiler.step()

                gan.update_learning_rate() if it > 2 * cfg_t.d_g_train_period else None

                l = gan.get_new_status_logs()

                if len(l) > 0:
                    for log in l:
                        train_logger.info(log)

                if it % cfg_t.save_model_period == 0:
                    status_logger.debug(f"saving model (it {it})")
                    gan.save_model(cfg.env.this_runs_folder, epoch, it)
                    status_logger.debug(f"storing visuals (it {it})")
                    # store_current_visuals(cfg, it, gan, dataloader_val)

                if it % cfg_t.log_period == 0:
                    if cfg.use_tensorboard_logger:
                        losses = dict(
                            (val_name, val.item())
                            for val_name, val in gan.get_G_train_loss_dict_ref().items()
                        )
                        tb_writer.add_scalars("G_loss/train", losses, it)

                if dataloader_val is None:
                    continue
                if it % cfg_t.val_period == 0:
                    status_logger.debug(f"validation epoch (it {it})")
                    G_loss_vals = dict(
                        (val_name, 0)
                        for (val_name) in gan.get_G_val_loss_dict_ref().keys()
                    )
                    D_loss_vals = dict(
                        (val_name, 0) for (val_name) in gan.get_D_loss_dict_ref().keys()
                    )

                    metrics_vals = dict(
                        (val_name, 0)
                        for (val_name) in gan.get_metrics_dict_ref().keys()
                    )
                    n = len(dataloader_val)
                    for _, (LR, HR, Z) in enumerate(dataloader_val):
                        LR = LR.to(cfg.device, non_blocking=True)
                        HR = HR.to(cfg.device, non_blocking=True)
                        Z = Z.to(cfg.device, non_blocking=True)
                        gan.validation(LR, HR, Z, it)
                        for val_name, val in gan.get_G_val_loss_dict_ref().items():
                            G_loss_vals[val_name] += val.item() / n

                        for val_name, val in gan.get_D_loss_dict_ref().items():
                            D_loss_vals[val_name] += val.item() / n

                        for val_name, val in gan.get_metrics_dict_ref().items():
                            metrics_vals[val_name] += val.item() / n

                    batch_quiver = torch.randint(
                        LR.shape[0], size=(1,), device=cfg.device
                    )[0]

                    LR_i = torch.index_select(LR, 0, batch_quiver, out=None)

                    TL_i = (
                        dataset_train.UVW_MAX
                        * nn.functional.interpolate(
                            LR_i[:, :3],
                            scale_factor=(cfg.scale, cfg.scale, 1),
                            mode="trilinear",
                            align_corners=True,
                        ).squeeze()
                    )

                    with torch.no_grad():
                        SR_i = (
                            dataset_train.UVW_MAX
                            * gan.G(
                                LR_i,
                                torch.index_select(Z, 0, batch_quiver, out=None),
                            )
                        ).squeeze()

                    LR_i = dataset_train.UVW_MAX * LR_i[0, :3]
                    HR_i = (
                        dataset_train.UVW_MAX
                        * torch.index_select(HR, 0, batch_quiver, out=None).squeeze()
                    )

                    if cfg.use_tensorboard_logger:
                        rand_wind_comp = np.random.randint(0, 3)
                        rand_z_index = np.random.randint(0, HR_i.shape[-1])

                        pix_criterion = nn.L1Loss()

                        std_slice_SR_error = pix_criterion(
                            HR_i[0][:, :, 3],
                            SR_i[0][:, :, 3],
                        ).item()
                        std_slice_TL_error = pix_criterion(
                            HR_i[0][:, :, 3],
                            TL_i[0][:, :, 3],
                        ).item()

                        rand_slice_SR_error = pix_criterion(
                            HR_i[rand_wind_comp][:, :, rand_z_index],
                            SR_i[rand_wind_comp][:, :, rand_z_index],
                        ).item()

                        rand_slice_TL_error = pix_criterion(
                            HR_i[rand_wind_comp][:, :, rand_z_index],
                            TL_i[rand_wind_comp][:, :, rand_z_index],
                        ).item()

                        HR_i = HR_i.cpu().numpy()
                        SR_i = SR_i.cpu().numpy()
                        TL_i = TL_i.cpu().numpy()
                        LR_i = LR_i.cpu().numpy()

                        save_validation_images_to_tb(
                            "u_field_z_index",
                            3,
                            LR_i[0],
                            HR_i[0],
                            SR_i[0],
                            TL_i[0],
                            tb_writer,
                            it,
                            round(std_slice_SR_error, 3),
                            round(std_slice_TL_error, 3),
                        )

                        save_validation_images_to_tb(
                            wind_comp_dict[rand_wind_comp] + "_field_z_index",
                            rand_z_index,
                            LR_i[rand_wind_comp],
                            HR_i[rand_wind_comp],
                            SR_i[rand_wind_comp],
                            TL_i[rand_wind_comp],
                            tb_writer,
                            it,
                            round(rand_slice_SR_error, 3),
                            round(rand_slice_TL_error, 3),
                        )

                        tb_writer.add_scalars("G_loss/validation", G_loss_vals, it)
                        tb_writer.add_scalars("D_loss/", D_loss_vals, it)
                        # for hist_name, val in hist_vals.items():
                        #    tb_writer.add_histogram(f"data/hist/{hist_name}", val, it)
                        PSNR_metrics = dict(
                            (key, value)
                            for key, value in metrics_vals.items()
                            if "PSNR" in key
                        )
                        pix_metrics = dict(
                            (key, value)
                            for key, value in metrics_vals.items()
                            if "pix" in key
                        )
                        tb_writer.add_scalars("metrics/PSNR", PSNR_metrics, it)
                        tb_writer.add_scalars("metrics/pix", pix_metrics, it)

                        imgs = dict()
                        imgs["HR"] = HR_i
                        imgs["SR"] = SR_i
                        imgs["BC"] = TL_i
                        imgs["LR"] = LR_i

                    else:
                        imgs = dict()
                        imgs["HR"] = HR_i.cpu().numpy()
                        imgs["SR"] = SR_i.cpu().numpy()
                        imgs["BC"] = TL_i.cpu().numpy()
                        imgs["LR"] = LR_i.cpu().numpy()

                    with open(
                        cfg.env.this_runs_folder
                        + "/images/val_imgs__it_"
                        + str(it)
                        + ".pkl",
                        "wb",
                    ) as f:
                        pkl.dump(imgs, f)

                    stat_log_str = f"it: {it} "
                    for k, v in G_loss_vals.items():
                        stat_log_str += f"{k}: {v} "
                    for k, v in metrics_vals.items():
                        stat_log_str += f"{k}: {v} "
                    status_logger.debug(stat_log_str)
    return


def save_validation_images_to_tb(
    title,
    wind_height_index,
    wind_comp_LR,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_TL,
    tb_writer,
    it,
    avg_loss,
    average_TL_error,
):
    fig1 = create_comparison_figure(
        wind_height_index,
        wind_comp_LR,
        wind_comp_HR,
        wind_comp_SR,
        wind_comp_TL,
    )

    tb_writer.add_figure(
        "im/" + str(it) + "/wind_fields/" + title + str(wind_height_index), fig1, it
    )

    fig2 = create_error_figure(
        wind_height_index,
        wind_comp_HR,
        wind_comp_SR,
        wind_comp_TL,
        avg_loss,
        average_TL_error,
    )

    tb_writer.add_figure(
        "im/" + str(it) + "/Error/" + title + str(wind_height_index), fig2, it
    )


def log_status_logs(status_logger: logging.Logger, logs: list):
    for log in logs:
        status_logger.info(log)


def create_error_figure(
    wind_height_index,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_TL,
    average_SR_error,
    average_TL_error,
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
    axes2[0, 1].set_title(f"SR, avg error: {round(average_SR_error,3)} m/s")
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
    axes2[1, 1].set_title(f"TL, avg error: {round(average_TL_error,3)} m/s")
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
