"""
train.py
Written by Eirik Vesterkjær, 2019, edited by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements a GAN training loop
Use run.py to run.
"""
import logging
import os
import cv2
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
        # dataloader_train = imageset.createDataloader(cfg, is_train_dataloader=True, downsampler_mode="trilinear")
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
        # dataloader_val = imageset.createDataloader(cfg, is_validation_dataloader=True, downsampler_mode="trilinear")
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
            discriminator_load_path=cfg.env.discriminator_load_path,
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

    status_logger.info(
        "storing LR and HR validation images in run folder, for reference"
    )
    # store_LR_HR_in_runs_folder(cfg, dataloader_val)

    # Debugging
    if torch.cuda.is_available():
        print(
            "Current device:",
            torch.cuda.current_device(),
            "- num devices:",
            torch.cuda.device_count(),
            "- device name:",
            torch.cuda.get_device_name(0),
        )
    # end debugging
    status_logger.info(f"beginning run from epoch {start_epoch}, it {it}")
    training_time_dict = {}
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
                        x, y, torch.tensor(cfg_t.niter, device=cfg.device), cfg_t.d_g_train_ratio, cfg_t.d_g_train_period
                    )

                gan.optimize_parameters(LR, HR, Z, it)

                profiler.step()

                gan.update_learning_rate() if it > 2*cfg_t.d_g_train_period else None

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
                        (val_name, 0)
                        for (val_name) in gan.get_D_loss_dict_ref().keys()
                    )

                    metrics_vals = dict(
                        (val_name, 0)
                        for (val_name) in gan.get_metrics_dict_ref().keys()
                    )
                    n = len(dataloader_val)
                    i_val = 0
                    for _, (LR, HR, Z) in enumerate(dataloader_val):
                        i_val += 1
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
                        # VISUALIZING
                        # if (it % 40000 == 0):
                        if i_val % len(dataloader_val) == 0:
                            with torch.no_grad():
                                batch_quiver = torch.randint(
                                    LR.shape[0], size=(1,), device=cfg.device
                                )[0]

                                # Fetch validation sample from device, using random index
                                LR_i = torch.index_select(LR, 0, batch_quiver, out=None)
                                HR_i = (
                                    torch.index_select(HR, 0, batch_quiver, out=None)
                                    .squeeze()
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )

                                # Low Resolution
                                LR_ori = LR_i.squeeze().detach().cpu().numpy()
                                u_LR = dataset_train.UVW_MAX*LR_ori[0, :, :, :]
                                v_LR = dataset_train.UVW_MAX*LR_ori[1, :, :, :]
                                w_LR = dataset_train.UVW_MAX*LR_ori[2, :, :, :]
                                imgs_trilinear = (
                                    nn.functional.interpolate(
                                        LR_i,
                                        scale_factor=(4, 4, 1),
                                        mode="trilinear",
                                        align_corners=True,
                                    )
                                    .squeeze()
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                u_trilinear = dataset_train.UVW_MAX*imgs_trilinear[
                                    0, :, :, :
                                ]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
                                v_trilinear = dataset_train.UVW_MAX*imgs_trilinear[
                                    1, :, :, :
                                ]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
                                w_trilinear = dataset_train.UVW_MAX*imgs_trilinear[
                                    2, :, :, :
                                ]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

                                # loss_trilinear = torch.sum(((HR_i - imgs_trilinear_tensor)**2)/(128*128)).item()
                                # rounded_loss_trilinear = round(loss_trilinear, 3)

                                # Generated HR
                                # loss_sr = torch.sum(((HR_i - gan.G(LR_i))**2)/(128*128)).item()
                                # rounded_loss_sr = round(loss_sr, 3)
                                gen_HR = (
                                    gan.G(
                                        LR_i,
                                        torch.index_select(
                                            Z, 0, batch_quiver, out=None
                                        ),
                                    )
                                    .squeeze()
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                u_sr = dataset_train.UVW_MAX*gen_HR[
                                    0, :, :, :
                                ]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
                                v_sr = dataset_train.UVW_MAX*gen_HR[
                                    1, :, :, :
                                ]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
                                w_sr = dataset_train.UVW_MAX*gen_HR[
                                    2, :, :, :
                                ]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

                                # HR
                                u_HR = dataset_train.UVW_MAX*HR_i[
                                    0, :, :, :
                                ]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
                                v_HR = dataset_train.UVW_MAX*HR_i[
                                    1, :, :, :
                                ]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
                                w_HR = dataset_train.UVW_MAX*HR_i[
                                    2, :, :, :
                                ]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

                                # Store results to file:
                                HR_img = [u_HR, v_HR, w_HR]
                                sr_img = [u_sr, v_sr, w_sr]
                                tl_img = [u_trilinear, v_trilinear, w_trilinear]
                                LR_img = [u_LR, v_LR, w_LR]

                                # tb_writer.add_image(
                                #     "LR_" + str(it), LR_i[:, :3, :, :, 3].squeeze()
                                # )
                                # tb_writer.add_image(
                                #     "HR_" + str(it), HR_i[:, :3, :, :, 3].squeeze()
                                # )
                                # tb_writer.add_image(
                                #     "SR_" + str(it), gen_HR[:, :3, :, :, 3].squeeze()
                                # )
                                rand_wind_comp = np.random.randint(0, 3)
                                rand_z_index = np.random.randint(0, u_HR.shape[2])

                                pix_criterion = nn.L1Loss()
                                u_SR_loss = pix_criterion(torch.from_numpy(HR_img[0][:,:,3]), torch.from_numpy(sr_img[0][:,:,3])).item()
                                u_trilinear_loss = pix_criterion(torch.from_numpy(HR_img[0][:,:,3]), torch.from_numpy(tl_img[0][:,:,3])).item()

                                rand_avg_loss = pix_criterion(torch.from_numpy(HR_img[rand_wind_comp][:,:,rand_z_index]), torch.from_numpy(sr_img[rand_wind_comp][:,:,rand_z_index])).item()
                                rand_avg_loss_trilinear = pix_criterion(torch.from_numpy(HR_img[rand_wind_comp][:,:,rand_z_index]), torch.from_numpy(tl_img[rand_wind_comp][:,:,rand_z_index])).item()


                                save_validation_images(
                                    "u_field_z_index",
                                    3,
                                    u_LR,
                                    u_HR,
                                    u_sr,
                                    u_trilinear,
                                    tb_writer,
                                    it,
                                    round(u_SR_loss, 3),
                                    round(u_trilinear_loss, 3),
                                )
                                save_validation_images(
                                    wind_comp_dict[rand_wind_comp] + "_field_z_index",
                                    rand_z_index,
                                    LR_img[rand_wind_comp],
                                    HR_img[rand_wind_comp],
                                    sr_img[rand_wind_comp],
                                    tl_img[rand_wind_comp],
                                    tb_writer,
                                    it,
                                    round(rand_avg_loss, 3),
                                    round(rand_avg_loss_trilinear,3),
                                )

                                imgs = dict()
                                imgs["HR"] = HR_img
                                imgs["SR"] = sr_img
                                imgs["BC"] = tl_img
                                imgs["LR"] = LR_img

                                with open(
                                    cfg.env.this_runs_folder
                                    + "/images/val_imgs__it_"
                                    + str(it)
                                    + ".pkl",
                                    "wb",
                                ) as f:
                                    pkl.dump(imgs, f)

                    if cfg.use_tensorboard_logger:
                        tb_writer.add_scalars("G_loss/validation", G_loss_vals, it)
                        tb_writer.add_scalars("D_loss/", D_loss_vals, it)
                        # for hist_name, val in hist_vals.items():
                        #    tb_writer.add_histogram(f"data/hist/{hist_name}", val, it)
                        PSNR_metrics = dict((key, value) for key, value in metrics_vals.items() if "PSNR" in key)
                        pix_metrics = dict((key, value) for key, value in metrics_vals.items() if "pix" in key)
                        tb_writer.add_scalars("metrics/PSNR", PSNR_metrics, it)
                        tb_writer.add_scalars("metrics/pix", pix_metrics, it)

                    stat_log_str = f"it: {it} "
                    for k, v in G_loss_vals.items():
                        stat_log_str += f"{k}: {v} "
                    for k, v in metrics_vals.items():
                        stat_log_str += f"{k}: {v} "
                    status_logger.debug(stat_log_str)

            if torch.cuda.is_available() and epoch == start_epoch + 1:
                prev = 0
                for key, value in gan.memory_dict.items():
                    diff = value - prev
                    status_logger.debug(
                        key
                        + " memory usage (MB) "
                        + str(value)
                        + ", diff from previous: "
                        + str(diff)
                    )
                    prev = value
                status_logger.info(
                    "max memory allocated: "
                    + str(torch.cuda.max_memory_allocated(cfg.device) / 1024**2)
                )
                status_logger.info("devices D_forward: " + gan.device_check)
    return

def create_error_figure(
    wind_height_index,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_trilinear,
    average_SR_error,
    average_trilinear_error,
):
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('viridis') )
    vmin, vmax = np.min(wind_comp_HR[:, :, wind_height_index]), np.max(
        wind_comp_HR[:, :, wind_height_index]
    )
    vmin_wind_field, vmax_wind_field = np.min(np.concatenate((wind_comp_trilinear[:, :, wind_height_index], wind_comp_SR[:, :, wind_height_index]), axis=(0))), np.max(np.concatenate((wind_comp_trilinear[:, :, wind_height_index], wind_comp_SR[:, :, wind_height_index]), axis=(0)))
    vmin_error, vmax_error = np.min(np.concatenate((wind_comp_trilinear[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index], wind_comp_SR[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index]), axis=(0))), np.max(np.concatenate((wind_comp_trilinear[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index], wind_comp_SR[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index]), axis=(0)))
    vmin_abs_error, vmax_abs_error = 0.0, max(abs(vmax_error), abs(vmin_error))
    sm.set_clim(vmin=vmin, vmax=vmax)
    sm_error = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('coolwarm'))
    sm_error.set_clim(vmin=vmin_error, vmax=vmax_error)
    sm_abs_error = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet'))
    sm_abs_error.set_clim(vmin=vmin_abs_error, vmax=vmax_abs_error)

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex=True)
    axes2[0,1].pcolor(wind_comp_SR[:, :, wind_height_index], vmin=vmin_wind_field, vmax=vmax_wind_field, cmap="viridis")
    axes2[0,1].set_title(f"SR wind field, avg error: {average_SR_error} m/s")
    axes2[0,0].pcolor(
        wind_comp_SR[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index],
        vmin=vmin_error,
        vmax=vmax_error,
        cmap="coolwarm",
    )
    axes2[0,0].set_title("Error SR-HR (m/s)")
    axes2[0,2].pcolor(
        abs(
            wind_comp_HR[:, :, wind_height_index]
            - wind_comp_SR[:, :, wind_height_index]
        ),
        vmin=vmin_abs_error,
        vmax=vmax_abs_error,
        cmap="jet",
    )
    axes2[0,2].set_title("Absolute error SR (m/s)")
    fig2.colorbar(sm, ax=axes2[0,1])
    fig2.colorbar(
        sm_error,
        ax=axes2[0,0],
    )

    fig2.colorbar(
        sm_abs_error,
        ax=axes2[0,2],
    )
    axes2[1,1].pcolor(wind_comp_trilinear[:, :, wind_height_index])
    axes2[1,1].set_title(f"Trilinear wind field, avg error: {average_trilinear_error} m/s")
    axes2[1,0].pcolor(
        wind_comp_trilinear[:, :, wind_height_index] - wind_comp_HR[:, :, wind_height_index],
        cmap="coolwarm",
    )
    axes2[1,0].set_title("Error Trilinear-HR (m/s)")
    axes2[1,2].pcolor(
        abs(
            wind_comp_HR[:, :, wind_height_index]
            - wind_comp_trilinear[:, :, wind_height_index]
        ),
        cmap="jet",
    )
    axes2[1,2].set_title("Absolute error Trilinear (m/s)")
    fig2.colorbar(sm, ax=axes2[1,1])
    fig2.colorbar(
        sm_error,
        ax=axes2[1,0],
    )

    fig2.colorbar(
        sm_abs_error,
        ax=axes2[1,2],
    )
    fig2.subplots_adjust(hspace=0.2)
    return fig2

def create_comparison_figure(
    wind_height_index,
    wind_comp_LR,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_trilinear,
):    
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    vmin, vmax = np.min(wind_comp_HR[:, :, wind_height_index]), np.max(
        wind_comp_HR[:, :, wind_height_index]
    )
    axes[0, 0].pcolor(
        wind_comp_LR[:, :, wind_height_index], vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[0, 0].set_title("LR")
    axes[0, 1].pcolor(
        wind_comp_HR[:, :, wind_height_index], vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[0, 1].set_title("HR")
    axes[1, 0].pcolor(
        wind_comp_SR[:, :, wind_height_index], vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[1, 0].set_title("SR")
    axes[1, 1].pcolor(
        wind_comp_trilinear[:, :, wind_height_index],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[1, 1].set_title("Trilinear")
    fig.subplots_adjust(hspace=0.3)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('viridis') )
    sm.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(sm, ax=axes)
    return fig

def save_validation_images(
    title,
    wind_height_index,
    wind_comp_LR,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_trilinear,
    tb_writer,
    it,
    avg_loss,
    average_trilinear_error,
):
    fig1 = create_comparison_figure(
        wind_height_index,
        wind_comp_LR,
        wind_comp_HR,
        wind_comp_SR,
        wind_comp_trilinear,
    )
    
    tb_writer.add_figure(
        "im/"+str(it)+"/wind_fields/" + title + str(wind_height_index), fig1, it
    )
    
    fig2 = create_error_figure(wind_height_index, wind_comp_HR, wind_comp_SR, wind_comp_trilinear, avg_loss, average_trilinear_error)
    
    tb_writer.add_figure(
        "im/"+str(it)+"/Error/" + title + str(wind_height_index), fig2, it
    )


def log_status_logs(status_logger: logging.Logger, logs: list):
    for log in logs:
        status_logger.info(log)


def store_LR_HR_in_runs_folder(cfg: config.Config, dataloader):
    HR_LR_folder_path = os.path.join(cfg.env.this_runs_folder + "/", f"HR_LR")
    if not os.path.exists(HR_LR_folder_path):
        os.makedirs(HR_LR_folder_path)

    for it, data in enumerate(dataloader):
        LRs = data[0]
        HRs = data[1]
        # handles batch sizes > 0
        for i in range(LRs.shape[0]):
            indx = torch.as_tensor([i])
            LR_i = torch.index_select(LRs, 0, indx, out=None)
            HR_i = torch.index_select(HRs, 0, indx, out=None)
            LR_np = LR_i.squeeze().detach().numpy()  # * 255
            HR_np = HR_i.squeeze().detach().numpy()  # * 255

            # CxHxW -> HxWxC
            # LR_np = LR_np.transpose((1,2,0))
            # HR_np = HR_np.transpose((1,2,0))

            img_name = data["HR_name"][i]
            filename_LR = os.path.join(HR_LR_folder_path, f"{img_name}_LR.png")
            filename_HR = os.path.join(HR_LR_folder_path, f"{img_name}.png")

            cv2.imwrite(filename_LR, LR_np)
            cv2.imwrite(filename_HR, HR_np)


def store_current_visuals(cfg: config.Config, it, gan, dataloader):
    it_folder_path = os.path.join(cfg.env.this_runs_folder + "/", f"{it}_visuals")
    if not os.path.exists(it_folder_path):
        os.makedirs(it_folder_path)

    for v, (LRs, HRs, Zs) in enumerate(dataloader):
        LRs = LRs.to(cfg.device)
        for i in range(LRs.shape[0]):
            indx = torch.as_tensor([i]).to(cfg.device)
            LR_i = torch.index_select(LRs, 0, indx, out=None)
            # new record in # of .calls ?
            sr_np = (
                gan.G(LR_i, torch.index_select(Zs, 0, indx, out=None))
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                * 255
            )
            sr_np[sr_np < 0] = 0
            sr_np[sr_np > 255] = 255

            # c,h,w -> cv2 img shape h,w,c
            sr_np = sr_np.transpose((1, 2, 0))

            # img_name = data["HR_name"][i]

            # filename_HR_generated = os.path.join(it_folder_path, f"{img_name}_{it}.png")
            # cv2.imwrite(filename_HR_generated, sr_np)
