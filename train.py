"""
train.py
Written by Eirik Vesterkjær, 2019, edited by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements a GAN training loop
Use run.py to run.
"""
import logging
import random
import os

import cv2
import pickle as pkl
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import tensorboardX
from process_data import preprosess
import config.config as config
import matplotlib.pyplot as plt


# import data.imageset as imageset
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

    # gan: nn.Module = None
    if cfg.model.lower() == "wind_field_gan_3d":
        gan = wind_field_GAN_3D(cfg)
        status_logger.info(f"Making model wind_field_GAN_3D from config {cfg.name}")
    else:
        status_logger.info(
            f"only wind_field_GAN_2D (wind_field_GAN_2D) and wind_field_GAN_3D(wind_field_gan_3d) is supported at this time - not {cfg.name}"
        )

    status_logger.info(f"GAN:\n{str(gan)}\n")
    log_status_logs(status_logger, gan.get_new_status_logs())

    start_epoch = 0
    it = 0
    it_per_epoch = len(dataloader_train)
    count_train_epochs = 1+cfg_t.niter // it_per_epoch
    
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
    with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.env.this_runs_tensorboard_log_folder),
    with_stack=True
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
                
                if it == 1 if cfg_t.resume_training_from_save else it == loaded_it + 1:
                    x = x.to(cfg.device, non_blocking=True)
                    y = y.to(cfg.device, non_blocking=True)
                    gan.feed_data(x, y)

                gan.optimize_parameters(LR, HR, Z, it)
                
                profiler.step()

                if i == 1 and torch.cuda.is_available():
                    prev = 0
                    for key, value in gan.memory_dict.items():
                        diff = value - prev
                        status_logger.info(key+" memory usage (MB) "+str(value)+", diff from previous: "+str(diff))
                        prev = value
                
                gan.update_learning_rate() if i > 0 else None

                l = gan.get_new_status_logs()

                if len(l) > 0:
                    for log in l:
                        train_logger.info(log)

                if it % cfg_t.save_model_period == 0:
                    status_logger.debug(f"saving model (it {it})")
                    gan.save_model(cfg.env.this_runs_folder, epoch, it)
                    status_logger.debug(f"storing visuals (it {it})")
                    # store_current_visuals(cfg, it, gan, dataloader_val)

                if dataloader_val is None:
                    continue
                if it % cfg_t.val_period == 0:
                    del LR, HR, Z
                    status_logger.debug(f"validation epoch (it {it})")
                    loss_vals = dict(
                        (val_name, val * 0)
                        for (val_name, val) in gan.get_loss_dict_ref().items()
                    )
                    metrics_vals = dict(
                        (val_name, val * 0)
                        for (val_name, val) in gan.get_metrics_dict_ref().items()
                    )
                    n = len(dataloader_val)
                    # VISUALIZING
                    # it_folder_path = os.path.join(cfg.env.this_runs_folder + "/", f"{it}_visuals" )
                    # if not os.path.exists(it_folder_path):
                    #    os.makedirs(it_folder_path)
                    os.makedirs("images/training/grnd_est", exist_ok=True)

                    i_val = 0
                    for _, (val_LR, val_HR, val_Z) in enumerate(dataloader_val):
                        i_val += 1
                        val_LR = val_LR.to(cfg.device, non_blocking=True)
                        val_HR = val_HR.to(cfg.device, non_blocking=True)
                        val_Z = val_Z.to(cfg.device, non_blocking=True)
                        gan.validation(val_LR, val_HR, val_Z, it)
                        for val_name, val in gan.get_loss_dict_ref().items():
                            loss_vals[val_name] += val / n

                        for val_name, val in gan.get_metrics_dict_ref().items():
                            metrics_vals[val_name] += val / n
                        # VISUALIZING
                        # if (it % 40000 == 0):
                        if i_val % len(dataloader_val) == 0:
                            with torch.no_grad():
                                batch_quiver = torch.randint(val_LR.shape[0], size=(1,), device=cfg.device)[0] 

                                # Fetch validation sample from device, using random index
                                LR_i = torch.index_select(val_LR, 0, batch_quiver, out=None)
                                HR_i = torch.index_select(val_HR, 0, batch_quiver, out=None).squeeze().detach().cpu().numpy()

                                # Low Resolution
                                LR_ori = LR_i.squeeze().detach().cpu().numpy()
                                u_LR = LR_ori[
                                    0, :, :, :
                                ]  
                                v_LR = LR_ori[
                                    1, :, :, :
                                ] 
                                w_LR = LR_ori[
                                    2, :, :, :
                                ]  
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
                                u_trilinear = imgs_trilinear[
                                    0, :, :, :
                                ]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
                                v_trilinear = imgs_trilinear[
                                    1, :, :, :
                                ]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
                                w_trilinear = imgs_trilinear[
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
                                            val_Z, 0, batch_quiver, out=None
                                        ),
                                    )
                                    .squeeze()
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                u_sr = gen_HR[
                                    0, :, :, :
                                ]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
                                v_sr = gen_HR[
                                    1, :, :, :
                                ]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
                                w_sr = gen_HR[
                                    2, :, :, :
                                ]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

                                # HR
                                u_HR = HR_i[
                                    0, :, :, :
                                ]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
                                v_HR = HR_i[
                                    1, :, :, :
                                ]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
                                w_HR = HR_i[
                                    2, :, :, :
                                ]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

                                # Store results to file:
                                HR_img = [u_HR, v_HR, w_HR]
                                sr_img = [u_sr, v_sr, w_sr]
                                bc_img = [u_trilinear, v_trilinear, w_trilinear]
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

                                fig, axes = plt.subplots(2, 2)
                                axes[0, 0].pcolor(u_LR[:, :, 3])
                                axes[0, 0].set_title("LR u, z=3")
                                axes[0, 1].pcolor(u_HR[:, :, 3])
                                axes[0, 1].set_title("HR u, z=3")
                                axes[1, 0].pcolor(u_sr[:, :, 3])
                                axes[1, 0].set_title("SR u, z=3")
                                axes[1, 1].pcolor(u_trilinear[:, :, 3])
                                axes[1, 1].set_title("Trilinear u, z=3")
                                tb_writer.add_figure("u_field_" + str(it), fig, it)

                                imgs = dict()
                                imgs["HR"] = HR_img
                                imgs["SR"] = sr_img
                                imgs["BC"] = bc_img
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
                        tb_writer.add_scalars("data/losses", loss_vals, it)
                        # for hist_name, val in hist_vals.items():
                        #    tb_writer.add_histogram(f"data/hist/{hist_name}", val, it)
                        tb_writer.add_scalars("data/metrics", metrics_vals, it)

                    stat_log_str = f"it: {it} "
                    for k, v in loss_vals.items():
                        stat_log_str += f"{k}: {v} "
                    for k, v in metrics_vals.items():
                        stat_log_str += f"{k}: {v} "
                    status_logger.debug(stat_log_str)
                    if torch.cuda.is_available() and it // cfg_t.val_period == 1:
                        status_logger.info("max memory allocated: "+str(torch.cuda.max_memory_allocated(cfg.device)/1024**2))
                    # store_current_visuals(cfg, 0, gan, dataloader_val) # tricky way of always having the newest images.
    return


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

            img_name = data["HR_name"][i]

            filename_HR_generated = os.path.join(it_folder_path, f"{img_name}_{it}.png")
            cv2.imwrite(filename_HR_generated, sr_np)
