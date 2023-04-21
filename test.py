"""
test.py
Written by Eirik Vesterkjær, 2019, edited by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements GAN testing, and upscaling LR images
Use run.py to run.
"""

import logging
import math
import os

import pickle as pkl
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import config.config as config
import GAN_models.wind_field_GAN_3D as wind_field_GAN_3D
import GAN_models.wind_field_GAN_2D as wind_field_GAN_2D
from process_data import preprosess


def test(cfg: config.Config, dataset_test):
    status_logger = logging.getLogger("status")

    dataloader_test = None
  
    if cfg.dataset_test is not None:
        if cfg.dataset_test.mode.lower() == "hrlr":
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=1,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            # dataloader_test = imageset.createDataloader(cfg, is_test_dataloader=True, downsampler_mode="bicubic")
        # elif cfg.dataset_test.mode.lower() == "lr":
        #     dataloader_test = imageset.createDataloader(cfg, is_test_dataloader=True)
    else:
        raise ValueError("Test dataset not supplied")

    gan: nn.Module = None
    if cfg.model.lower() == "wind_field_gan_2d":
        gan = wind_field_GAN_2D(cfg)
        status_logger.info(f"Making model wind_field_GAN_2D from config {cfg.name}")
    elif cfg.model.lower() == "wind_field_gan_3d":
        gan = wind_field_GAN_3D(cfg)
        status_logger.info(f"Making model wind_field_GAN_3D from config {cfg.name}")
    else:
        status_logger.info(
            f"only wind_field_GAN_2D (wind_field_GAN_2D) and wind_field_GAN_3D(wind_field_gan_3d) is supported at this time - not {cfg.name}"
        )

    status_logger.info(
        f"loading model from from saves. G: {cfg.env.generator_load_path}, D: {cfg.env.discriminator_load_path}"
    )
    _, __ = gan.load_model(
        generator_load_path=cfg.env.generator_load_path,
        discriminator_load_path=cfg.env.discriminator_load_path,
        state_load_path=None,
    )

    status_logger.info(f"beginning test")

    output_folder_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)) + "/", "../output"
    )
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    metrics_file = os.path.join(output_folder_path + "/" + cfg.name + "_metrics.csv")

    if cfg.is_use:
        for j, data in enumerate(dataloader_test):
            status_logger.info(f"batch {j}")
            lrs = data[0]
            names = data[1]

            for i in range(len(lrs)):
                status_logger.info(f"image {i}")
                indx = torch.tensor([i])
                lr_i = torch.index_select(lrs, 0, indx, out=None)
                img_name = names[i]
                sr_i = gan.G(lr_i.float().to(cfg.device)).cpu()
                make_and_write_images(
                    lr_i, None, sr_i, output_folder_path, img_name, cfg.scale
                )

    if cfg.is_test:
        with open(metrics_file, "w") as f:
            f.write("image,PSNR\n")
            for j, data in enumerate(dataloader_test):
                status_logger.info(f"batch {j}")
                lrs = data[0]
                hrs = data[1]
                # names = data["hr_name"]

                for i in range(len(lrs)):
                    status_logger.info(f"image {i}")
                    indx = torch.tensor([i])
                    lr_i = torch.index_select(lrs, 0, indx, out=None)
                    hr_i = torch.index_select(hrs, 0, indx, out=None)
                    # img_name = names[i]

                    sr_i = gan.G(lr_i.float().to(cfg.device)).cpu()
                    # img_name = "lr_bc_sr_hr"

                    imgs = make_and_write_images(
                        lr_i, hr_i, sr_i, cfg.env.this_runs_folder, j, cfg.scale
                    )

                    write_metrics(imgs, j, f)


def make_and_write_images(
    lr: torch.Tensor,
    hr: torch.Tensor,
    sr: torch.Tensor,
    folder_path: str,
    img_name: int,
    scale: int,
) -> dict:
    # transpose: c,h,w -> cv2 img shape h,w,c
    also_save_hr = True
    if hr is None:
        hr = sr
        also_save_hr = False

    # ch w h -> w, h, ch as numpy
    lr_np = lr.squeeze().detach().cpu().numpy()
    hr_np = hr.squeeze().detach().cpu().numpy()
    sr_G_np = sr.squeeze().detach().cpu().numpy()

    # Get LR data
    u_lr = lr_np[0, :, :]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
    v_lr = lr_np[1, :, :]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
    w_lr = lr_np[2, :, :]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)
    # xwp_lr = lr_np[3, :, :]  # *(np.max(x_wp_nomask)-np.min(x_wp_nomask))+np.min(x_wp_nomask)
    # zwp_lr = lr_np[4, :, :]  # *(np.max(z_wp_nomask)-np.min(z_wp_nomask))+np.min(z_wp_nomask)

    # Bicubic
    # imgs_bicubic_tensor = nn.functional.interpolate(lr_i, scale_factor=4, mode='bicubic')
    imgs_bicubic = (
        nn.functional.interpolate(
            lr, scale_factor=4, mode="bicubic", align_corners=True
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    u_bicubic = imgs_bicubic[
        0, :, :
    ]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
    v_bicubic = imgs_bicubic[
        1, :, :
    ]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
    w_bicubic = imgs_bicubic[
        2, :, :
    ]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

    # Generated HR
    u_sr = sr_G_np[0, :, :]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
    v_sr = sr_G_np[1, :, :]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
    w_sr = sr_G_np[2, :, :]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

    # HR
    u_hr = hr_np[0, :, :]  # *(np.max(u_nomask)-np.min(u_nomask))+np.min(u_nomask)
    v_hr = hr_np[1, :, :]  # *(np.max(v_nomask)-np.min(v_nomask))+np.min(v_nomask)
    w_hr = hr_np[2, :, :]  # *(np.max(w_nomask)-np.min(w_nomask))+np.min(w_nomask)

    # Store results to file:
    hr_img = [u_hr, v_hr, w_hr]
    sr_img = [u_sr, v_sr, w_sr]
    bc_img = [u_bicubic, v_bicubic, w_bicubic]
    lr_img = [u_lr, v_lr, w_lr]

    imgs = dict()
    imgs["HR"] = np.array(hr_img)
    imgs["SR"] = np.array(sr_img)
    imgs["BC"] = np.array(bc_img)
    imgs["LR"] = np.array(lr_img)

    print("Func: make_and_write_images():")
    dest_file = folder_path + "/images/test_imgs_" + str(img_name) + ".pkl"
    with open(dest_file, "wb") as f:
        pkl.dump(imgs, f)

    # return { "LR": lr_np, "HR": hr_np, "SR": sr_G_np, "SR_bicubic": sr_bicubic_np} # "SR_nearest": sr_nn_np,
    return imgs


def write_metrics(images: dict, img_name: int, dest_file):
    print("Func: write_metrics")

    # Store MSE loss for both SR an BC images vs HR.
    psnr = img_psnr(images["SR"], images["HR"])
    dest_file.write(f"{img_name}_sr,{psnr}" + "\n")
    # psnr_nn = img_psnr(images["SR_nearest"], images["HR"])
    # dest_file.write(f"{img_name}_nearest,{psnr_nn}"+"\n")
    psnr_bic = img_psnr(images["BC"], images["HR"])
    dest_file.write(f"{img_name}_bicubic,{psnr_bic}" + "\n")


def img_psnr(sr, hr) -> float:
    print("img_psnr: hr_shape:", hr.shape)
    w, h, c = hr.shape[0], hr.shape[1], hr.shape[2]
    sr = sr.reshape(w * h * c)
    hr = hr.reshape(w * h * c)

    MSE = np.square((hr - sr)).sum(axis=0) / (w * h * c)
    MSE = MSE.item()
    R_squared = (
        np.max(hr) ** 2
    )  # 255.0*255.0 # R is max fluctuation, and data is cv2 img: int [0, 255] -> R² = 255²
    epsilon = 1e-8  # PSNR is usually ~< 50 so this should not impact the result much
    return 10 * math.log10(R_squared / (MSE + epsilon))
