"""
test.py
Written by Eirik Vesterkjær, 2019, edited by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements GAN testing, and upscaling LR fields
Use run.py to run.
"""

import logging
import os
import pickle as pkl
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import config.config as config
from GAN_models.wind_field_GAN_3D import wind_field_GAN_3D, calculate_PSNR
import iocomponents.displaybar as displaybar


def test(cfg: config.Config, dataset_test):
    status_logger = logging.getLogger("status")
    wind_comp_dict = {0: "u", 1: "v", 2: "w"}
    num_epochs = 4 if cfg.gan_config.enable_slicing else 1
    niter = len(dataset_test)*num_epochs
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
    else:
        raise ValueError("Test dataset not supplied")

    if cfg.model.lower() == "wind_field_gan_3d":
        gan = wind_field_GAN_3D(cfg)
        status_logger.info(f"Making model wind_field_GAN_3D from config {cfg.name}")
    else:
        status_logger.info(
            f"only wind_field_GAN_2D (wind_field_GAN_2D) and wind_field_GAN_3D(wind_field_gan_3d) is supported at this time - not {cfg.name}"
        )

    status_logger.info(
        f"loading model from from saves. G: {cfg.env.generator_load_path}"
    )
    _, __ = gan.load_model(
        generator_load_path=cfg.env.generator_load_path,
        discriminator_load_path=None,
        state_load_path=None,
    )

    status_logger.info(f"beginning test")

    if not os.path.exists(cfg.env.this_runs_folder + "/fields/"):
        os.makedirs(cfg.env.this_runs_folder + "/fields/")
    if not os.path.exists("./test_output/"):
        os.makedirs("./test_output/")

    metrics_file = os.path.join("./test_output/" + cfg.name + "____metrics.csv")
    avg_PSNR = 0
    avg_PSNR_trilinear = 0
    avg_pix = 0
    avg_pix_trilinear = 0

    if cfg.is_use:
        for j, (LR, HR, Z, filenames) in enumerate(dataloader_test):
            status_logger.info(f"batch {j}")
            for i in range(len(LR)):
                status_logger.info(f"field {i}")
                indx = torch.tensor([i])
                LR_i = torch.index_select(LR, 0, indx, out=None)
                SR_i = gan.G(LR_i.to(cfg.device, non_blocking=True)).cpu()
                write_fields(
                    LR_i, None, SR_i, cfg.env.this_runs_folder + "/fields/", filenames[i], cfg.scale
                )

    if cfg.is_test:
        bar = displaybar.DisplayBar(
            max_value=len(dataloader_test),
            start_epoch=0,
            start_it=0,
            niter=niter,
        )
        with open(metrics_file, "w") as write_file:
            write_file.write("field,PSNR,pix,PSNR_trilinear, pix_trilinear\n")
            
            for epoch in range(num_epochs):
                dataloader_test.dataset.slice_index = epoch

                for j, (LR, HR, Z, filenames) in enumerate(dataloader_test):
                    # status_logger.info(f"batch {j}")
                    # names = data["hr_name"]
                    bar.update(j, epoch, len(dataloader_test)*(epoch)+j)

                    for i in range(LR.shape[0]):
                        # status_logger.info(f"field {i}")
                        indx = torch.as_tensor([i])
                        LR_i = torch.index_select(LR, 0, indx, out=None)
                        HR_i = torch.index_select(HR, 0, indx, out=None)
                        with torch.no_grad():
                            SR_i = gan.G(LR_i.to(cfg.device, non_blocking=True), Z.to(cfg.device, non_blocking=True)).cpu()
                        
                        interpolated_LR = nn.functional.interpolate(
                            LR[:, :3, :, :, :],
                            scale_factor=(4, 4, 1),
                            mode="trilinear",
                            align_corners=True,
                        )
                        PSNR, PSNR_trilinear, pix, trilinear_pix = write_metrics(HR_i[:,:3], SR_i, interpolated_LR, filenames[i], write_file)
                        
                        avg_PSNR += PSNR/niter
                        avg_PSNR_trilinear += PSNR_trilinear/niter
                        avg_pix += pix/niter
                        avg_pix_trilinear += trilinear_pix/niter
                        
                        if (len(dataloader_test)*(epoch)+j) % cfg.training.log_period == 0:
                            write_fields(
                                LR_i[:,:3], HR_i[:,:3], SR_i, interpolated_LR, Z, cfg.env.this_runs_folder, filenames[i]
                            )

        print(f"Average PSNR: {avg_PSNR}")
        print(f"Average PSNR trilinear: {avg_PSNR_trilinear}")
        print(f"Average pix: {avg_pix}")
        print(f"Average pix trilinear: {avg_pix_trilinear}")
       
def write_fields(
    LR: torch.Tensor,
    HR: torch.Tensor,
    SR: torch.Tensor,
    interpolated_LR: torch.Tensor,
    Z: torch.Tensor,
    folder_path: str,
    field_name: int,
) -> dict:
    
    fields = dict()
    fields["HR"] = HR.squeeze().numpy()
    fields["SR"] = SR.squeeze().numpy()
    fields["TL"] = interpolated_LR.squeeze().numpy()
    fields["LR"] = LR.squeeze().numpy()
    fields["Z"] = Z.squeeze().numpy()

    with open(folder_path + "/fields/test_fields_" + str(field_name) + ".pkl", "wb") as f:
        pkl.dump(fields, f)

def write_metrics(HR, SR, trilinear, field_name: int, dest_file):
    PSNR = calculate_PSNR(HR, SR)
    PSNR_trilinear = calculate_PSNR(HR, trilinear)
    pix_criterion = nn.L1Loss()
    pix = pix_criterion(HR, SR)
    trilinear_pix = pix_criterion(HR, trilinear)
    dest_file.write(f"{field_name},{PSNR},{pix},{PSNR_trilinear},{trilinear_pix}" + "\n")

    return PSNR, PSNR_trilinear, pix, trilinear_pix

# (1.0, 0.25, 1.25, 1.25, 0.15|10.0, 0.25, 2.5, 0.25, 0.25|10.0, 1.0, 0.5, 1.0, 0.2|22.627515146672238|25.805853036899766|30.63578914390694|31.348732309433014)
