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
import torch
import torch.cuda
import torch.nn as nn
import config.config as config
from GAN_models.wind_field_GAN_3D import wind_field_GAN_3D, calculate_PSNR
import iocomponents.displaybar as displaybar
from download_data import reverse_interpolate_z_axis

# UVW_MAX = 32.32976150


def test(cfg: config.Config, dataset_test):
    status_logger = logging.getLogger("status")
    niter = len(dataset_test)
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

    gan.G.eval()

    status_logger.info(f"beginning test")

    UVW_MAX = dataset_test.UVW_MAX

    if not os.path.exists(cfg.env.this_runs_folder + "/fields/"):
        os.makedirs(cfg.env.this_runs_folder + "/fields/")
    if not os.path.exists("./test_output/"):
        os.makedirs("./test_output/")
    if not os.path.exists("./test_output/averages.csv"):
        with open("./test_output/averages.csv", "w") as f:
            f.write(
                "Name, Average PSNR, Average PSNR trilinear, Average pix, Average pix trilinear\n"
            )
    if cfg.gan_config.interpolate_z:
        if not os.path.exists("./test_output/averages_reverse_interpolate.csv"):
            with open("./test_output/averages_reverse_interpolate.csv", "w") as f:
                f.write(
                    "Name, PSNR, PSNR_trilinear, relative_error, pix, trilinear_pix, relative_error_trilinear\n"
                )

    metrics_file = os.path.join("./test_output/" + cfg.name + "____metrics.csv")
    avg_PSNR = 0
    avg_PSNR_trilinear = 0
    avg_pix = 0
    avg_pix_trilinear = 0
    avg_relative_error = 0
    avg_relative_error_trilinear = 0
    avg_wind_speed = 0

    if cfg.gan_config.interpolate_z:
        reverse_interpolate_metrics_file = os.path.join(
            "./test_output/" + cfg.name + "____metrics_reverse_interpolate.csv"
        )
        open(reverse_interpolate_metrics_file, "a").write("field, PSNR, PSNR_trilinear, relative_error, pix, trilinear_pix, relative_error_trilinear, average_wind_speed\n")
        avg_PSNR_reverse_interpolate = 0
        avg_PSNR_trilinear_reverse_interpolate = 0
        avg_pix_reverse_interpolate = 0
        avg_pix_trilinear_reverse_interpolate = 0
        avg_relative_error_reverse_interpolate = 0
        avg_relative_error_trilinear_reverse_interpolate = 0

    if cfg.is_use:
        for j, (LR, HR, Z, filenames, _, _) in enumerate(dataloader_test):
            status_logger.info(f"batch {j}")
            for i in range(len(LR)):
                status_logger.info(f"field {i}")
                indx = torch.tensor([i])
                LR_i = torch.index_select(LR, 0, indx, out=None)
                SR_i = gan.G(LR_i.to(cfg.device, non_blocking=True)).cpu()
                write_fields(
                    LR_i,
                    None,
                    SR_i,
                    cfg.env.this_runs_folder + "/fields/",
                    filenames[i],
                    cfg.scale,
                )

    if cfg.is_test:
        bar = displaybar.DisplayBar(
            max_value=len(dataloader_test),
            start_epoch=0,
            start_it=0,
            niter=niter,
        )
        with open(metrics_file, "w") as write_file:
            write_file.write("field, PSNR, PSNR_trilinear, relative_error, pix, trilinear_pix, relative_error_trilinear, average wind speed\n")
            for j, (LR, HR, Z, filenames, HR_raw, Z_raw) in enumerate(dataloader_test):
                # status_logger.info(f"batch {j}")
                # names = data["hr_name"]
                bar.update(j, 0, len(dataloader_test) * (0) + j)
                interpolated_LR = nn.functional.interpolate(
                    LR[:, :3, :, :, :],
                    scale_factor=(cfg.scale, cfg.scale, 1),
                    align_corners=True,
                    mode="trilinear",
                )

                for i in range(LR.shape[0]):
                    LR_i = torch.index_select(LR, 0, torch.as_tensor([i]), out=None)
                    interpolated_LR_i = torch.index_select(
                        interpolated_LR, 0, torch.as_tensor([i]), out=None
                    )
                    HR_i = torch.index_select(HR, 0, torch.as_tensor([i]), out=None)

                    with torch.no_grad():
                        SR_i = gan.G(
                            LR_i.to(cfg.device, non_blocking=True),
                            torch.index_select(Z, 0, torch.as_tensor([i]), out=None).to(
                                cfg.device, non_blocking=True
                            ),
                        ).cpu()

                    if cfg.gan_config.interpolate_z:
                        reverse_SR_i = reverse_interpolate_z_axis(
                            SR_i,
                            torch.index_select(
                                Z_raw, 0, torch.as_tensor([i]), out=None
                            ),
                            torch.index_select(Z, 0, torch.as_tensor([i]), out=None),
                        )
                        reverse_TL_i = reverse_interpolate_z_axis(interpolated_LR_i, torch.index_select(Z_raw, 0, torch.as_tensor([i]), out=None), torch.index_select(Z, 0, torch.as_tensor([i]), out=None))

                        (
                            reverse_PSNR, reverse_PSNR_trilinear, reverse_relative_error, reverse_pix, reverse_trilinear_pix, reverse_relative_error_trilinear, HR_mean_wind_speed
                        ) = write_metrics(
                            torch.index_select(
                                HR_raw, 0, torch.as_tensor([i]), out=None
                            )[:, :3],
                            reverse_SR_i,
                            reverse_TL_i,
                            filenames[i],
                            open(reverse_interpolate_metrics_file, "a"),
                            UVW_MAX,
                        )
                        avg_PSNR_reverse_interpolate += reverse_PSNR / niter
                        avg_PSNR_trilinear_reverse_interpolate += reverse_PSNR_trilinear/niter
                        avg_pix_reverse_interpolate += reverse_pix / niter
                        avg_pix_trilinear_reverse_interpolate += reverse_trilinear_pix/niter
                        avg_relative_error_reverse_interpolate += reverse_relative_error/niter
                        avg_relative_error_trilinear_reverse_interpolate += reverse_relative_error_trilinear/niter

                    PSNR, PSNR_trilinear, relative_error, pix, trilinear_pix, relative_error_trilinear, HR_mean_wind_speed = write_metrics(
                        HR_i[:, :3], SR_i, interpolated_LR_i, filenames[i], write_file, UVW_MAX
                    )

                    avg_PSNR += PSNR / niter
                    avg_PSNR_trilinear += PSNR_trilinear / niter
                    avg_pix += pix / niter
                    avg_pix_trilinear += trilinear_pix / niter
                    avg_relative_error += relative_error / niter
                    avg_relative_error_trilinear += relative_error_trilinear / niter
                    avg_wind_speed += HR_mean_wind_speed / niter

                    if j % cfg.training.log_period == 0:
                        write_fields(
                            LR[i],
                            HR[i],
                            SR_i[0],
                            interpolated_LR[i],
                            Z[i],
                            cfg.env.this_runs_folder,
                            filenames[i],
                            HR_raw[i] if cfg.gan_config.interpolate_z else torch.tensor([]),
                            Z_raw[i] if cfg.gan_config.interpolate_z else torch.tensor([]),
                            reverse_SR_i[0] if cfg.gan_config.interpolate_z else torch.tensor([]),
                        )
        with open("./test_output/averages.csv", "a") as f:
            f.write(
                f"{cfg.name},{avg_PSNR}, {avg_PSNR_trilinear}, {avg_pix}, {avg_pix_trilinear}\n"
            )

        print(f"Average PSNR: {avg_PSNR}")
        print(f"Average PSNR trilinear: {avg_PSNR_trilinear}")
        print(f"Average pix: {avg_pix}")
        print(f"Average pix trilinear: {avg_pix_trilinear}")
        print(f"Average relative error: {avg_relative_error}")
        print(f"Average relative error trilinear: {avg_relative_error_trilinear}")
        print(f"Average wind speed: {avg_wind_speed}")
        
        if cfg.gan_config.interpolate_z:
            with open("./test_output/averages_reverse_interpolate.csv", "a") as f:
                f.write(
                    f"{cfg.name},{avg_PSNR_reverse_interpolate}, {avg_relative_error_reverse_interpolate}, {avg_PSNR_trilinear_reverse_interpolate}, {avg_pix_reverse_interpolate}, {avg_pix_trilinear_reverse_interpolate}, {avg_relative_error_trilinear_reverse_interpolate} \n"
                )
            print(f"Average PSNR reverse interpolate: {avg_PSNR_reverse_interpolate}")
            print(
                f"Average PSNR trilinear reverse interpolate: {avg_PSNR_trilinear_reverse_interpolate}"
            )
            print(f"Average pix reverse interpolate: {avg_pix_reverse_interpolate}")
            print(
                f"Average pix trilinear reverse interpolate: {avg_pix_trilinear_reverse_interpolate}"
            )
            print(f"Average relative error reverse interpolate: {avg_relative_error_reverse_interpolate}")
            print(f"Average relative error trilinear reverse interpolate: {avg_relative_error_trilinear_reverse_interpolate}")


def write_fields(
    LR: torch.Tensor,
    HR: torch.Tensor,
    SR: torch.Tensor,
    interpolated_LR: torch.Tensor,
    Z: torch.Tensor,
    folder_path: str,
    field_name: int,
    rawHR: torch.Tensor([]),
    Z_raw: torch.Tensor([]),
    SR_orig: torch.Tensor([]),
) -> dict:
    fields = dict()
    fields["HR"] = HR.squeeze().numpy()
    fields["SR"] = SR.squeeze().numpy()
    fields["TL"] = interpolated_LR.squeeze().numpy()
    fields["LR"] = LR.squeeze().numpy()
    fields["Z"] = Z.squeeze().numpy()
    if rawHR.shape[0] > 0:
        fields["HR_orig"] = rawHR.squeeze().numpy()
        fields["Z_orig"] = Z_raw.squeeze().numpy()
        fields["SR_orig"] = SR_orig.squeeze().numpy()

    with open(
        folder_path + "/fields/test_fields_" + str(field_name) + ".pkl", "wb"
    ) as f:
        pkl.dump(fields, f)


def write_metrics(HR, SR, trilinear, field_name: int, dest_file, UVW_MAX):
    PSNR = calculate_PSNR(HR, SR)
    PSNR_trilinear = calculate_PSNR(HR, trilinear)
    ERR = HR - SR
    trilinear_ERR = HR - trilinear
    mean_ERR_length = torch.mean(torch.sqrt(ERR[:, 0] ** 2 + ERR[:, 1] ** 2 + ERR[:, 2] ** 2))
    mean_trilinear_ERR_length = torch.mean(torch.sqrt(trilinear_ERR[:, 0] ** 2 + trilinear_ERR[:, 1] ** 2 + trilinear_ERR[:, 2] ** 2))
    mean_HR_length = torch.mean(torch.sqrt(HR[:, 0] ** 2 + HR[:, 1] ** 2 + HR[:, 2] ** 2))
    pix = mean_ERR_length*UVW_MAX
    trilinear_pix = mean_trilinear_ERR_length*UVW_MAX
    relative_error = mean_trilinear_ERR_length/mean_HR_length
    relative_error_trilinear = mean_trilinear_ERR_length/mean_HR_length
    dest_file.write(
        f"{field_name},{PSNR},{PSNR_trilinear},{relative_error},{pix},{trilinear_pix}, {relative_error_trilinear}, {mean_HR_length*UVW_MAX}" + "\n"
    )
    
    return PSNR, PSNR_trilinear, relative_error, pix, trilinear_pix, relative_error_trilinear, mean_HR_length*UVW_MAX


# (1.0, 0.25, 1.25, 1.25, 0.15|10.0, 0.25, 2.5, 0.25, 0.25|10.0, 1.0, 0.5, 1.0, 0.2|22.627515146672238|25.805853036899766|30.63578914390694|31.348732309433014)
