"""
esrdgan.py
based on Eirik Vesterkjær, 2019, modified by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements the ESRDGAN inspired 3D GAN for super resolving wind fields
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim

import config.config as config
from GAN_models.baseGAN import BaseGAN
from CNN_models.Discriminator_3D import Discriminator_3D
from CNN_models.Generator_3D_Resnet_ESRGAN import Generator_3D
import tools.initialization as initialization
import tools.trainingtricks as trainingtricks
from process_data import calculate_gradient_of_wind_field


class wind_field_GAN_3D(BaseGAN):
    # feature extractor. Generator and discriminator are defined in BaseGAN
    # F: nn.Module = None

    def __init__(self, cfg: config.Config):
        super(wind_field_GAN_3D, self).__init__(cfg)  # BaseGAN
        self.optimizers = []
        self.schedulers = []
        self.train_G_loss_dict = {
            "total": torch.zeros(1),
            "adversarial": torch.zeros(1),
            "pix": torch.zeros(1),
            "xy_gradient": torch.zeros(1),
            "z_gradient": torch.zeros(1),
            "divergence": torch.zeros(1),
            "xy_divergence": torch.zeros(1),
            "feature_D": torch.zeros(1),
        }
        self.D_loss_dict = {
            "train_loss": torch.zeros(1),
            "validation_loss": torch.zeros(1),
        }
        self.validation_G_loss_dict = {
            "total": torch.zeros(1),
            "adversarial": torch.zeros(1),
            "pix": torch.zeros(1),
            "xy_gradient": torch.zeros(1),
            "z_gradient": torch.zeros(1),
            "divergence": torch.zeros(1),
            "xy_divergence": torch.zeros(1),
            "feature_D": torch.zeros(1),
        }
        self.hist_dict = {
            "val_grad_G_first_layer": torch.zeros(1),
            "val_grad_G_last_layer": torch.zeros(1),
            "val_grad_D_first_layer": torch.tensor(-1.0),
            "val_grad_D_last_layer": torch.tensor(-1.0),
            "val_weight_G_first_layer": torch.zeros(1),
            "val_weight_G_last_layer": torch.zeros(1),
            "val_weight_D_first_layer": torch.tensor(-1.0),
            "val_weight_D_last_layer": torch.tensor(-1.0),
            "SR_pix_distribution": torch.zeros(1),
            "D_pred_HR": torch.zeros(1),
            "D_pred_SR": torch.zeros(1),
        }

        self.metrics_dict = {
            "val_PSNR": torch.zeros(1),
            "Trilinear_PSNR": torch.zeros(1),
            "pix_loss_unscaled": torch.zeros(1),
            "trilinear_pix_loss": torch.zeros(1),
        }
        self.device_check = ""
        self.batch_size: int = 1
        self.max_diff_squared = torch.tensor(4.0, device=cfg.device)  # HR is in [-1, 1]
        self.epsilon_PSNR = torch.tensor(
            1e-8, device=cfg.device
        )  # PSNR is usually ~< 50 so this should not impact the result much

        self.feature_extractor = None

        ###################
        # Define generator, discriminator
        ###################
        cfg_G: config.GeneratorConfig = cfg.generator
        cfg_gan: config.GANConfig = cfg.gan_config
        self.G = Generator_3D(
            cfg_G.in_num_ch
            + cfg_gan.include_pressure
            + cfg_gan.include_z_channel
            + cfg_gan.include_above_ground_channel,
            cfg_G.out_num_ch,
            cfg_G.num_features,
            cfg_G.num_RRDB,
            upscale=cfg.scale,
            hr_kern_size=cfg_G.hr_kern_size,
            number_of_RDB_convs=cfg_G.num_RDB_convs,
            RDB_gc=cfg_G.RDB_growth_chan,
            lff_kern_size=cfg_G.lff_kern_size,
            RDB_residual_scaling=cfg_G.RDB_res_scaling,
            RRDB_residual_scaling=cfg_G.RRDB_res_scaling,
            act_type=cfg_G.act_type,
            device=self.device,
            number_of_z_layers=cfg_gan.number_of_z_layers,
            conv_mode=cfg_gan.conv_mode,
            use_mixed_precision=cfg_G.use_mixed_precision,
            terrain_number_of_features=cfg_G.terrain_number_of_features,
            dropout_probability=cfg_G.dropout_probability,
            max_norm=cfg_G.max_norm,
        ).to(self.device, non_blocking=True)

        initialization.init_weights(self.G, scale=cfg_G.weight_init_scale)

        self.conv_mode = cfg_G.conv_mode
        self.use_D_feature_extractor_cost = cfg_gan.use_D_feature_extractor_cost

        if cfg.is_train:
            cfg_D: config.DiscriminatorConfig = cfg.discriminator
            self.D = Discriminator_3D(
                cfg_D.in_num_ch,
                cfg_D.num_features,
                feat_kern_size=cfg_D.feat_kern_size,
                normalization_type=cfg_D.norm_type,
                act_type=cfg_D.act_type,
                mode=cfg_D.layer_mode,
                device=self.device,
                number_of_z_layers=cfg_gan.number_of_z_layers,
                conv_mode=cfg_gan.conv_mode,
                use_mixed_precision=cfg_D.use_mixed_precision,
                enable_slicing=cfg_gan.enable_slicing,
                dropout_probability=cfg_D.dropout_probability,
            ).to(self.device, non_blocking=True)

            # move to CUDA if available
            # self.HR_labels = self.HR_labels.to(cfg.device, non_blocking=True)
            # self.fake_HR_labels = self.fake_HR_labels.to(cfg.device, non_blocking=True)

            initialization.init_weights(self.D, scale=cfg_D.weight_init_scale)

        ###################
        # Define optimizers, schedulers, and losses
        ###################

        if cfg.is_train:
            cfg_t: config.TrainingConfig = cfg.training
            self.optimizer_G = torch.optim.Adam(
                self.G.parameters(),
                lr=cfg_t.learning_rate_g,
                weight_decay=cfg_t.adam_weight_decay_g,
                betas=(cfg_t.adam_beta1_g, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                self.D.parameters(),
                lr=cfg_t.learning_rate_d,
                weight_decay=cfg_t.adam_weight_decay_d,
                betas=(cfg_t.adam_beta1_d, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if cfg_t.multistep_lr_steps:
                self.scheduler_G = lr_scheduler.MultiStepLR(
                    self.optimizer_G, cfg_t.multistep_lr_steps, gamma=cfg_t.lr_gamma
                )
                self.scheduler_D = lr_scheduler.MultiStepLR(
                    self.optimizer_D, cfg_t.multistep_lr_steps, gamma=cfg_t.lr_gamma
                )
                self.schedulers.append(self.scheduler_G)
                self.schedulers.append(self.scheduler_D)

            # pixel loss
            self.gradient_xy_criterion = nn.MSELoss().to(cfg.device, non_blocking=True)
            self.gradient_z_criterion = nn.MSELoss().to(cfg.device, non_blocking=True)
            self.divergence_criterion = nn.MSELoss().to(cfg.device, non_blocking=True)
            self.xy_divergence_criterion = nn.MSELoss().to(
                cfg.device, non_blocking=True
            )
            self.feature_D_criterion = nn.MSELoss().to(cfg.device, non_blocking=True)

            if cfg_t.pixel_criterion is None or cfg_t.pixel_criterion == "none":
                self.pixel_criterion = None
            elif cfg_t.pixel_criterion == "l1":
                self.pixel_criterion = nn.L1Loss().to(cfg.device, non_blocking=True)
            elif cfg_t.pixel_criterion == "l2":
                self.pixel_criterion = nn.MSELoss().to(cfg.device, non_blocking=True)
            else:
                raise NotImplementedError(
                    f"Only l1 and l2 (MSE) loss have been implemented for pixel loss, not {cfg_t.pixel_criterion}"
                )

            # GAN adversarial loss
            if cfg_t.gan_type == "relativistic" or cfg_t.gan_type == "relativisticavg":
                self.criterion = nn.BCEWithLogitsLoss().to(
                    cfg.device, non_blocking=True
                )
            else:
                raise NotImplementedError(
                    f"Only relativistic and relativisticavg GAN are implemented, not {cfg_t.gan_type}"
                )
            return

    def feed_xy_niter(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        niter: torch.Tensor,
        d_g_train_ratio: int,
        d_g_train_period: int,
    ):
        self.x = x
        self.y = y
        self.niter = niter
        self.d_g_train_ratio = d_g_train_ratio
        self.d_g_train_period = d_g_train_period

    def D_forward(
        self,
        HR: torch.Tensor,
        fake_HR: torch.Tensor,
        it: torch.Tensor,
        train_D: bool,
    ):
        if self.device_check == "":
            self.device_check += (
                str(HR.device)
                + str(fake_HR.device)
                + str(fake_HR.detach().device)
                + str(it.device)
                + str(self.niter.device)
                + str(next(self.D.parameters()).device)
                + str(
                    trainingtricks.instance_noise(
                        torch.tensor(2.0, device=self.device),
                        HR.size(),
                        it,
                        self.niter,
                        device=self.device,
                    ).device
                )
                + str(self.device)
            )
        if train_D:
            self.D.train()
            if self.cfg.training.use_instance_noise:
                y_pred = self.D(
                    HR
                    + trainingtricks.instance_noise(
                        torch.tensor(1.0, device=self.device),
                        HR.size(),
                        it,
                        self.niter,
                        device=self.device,
                    )
                ).squeeze()
                fake_y_pred = self.D(
                    fake_HR.detach()
                    + trainingtricks.instance_noise(
                        torch.tensor(1.0, device=self.device),
                        HR.size(),
                        it,
                        self.niter,
                        device=self.device,
                    )
                ).squeeze()  # detach -> avoid BP to G
            else:
                y_pred = self.D(HR).squeeze()
                fake_y_pred = self.D(fake_HR.detach()).squeeze()
        else:
            self.D.eval()
            if self.cfg.training.use_instance_noise:
                y_pred = (
                    self.D(
                        HR
                        + trainingtricks.instance_noise(
                            torch.tensor(2.0, device=self.device),
                            HR.size(),
                            it,
                            self.niter,
                            device=self.device,
                        )
                    )
                    .squeeze()
                    .detach()
                )
                fake_y_pred = self.D(
                    fake_HR
                    + trainingtricks.instance_noise(
                        torch.tensor(2.0, device=self.device),
                        HR.size(),
                        it,
                        self.niter,
                        device=self.device,
                    )
                ).squeeze()
            else:
                y_pred = self.D(HR).squeeze().detach()
                fake_y_pred = self.D(fake_HR).squeeze()

        return y_pred, fake_y_pred

    def log_G_losses(
        self,
        fake_HR,
        loss_G,
        loss_G_adversarial,
        loss_G_pix,
        loss_G_xy_gradient,
        loss_G_z_gradient,
        loss_G_divergence,
        loss_G_xy_divergence,
        loss_G_feature_D,
        training_iteration: bool,
    ):
        if training_iteration:
            self.train_G_loss_dict["total"] = loss_G
            self.train_G_loss_dict["adversarial"] = loss_G_adversarial
            self.train_G_loss_dict["xy_gradient"] = loss_G_xy_gradient
            self.train_G_loss_dict["z_gradient"] = loss_G_z_gradient
            self.train_G_loss_dict["divergence"] = loss_G_divergence
            self.train_G_loss_dict["xy_divergence"] = loss_G_xy_divergence
            self.train_G_loss_dict["feature_D"] = loss_G_feature_D
            self.train_G_loss_dict["pix"] = loss_G_pix
        else:
            self.validation_G_loss_dict["total"] = loss_G
            self.validation_G_loss_dict["adversarial"] = loss_G_adversarial
            self.validation_G_loss_dict["xy_gradient"] = loss_G_xy_gradient
            self.validation_G_loss_dict["z_gradient"] = loss_G_z_gradient
            self.validation_G_loss_dict["divergence"] = loss_G_divergence
            self.validation_G_loss_dict["xy_divergence"] = loss_G_xy_divergence
            self.validation_G_loss_dict["feature_D"] = loss_G_feature_D
            self.validation_G_loss_dict["pix"] = loss_G_pix
            self.metrics_dict["pix_loss_unscaled"] = (
                loss_G_pix / self.cfg.training.pixel_loss_weight
            )
            self.hist_dict["SR_pix_distribution"] = fake_HR.detach().cpu().numpy()

    def calculate_optimize_and_log_G_loss(
        self,
        HR,
        fake_HR,
        Z,
        y_pred,
        fake_y_pred,
        training_iteration: bool,
    ):
        loss_G_adversarial = 0

        if self.cfg.training.gan_type == "dcgan":
            loss_G_adversarial = self.criterion(
                fake_y_pred, self.HR_labels
            ) + self.criterion(y_pred, self.fake_HR_labels)
        if self.cfg.training.gan_type == "relativistic":
            loss_G_adversarial = self.criterion(fake_y_pred - y_pred, self.HR_labels)

        elif self.cfg.training.gan_type == "relativisticavg":
            loss_G_adversarial = (
                self.criterion(fake_y_pred - torch.mean(y_pred), self.HR_labels)
                + self.criterion(y_pred - torch.mean(fake_y_pred), self.fake_HR_labels)
            ) / 2.0
        else:
            raise NotImplementedError(
                f"Only relativistic and relativisticavg GAN are implemented, not {self.cfg.training.gan_type}"
            )

        loss_G_feature_D = torch.zeros(1, device=self.device)

        if self.feature_extractor is not None:
            features = self.feature_extractor(HR).detach()
            fake_features = self.feature_extractor(fake_HR)
            loss_G_feature_D = self.feature_D_criterion(features, fake_features)

        loss_G_pix = torch.zeros(1, device=self.device)
        if self.pixel_criterion:
            loss_G_pix = self.pixel_criterion(HR, fake_HR)

        HR_wind_gradient = calculate_gradient_of_wind_field(
            HR[:, :3], self.x, self.y, Z
        )
        SR_wind_gradient = calculate_gradient_of_wind_field(
            fake_HR[:, :3], self.x, self.y, Z
        )

        (
            max_xy_gradient,
            max_z_gradient,
            max_divergence,
            max_xy_divergence,
        ) = get_norm_factors_of_gradients(HR_wind_gradient, SR_wind_gradient)

        loss_G_xy_gradient = self.gradient_xy_criterion(
            SR_wind_gradient[:, :6] / max_xy_gradient,
            HR_wind_gradient[:, :6] / max_xy_gradient,
        )
        loss_G_z_gradient = self.gradient_z_criterion(
            SR_wind_gradient[:, 6:] / max_z_gradient,
            HR_wind_gradient[:, 6:] / max_z_gradient,
        )

        loss_G_divergence = self.divergence_criterion(
            (
                HR_wind_gradient[:, 0, :, :, :]
                + HR_wind_gradient[:, 4, :, :, :]
                + HR_wind_gradient[:, 8, :, :, :]
            )
            / max_divergence,
            (
                SR_wind_gradient[:, 0, :, :, :]
                + SR_wind_gradient[:, 4, :, :, :]
                + SR_wind_gradient[:, 8, :, :, :]
            )
            / max_divergence,
        )

        loss_G_xy_divergence = self.xy_divergence_criterion(
            (HR_wind_gradient[:, 0, :, :, :] + HR_wind_gradient[:, 4, :, :, :])
            / max_xy_divergence,
            (SR_wind_gradient[:, 0, :, :, :] + SR_wind_gradient[:, 4, :, :, :])
            / max_xy_divergence,
        )

        loss_G_adversarial *= self.cfg.training.adversarial_loss_weight
        loss_G_feature_D *= self.cfg.training.feature_D_loss_weight
        loss_G_pix *= self.cfg.training.pixel_loss_weight
        loss_G_xy_gradient *= self.cfg.training.gradient_xy_loss_weight
        loss_G_z_gradient *= self.cfg.training.gradient_z_loss_weight
        loss_G_divergence *= self.cfg.training.divergence_loss_weight
        loss_G_xy_divergence *= self.cfg.training.xy_divergence_loss_weight

        if (
            loss_G_divergence.isnan()
            or loss_G_xy_divergence.isnan()
            or loss_G_z_gradient.isnan()
            or loss_G_xy_gradient.isnan()
            or loss_G_divergence.isinf()
            or loss_G_xy_divergence.isinf()
            or loss_G_z_gradient.isinf()
            or loss_G_xy_gradient.isinf()
        ):
            loss_G = loss_G_adversarial + loss_G_pix + loss_G_feature_D
        else:
            loss_G = (
                loss_G_adversarial
                + loss_G_pix
                + loss_G_xy_gradient
                + loss_G_z_gradient
                + loss_G_divergence
                + loss_G_xy_divergence
                + loss_G_feature_D
            )
        if training_iteration:
            loss_G.backward()
            if not (loss_G.isnan() or loss_G.isinf()):
                total_norm = 0
                # torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.G.max_norm)
                self.optimizer_G.step()

        self.log_G_losses(
            fake_HR,
            loss_G,
            loss_G_adversarial,
            loss_G_pix,
            loss_G_xy_gradient,
            loss_G_z_gradient,
            loss_G_divergence,
            loss_G_xy_divergence,
            loss_G_feature_D,
            training_iteration=training_iteration,
        )
        return loss_G

    def update_G(self, LR, HR, Z, it, training_iteration: bool):
        if training_iteration:
            self.G.train()
            fake_HR = self.G(LR, Z)

            for param in self.D.parameters():
                param.requires_grad = False

            self.G.zero_grad(set_to_none=True)

            # with torch.autocast(self.device.type, enabled=self.cfg.discriminator.use_mixed_precision):
            y_pred, fake_y_pred = self.D_forward(HR, fake_HR, it, train_D=False)
            self.calculate_optimize_and_log_G_loss(
                HR, fake_HR, Z, y_pred, fake_y_pred, training_iteration
            )

        else:
            self.G.eval()
            with torch.no_grad():
                fake_HR = self.G(LR, Z)
                y_pred, fake_y_pred = self.D_forward(HR, fake_HR, it, train_D=False)
                self.calculate_optimize_and_log_G_loss(
                    HR, fake_HR, Z, y_pred, fake_y_pred, training_iteration
                )
        return fake_HR

    def log_D_losses(self, loss_D, y_pred, fake_y_pred, training_epoch):
        if training_epoch:
            self.D_loss_dict["train_loss"] = loss_D
            # BCEWithLogitsLoss has sigmoid activation.
        else:
            # if self.conv_mode == "horizontal_3D":
            #     grad_start = (
            #         self.D.features[0][0].convs[0][0].weight.grad.detach().cpu()
            #     )
            #     weight_start = self.D.features[0][0].convs[0][0].weight.detach().cpu()
            #     grad_end = self.D.classifier[-1].weight.grad.detach().cpu()
            #     weight_end = self.D.classifier[-1].weight.detach().cpu()
            # else:
            #     grad_start = self.D.features[0][0][0].weight.grad.detach().cpu()
            #     weight_start = self.D.features[0][0][0].weight.detach().cpu()
            #     grad_end = self.D.classifier[-1].weight.grad.detach().cpu()
            #     weight_end = self.D.classifier[-1].weight.detach().cpu()

            # self.hist_dict["val_grad_D_first_layer"] = grad_start.numpy()
            # self.hist_dict["val_grad_D_last_layer"] = grad_end.numpy()
            # self.hist_dict["val_weight_D_first_layer"] = weight_start.numpy()
            # self.hist_dict["val_weight_D_last_layer"] = weight_end.numpy()
            self.D_loss_dict["validation_loss"] = loss_D
            self.hist_dict["D_pred_HR"] = (
                torch.sigmoid(y_pred.detach()).cpu().numpy()[np.newaxis]
            )
            self.hist_dict["D_pred_SR"] = (
                torch.sigmoid(fake_y_pred.detach()).cpu().numpy()[np.newaxis]
            )

    def update_D(
        self, HR: torch.Tensor, fake_HR: torch.Tensor, it, training_epoch: bool
    ):
        if training_epoch:
            for param in self.D.parameters():
                param.requires_grad = True
                self.optimizer_D.zero_grad(set_to_none=True)

            y_pred, fake_y_pred = self.D_forward(HR, fake_HR, it, train_D=True)
        else:
            with torch.no_grad():
                y_pred, fake_y_pred = self.D_forward(HR, fake_HR, it, train_D=True)

        loss_D = None
        if self.cfg.training.gan_type == "dcgan":
            loss_D = self.criterion(y_pred, self.HR_labels) + self.criterion(
                fake_y_pred, self.fake_HR_labels
            )
        if self.cfg.training.gan_type == "relativistic":
            loss_D = self.criterion(y_pred - fake_y_pred, self.HR_labels)
        elif self.cfg.training.gan_type == "relativisticavg":
            loss_D = (
                self.criterion(y_pred - torch.mean(fake_y_pred), self.HR_labels)
                + self.criterion(fake_y_pred - torch.mean(y_pred), self.fake_HR_labels)
            ) / 2.0
            if torch.all(self.HR_labels == 0.9):
                loss_D -= 0.1985
        else:
            raise NotImplementedError(
                f"Only relativistic and relativisticavg GAN are implemented, not {self.cfg.training.gan_type}"
            )

        if training_epoch:
            loss_D.backward()
            self.optimizer_D.step()

        self.log_D_losses(loss_D, y_pred, fake_y_pred, training_epoch=training_epoch)

    def compute_losses_and_optimize(
        self, LR, HR, Z, it, training_iteration: bool = False
    ):
        self.batch_size = HR.size(0)
        it = torch.tensor(it, device=self.device)
        self.make_new_labels(it)

        if (
            it % self.cfg.training.feature_D_update_period == 0
            and self.use_D_feature_extractor_cost
        ):
            self.feature_extractor = copy.deepcopy(self.D.features)
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        train_period = it // self.d_g_train_period
        if training_iteration:
            if train_period % (self.d_g_train_ratio + 1) == 0:
                fake_HR = self.update_G(LR, HR, Z, it, training_iteration)
            else:
                with torch.no_grad():
                    self.G.eval()
                    fake_HR = self.G(LR, Z)
                self.update_D(HR, fake_HR, it, training_iteration)
        else:
            fake_HR = self.update_G(LR, HR, Z, it, training_iteration)
            self.update_D(HR, fake_HR, it, training_iteration)
            (
                self.metrics_dict["val_PSNR"],
                self.metrics_dict["Trilinear_PSNR"],
            ) = compute_PSNR_for_SR_and_trilinear(
                LR,
                HR,
                fake_HR,
                self.max_diff_squared,
                self.epsilon_PSNR,
                interpolate=True,
                device=self.device,
                scale=self.cfg.scale,
            )
            self.metrics_dict["trilinear_pix_loss"] = self.pixel_criterion(
                HR,
                nn.functional.interpolate(
                    LR[:, :3, :, :, :],
                    scale_factor=(self.cfg.scale, self.cfg.scale, 1),
                    mode="trilinear",
                    align_corners=True,
                ),
            )
        return

    def optimize_parameters(self, LR, HR, Z, it):
        self.compute_losses_and_optimize(LR, HR, Z, it, training_iteration=True)

    def validation(self, LR, HR, Z, it):
        self.compute_losses_and_optimize(LR, HR, Z, it, training_iteration=False)

    def make_new_labels(self, it):
        pred_real = True
        pred_fake = False

        if self.cfg.training.flip_labels:
            pred_real = False
            pred_fake = True

        real_label = torch.tensor(1.0, device=self.device)
        fake_label = torch.tensor(0.0, device=self.device)
        if (
            self.cfg.training.use_one_sided_label_smoothing
            and self.cfg.training.flip_labels
        ):
            real_label = torch.tensor(1.0, device=self.device)
            fake_label = torch.tensor(0.1, device=self.device) - 0.1 * it / self.niter
        elif self.cfg.training.use_one_sided_label_smoothing:
            real_label = torch.tensor(0.9, device=self.device) + 0.1 * it / self.niter
            fake_label = torch.tensor(0.0, device=self.device)

        if self.cfg.training.use_noisy_labels:
            self.HR_labels = trainingtricks.noisy_labels(
                pred_real,
                self.batch_size,
                true_label_val=real_label,
                false_label_val=fake_label,
                device=self.device,
            ).squeeze()
            self.fake_HR_labels = trainingtricks.noisy_labels(
                pred_fake,
                self.batch_size,
                true_label_val=real_label,
                false_label_val=fake_label,
                device=self.device,
            ).squeeze()
        else:  # no noise std dev -> no noise
            self.HR_labels = trainingtricks.noisy_labels(
                pred_real,
                self.batch_size,
                noise_stddev=0.0,
                true_label_val=real_label,
                false_label_val=fake_label,
                device=self.device,
            ).squeeze()
            self.fake_HR_labels = trainingtricks.noisy_labels(
                pred_fake,
                self.batch_size,
                noise_stddev=0.0,
                true_label_val=real_label,
                false_label_val=fake_label,
                device=self.device,
            ).squeeze()

    def get_G_train_loss_dict_ref(self):
        return self.train_G_loss_dict

    def get_G_val_loss_dict_ref(self):
        return self.validation_G_loss_dict

    def get_D_loss_dict_ref(self):
        return self.D_loss_dict

    def get_hist_dict_ref(self):
        return self.hist_dict

    def get_metrics_dict_ref(self):
        return self.metrics_dict

    def update_learning_rate(self):
        for s in self.schedulers:
            s.step()

    def count_params(self) -> tuple[int, int]:
        """
        count_params returns the number of parameter in the G, D, and F of the GAN (in that order)
        """
        G_params = sum(par.numel() for par in self.G.parameters())
        D_params = sum(par.numel() for par in self.D.parameters())
        # F_params = sum(par.numel() for par in self.F.parameters())
        return G_params, D_params

    def count_trainable_params(self) -> tuple[int, int]:
        G_params = sum(par.numel() for par in self.G.parameters() if par.requires_grad)
        D_params = sum(par.numel() for par in self.D.parameters() if par.requires_grad)
        # F_params = sum(par.numel() for par in self.F.parameters() if par.requires_grad)
        return G_params, D_params

    def __str__(self):
        G_params, D_params = self.count_params()
        G_params_t, D_params_t = self.count_trainable_params()
        return (
            f"*---------------*\nGenerator:\n{G_params} params, {G_params_t} trainable\n\n"
            + str(self.G)
            + "\n\n"
            + f"*---------------*\nDiscriminator:\n{D_params} params, {D_params_t} trainable\n\n"
            + str(self.D)
            # + "\n\n"
            # + f"*---------------*\nFeature Extractor (Perceptual network):\n{F_params} params, {F_params_t} trainable\n\n"
            # + str(self.F)
            + "\n"
        )


def calculate_PSNR(
    HR: torch.Tensor,
    fake_HR: torch.Tensor,
    max_diff_squared=torch.tensor(4.0),
    epsilon_PSNR=torch.tensor(1e-8),
    device=torch.device("cpu"),
):
    w, h, l = HR.shape[2], HR.shape[3], HR.shape[4]
    SR_batch_average_MSE = torch.sum((HR - fake_HR) ** 2) / (w * h * l * HR.shape[0])

    return torch.tensor(10, device=device) * math.log10(
        max_diff_squared / (SR_batch_average_MSE + epsilon_PSNR)
    )


def compute_PSNR_for_SR_and_trilinear(
    LR,
    HR: torch.Tensor,
    fake_HR: torch.Tensor,
    max_diff_squared,
    epsilon_PSNR,
    interpolate: bool = False,
    device=torch.device("cpu"),
    scale: int = 4,
):
    val_PSNR = calculate_PSNR(
        HR, fake_HR, max_diff_squared, epsilon_PSNR, device=device
    )
    if interpolate:
        interpolated_LR = nn.functional.interpolate(
            LR[:, :3, :, :, :],
            scale_factor=(scale, scale, 1),
            mode="trilinear",
            align_corners=True,
        )
        val_trilinear_PSNR = calculate_PSNR(
            HR, interpolated_LR, max_diff_squared, epsilon_PSNR, device=device
        )
        return val_PSNR, val_trilinear_PSNR
    else:
        return val_PSNR


@torch.jit.script
def get_norm_factors_of_gradients(
    HR_wind_gradient: torch.Tensor, SR_wind_gradient: torch.Tensor
):
    max_HR_xy_gradient = torch.max(torch.abs(HR_wind_gradient[:, :6]))
    max_SR_xy_gradient = torch.max(torch.abs(SR_wind_gradient[:, :6]))

    max_HR_z_gradient = torch.max(HR_wind_gradient[:, 6:])
    max_SR_z_gradient = torch.max(SR_wind_gradient[:, 6:])

    max_HR_divergence = torch.max(
        torch.abs(
            HR_wind_gradient[:, 0, :, :, :]
            + HR_wind_gradient[:, 4, :, :, :]
            + HR_wind_gradient[:, 8, :, :, :]
        )
    )
    max_SR_divergence = torch.max(
        torch.abs(
            SR_wind_gradient[:, 0, :, :, :]
            + SR_wind_gradient[:, 4, :, :, :]
            + SR_wind_gradient[:, 8, :, :, :]
        )
    )

    max_HR_xy_divergence = torch.max(
        torch.abs((HR_wind_gradient[:, 0, :, :, :] + HR_wind_gradient[:, 4, :, :, :]))
    )

    max_SR_xy_divergence = torch.max(
        torch.abs((SR_wind_gradient[:, 0, :, :, :] + SR_wind_gradient[:, 4, :, :, :]))
    )

    return [
        torch.max(HR_max, SR_max / 100)
        for (HR_max, SR_max) in [
            (max_HR_xy_gradient, max_SR_xy_gradient),
            (max_HR_z_gradient, max_SR_z_gradient),
            (max_HR_divergence, max_SR_divergence),
            (max_HR_xy_divergence, max_SR_xy_divergence),
        ]
    ]
