"""
esrdgan.py
Based on Eirik Vesterkjær, 2019, modified by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Implements the ESRD GAN model
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim

import config.config as config
from GAN_models.baseGAN import BaseGAN
from CNN_models.Discriminator_3D import Discriminator_3D
from CNN_models.Generator_3D import Generator_3D
import tools.initialization as initialization
import tools.loggingclass as loggingclass
import tools.trainingtricks as trainingtricks
from process_data import calculate_gradient_of_wind_field


class wind_field_GAN_3D(BaseGAN):
    # feature extractor. Generator and discriminator are defined in BaseGAN
    # F: nn.Module = None

    def __init__(self, cfg: config.Config):
        super(wind_field_GAN_3D, self).__init__(cfg)  # BaseGAN
        self.optimizers = []
        self.schedulers = []
        self.memory_dict = {}
        self.runtime_dict = {}
        self.loss_dict = {
            "train_loss_D": torch.zeros(1),
            "train_loss_G": torch.zeros(1),
            "train_loss_G_adversarial": torch.zeros(1),
            "train_loss_G_pix": torch.zeros(1),
            "train_loss_G_xy_gradient": torch.zeros(1),
            "train_loss_G_z_gradient": torch.zeros(1),
            "train_loss_G_divergence": torch.zeros(1),
            "train_loss_G_xy_divergence": torch.zeros(1),
            "train_loss_G_feature_D": torch.zeros(1),
            "val_loss_G_xy_gradient": torch.zeros(1),
            "val_loss_G_z_gradient": torch.zeros(1),
            "val_loss_G_divergence": torch.zeros(1),
            "val_loss_G_xy_divergence": torch.zeros(1),
            "val_loss_G_feature_D": torch.zeros(1),
            "val_loss_D": torch.zeros(1),
            "val_loss_G": torch.zeros(1),
            "val_loss_G_adversarial": torch.zeros(1),
            "val_loss_G_pix": torch.zeros(1),
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
        }
        self.batch_size = torch.ones(1, device=self.device)
        self.make_new_labels()  # updates self.HR_labels, self.fake_HR_labels

        ###################
        # Define generator, discriminator, feature extractor
        ###################
        cfg_G: config.GeneratorConfig = cfg.generator
        cfg_gan: config.GANConfig = cfg.gan_config
        self.G = Generator_3D(
            cfg_G.in_num_ch
            + cfg_gan.include_pressure
            + cfg_gan.include_z_channel
            + cfg_gan.include_above_ground_channel,
            cfg_G.out_num_ch + cfg_gan.include_pressure,
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
        ).to(cfg.device, non_blocking=True)
        initialization.init_weights(self.G, scale=cfg_G.weight_init_scale)
        if torch.cuda.is_available() and not self.memory_dict.get("G"):
            self.memory_dict["G"] = torch.cuda.memory_allocated(self.device) / 1024**2

        self.conv_mode = cfg_G.conv_mode
        self.use_D_feature_extractor_cost = cfg_gan.use_D_feature_extractor_cost

        if cfg.is_train:
            cfg_D: config.DiscriminatorConfig = cfg.discriminator
            if cfg.dataset_train.hr_img_size == 128:
                self.D = Discriminator_3D(
                    cfg_D.in_num_ch + cfg_gan.include_pressure,
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
                ).to(cfg.device, non_blocking=True)
            else:
                raise NotImplementedError(
                    f"Discriminator for image size {cfg.image_size} har not been implemented.\
                                            Please train with a size that has been implemented."
                )

            # move to CUDA if available
            # self.HR_labels = self.HR_labels.to(cfg.device, non_blocking=True)
            # self.fake_HR_labels = self.fake_HR_labels.to(cfg.device, non_blocking=True)

            initialization.init_weights(self.D, scale=cfg_D.weight_init_scale)
            if torch.cuda.is_available() and not self.memory_dict.get("G_and_D"):
                self.memory_dict["G_and_D"] = (
                    torch.cuda.memory_allocated(self.device) / 1024**2
                )

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
                device=self.device
            )
            self.optimizer_D = torch.optim.Adam(
                self.D.parameters(),
                lr=cfg_t.learning_rate_d,
                weight_decay=cfg_t.adam_weight_decay_d,
                betas=(cfg_t.adam_beta1_d, 0.999),
                device=self.device,
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
            self.xy_divergence_criterion = nn.MSELoss().to(cfg.device, non_blocking=True)
            self.feature_D_criterion = nn.L1Loss().to(cfg.device, non_blocking=True)

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
    ):
        self.x = x
        self.y = y
        self.niter = niter

    def D_forward(
        self,
        HR: torch.Tensor,
        fake_HR: torch.Tensor,
        it: int,
        train_D: bool,
    ):
        if train_D:
            self.D.train()
            if self.cfg.training.use_instance_noise:
                y_pred = self.D(
                    HR
                    + trainingtricks.instance_noise(
                        torch.tensor(1.0, device=self.device), HR.size(), it, self.niter, device=self.device
                    )
                ).squeeze()
                fake_y_pred = self.D(
                    fake_HR.detach()
                    + trainingtricks.instance_noise(
                        torch.tensor(1.0, device=self.device), HR.size(), it, self.niter, device=self.device
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
                            torch.tensor(1.0, device=self.device), HR.size(), it, self.niter, device=self.device
                        )
                    )
                    .squeeze()
                    .detach()
                )
                fake_y_pred = self.D(
                    fake_HR
                    + trainingtricks.instance_noise(
                        torch.tensor(1.0, device=self.device), HR.size(), it, self.niter, device=self.device
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
            self.loss_dict["train_loss_G"] = loss_G
            self.loss_dict["train_loss_G_adversarial"] = loss_G_adversarial
            self.loss_dict["train_loss_G_xy_gradient"] = loss_G_xy_gradient
            self.loss_dict["train_loss_G_z_gradient"] = loss_G_z_gradient
            self.loss_dict["train_loss_G_divergence"] = loss_G_divergence
            self.loss_dict["train_loss_G_xy_divergence"] = loss_G_xy_divergence
            self.loss_dict["train_loss_G_feature_D"] = loss_G_feature_D
            self.loss_dict["train_loss_G_pix"] = loss_G_pix
        else:
            self.loss_dict["val_loss_G"] = loss_G
            self.loss_dict["val_loss_G_adversarial"] = loss_G_adversarial
            self.loss_dict["val_loss_G_xy_gradient"] = loss_G_xy_gradient
            self.loss_dict["val_loss_G_z_gradient"] = loss_G_z_gradient
            self.loss_dict["val_loss_G_divergence"] = loss_G_divergence
            self.loss_dict["val_loss_G_xy_divergence"] = loss_G_xy_divergence
            self.loss_dict["val_loss_G_feature_D"] = loss_G_feature_D
            self.loss_dict["val_loss_G_pix"] = loss_G_pix
            self.hist_dict["SR_pix_distribution"] = fake_HR.detach().cpu().numpy()
            # if self.conv_mode == "horizontal_3D":
            #     grad_start = self.G.model[0].convs[0][0].weight.grad.cpu().detach()
            #     grad_end = self.G.hr_convs[-1].convs[-1][-1].weight.grad.cpu().detach()
            #     weight_start = self.G.model[0].convs[0][0].weight.cpu().detach()
            #     weight_end = self.G.hr_convs[-1].convs[-1][-1].weight.cpu().detach()
            # else:
            #     grad_start = self.G.model[0][0].weight.grad.cpu().detach()
            #     grad_end = self.G.hr_convs[-1].weight.grad.cpu().detach()
            #     weight_start = self.G.model[0][0].weight.cpu().detach()
            #     weight_end = self.G.hr_convs[-1].weight.cpu().detach()
            # self.hist_dict["val_grad_G_first_layer"] = grad_start.numpy()
            # self.hist_dict["val_grad_G_last_layer"] = grad_end.numpy()
            # self.hist_dict["val_weight_G_first_layer"] = weight_start.numpy()
            # self.hist_dict["val_weight_G_last_layer"] = weight_end.numpy()

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

        loss_G_feature_D = 0
        if self.use_D_feature_extractor_cost:
            features = self.D.features(HR).detach()
            fake_features = self.D.features(fake_HR)
            loss_G_feature_D = self.feature_D_criterion(features, fake_features)

        loss_G_pix = 0
        if self.pixel_criterion:
            loss_G_pix = self.pixel_criterion(HR, fake_HR)

        (
            HR_wind_gradient,
            HR_divergence,
            HR_xy_divergence,
        ) = calculate_gradient_of_wind_field(HR[:, :3], self.x, self.y, Z)
        (
            SR_wind_gradient,
            SR_divergence,
            SR_xy_divergence,
        ) = calculate_gradient_of_wind_field(fake_HR[:, :3], self.x, self.y, Z)

        loss_G_xy_gradient = self.gradient_xy_criterion(
            SR_wind_gradient[:, :6] / torch.max(abs(HR_wind_gradient[:, :6])),
            HR_wind_gradient[:, :6] / torch.max(abs(HR_wind_gradient[:, :6])),
        )
        loss_G_z_gradient = self.gradient_z_criterion(
            SR_wind_gradient[:, 6:] / torch.max(HR_wind_gradient[:, 6:]),
            HR_wind_gradient[:, 6:] / torch.max(HR_wind_gradient[:, 6:]),
        )

        max_divergence = torch.max(
            abs(HR_divergence)
        )
        loss_G_divergence = self.divergence_criterion(
            HR_divergence / max_divergence, SR_divergence / max_divergence
        )

        max_xy_divergence = torch.max(
            abs((HR_xy_divergence))
        )
        loss_G_xy_divergence = self.xy_divergence_criterion(
            HR_xy_divergence / max_xy_divergence, SR_xy_divergence / max_xy_divergence
        )

        loss_G_adversarial *= self.cfg.training.adversarial_loss_weight
        loss_G_feature_D *= self.cfg.training.feature_D_loss_weight
        loss_G_pix *= self.cfg.training.pixel_loss_weight
        loss_G_xy_gradient *= self.cfg.training.gradient_xy_loss_weight
        loss_G_z_gradient *= self.cfg.training.gradient_z_loss_weight
        loss_G_divergence *= self.cfg.training.divergence_loss_weight
        loss_G_xy_divergence *= self.cfg.training.xy_divergence_loss_weight

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
            if torch.cuda.is_available() and not self.runtime_dict.get("G_backward"):
                start_GB = torch.cuda.Event(enable_timing=True)
                end_GB = torch.cuda.Event(enable_timing=True)
                start_GB.record()
                self.G.scaler.scale(loss_G).backward()
                # loss_G.backward()
                end_GB.record()
                self.runtime_dict["G_backward"] = (start_GB, end_GB)
                self.memory_dict["after_G_backward"] = (
                    torch.cuda.memory_allocated(self.device) / 1024**2
                )
                start_Gs = torch.cuda.Event(enable_timing=True)
                end_Gs = torch.cuda.Event(enable_timing=True)
                start_Gs.record()
                # self.optimizer_G.step()
                self.G.scaler.step(self.optimizer_G)
                self.G.scaler.update()
                end_Gs.record()
                self.runtime_dict["G_step"] = (start_Gs, end_Gs)
                self.memory_dict["after_G_step"] = (
                    torch.cuda.memory_allocated(self.device) / 1024**2
                )
            else:
                # loss_G.backward()
                self.G.scaler.scale(loss_G).backward()
                # self.optimizer_G.step()
                self.G.scaler.step(self.optimizer_G)
                self.G.scaler.update()

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
        else:
            self.G.eval()

        # with torch.autocast(self.device.type, enabled=self.cfg.generator.use_mixed_precision):
        if torch.cuda.is_available() and not self.runtime_dict.get("G_forward"):
            start_GF = torch.cuda.Event(enable_timing=True)
            end_GF = torch.cuda.Event(enable_timing=True)
            start_GF.record()
            fake_HR = self.G(LR, Z)
            end_GF.record()
            self.runtime_dict["G_forward"] = (start_GF, end_GF)
            self.memory_dict["after_G_forward"] = (
                torch.cuda.memory_allocated(self.device) / 1024**2
            )
        else:
            fake_HR = self.G(LR, Z)

        for param in self.D.parameters():
            param.requires_grad = False

        self.G.zero_grad(set_to_none=True)

        # with torch.autocast(self.device.type, enabled=self.cfg.discriminator.use_mixed_precision):
        y_pred, fake_y_pred = self.D_forward(HR, fake_HR, it, train_D=False)

        # with torch.autocast(self.device.type, enabled=self.cfg.generator.use_mixed_precision):

        self.calculate_optimize_and_log_G_loss(
            HR, fake_HR, Z, y_pred, fake_y_pred, training_iteration
        )
            
        return fake_HR

    def log_D_losses(self, loss_D, y_pred, fake_y_pred, training_epoch):
        if training_epoch:
            self.loss_dict["train_loss_D"] = loss_D
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
            self.loss_dict["val_loss_D"] = loss_D
            self.hist_dict["D_pred_HR"] = (
                torch.sigmoid(y_pred.detach()).cpu().numpy()[np.newaxis]
            )
            self.hist_dict["D_pred_SR"] = (
                torch.sigmoid(fake_y_pred.detach()).cpu().numpy()[np.newaxis]
            )

    def update_D(
        self, HR: torch.Tensor, fake_HR: torch.Tensor, it, training_epoch: bool
    ):
        for param in self.D.parameters():
            param.requires_grad = True

        self.optimizer_D.zero_grad(set_to_none=True)

        # with torch.autocast(self.device.type, enabled=self.cfg.discriminator.use_mixed_precision):
        if torch.cuda.is_available() and not self.runtime_dict.get("D_forward"):
            start_DF = torch.cuda.Event(enable_timing=True)
            end_DF = torch.cuda.Event(enable_timing=True)
            start_DF.record()
            y_pred, fake_y_pred = self.D_forward(HR, fake_HR, it, train_D=True)
            end_DF.record()
            self.runtime_dict["D_forward"] = (start_DF, end_DF)
            self.memory_dict["after_D_forward"] = (
                torch.cuda.memory_allocated(self.device) / 1024**2
            )
        else:
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
        else:
            raise NotImplementedError(
                f"Only relativistic and relativisticavg GAN are implemented, not {self.cfg.training.gan_type}"
            )


        if training_epoch:
            if torch.cuda.is_available() and not self.runtime_dict.get("D_backward"):
                start_DB = torch.cuda.Event(enable_timing=True)
                end_DB = torch.cuda.Event(enable_timing=True)
                start_DB.record()
                self.D.scaler.scale(loss_D).backward()
                # loss_D.backward()
                end_DB.record()
                self.runtime_dict["D_backward"] = (start_DB, end_DB)
                self.memory_dict["after_D_backward"] = (
                    torch.cuda.memory_allocated(self.device) / 1024**2
                )
                start_Ds = torch.cuda.Event(enable_timing=True)
                end_Ds = torch.cuda.Event(enable_timing=True)
                start_Ds.record()
                # self.optimizer_D.step()
                self.D.scaler.step(self.optimizer_D)
                self.D.scaler.update()
                end_Ds.record()
                self.runtime_dict["D_step"] = (start_Ds, end_Ds)
                self.memory_dict["after_D_step"] = (
                    torch.cuda.memory_allocated(self.device) / 1024**2
                )
            else:
                # self.optimizer_D.step()
                self.D.scaler.step(self.optimizer_D)
                self.D.scaler.update()       
            
        self.log_D_losses(loss_D, y_pred, fake_y_pred, training_epoch=training_epoch)

    def compute_losses_and_optimize(
        self, LR, HR, Z, it, training_iteration: bool = False
    ):
        self.batch_size = HR.size(0)
        self.make_new_labels()
        it = torch.tensor(it, device=self.device)

        ###################
        # Update G
        ###################
        if it % 2 == 0:  # self.cfg.d_g_train_ratio == 0:
            fake_HR = self.update_G(LR, HR, Z, it, training_iteration)
        else:
            if torch.cuda.is_available() and not self.runtime_dict.get(
                "G_forward_no_grad"
            ):
                start_G_no_grad = torch.cuda.Event(enable_timing=True)
                end_G_no_grad = torch.cuda.Event(enable_timing=True)
                start_G_no_grad.record()
                with torch.no_grad():
                    fake_HR = self.G(LR, Z).to(self.device)
                end_G_no_grad.record()
                self.runtime_dict["G_forward_no_grad"] = (
                    start_G_no_grad,
                    end_G_no_grad,
                )
                self.memory_dict["after_G_forward_no_grad"] = (
                    torch.cuda.memory_allocated(self.device) / 1024**2
                )
            else:
                with torch.no_grad():
                    fake_HR = self.G(LR, Z).to(self.device)
        ###################
        # Update D
        ###################
        self.update_D(HR, fake_HR, it, training_iteration)


        if not training_iteration:
            if it / self.cfg.training.niter > 0.75:
                (
                    self.metrics_dict["val_PSNR"],
                    self.metrics_dict["Trilinear_PSNR"],
                ) = compute_psnr_for_SR_and_trilinear(LR, HR, fake_HR, interpolate=True, device=self.device)
            else:
                self.metrics_dict["val_PSNR"] = compute_psnr_for_SR_and_trilinear(
                    LR, HR, fake_HR, interpolate=False, device=self.device
                )

    def optimize_parameters(self, LR, HR, Z, it):
        self.compute_losses_and_optimize(LR, HR, Z, it, training_iteration=True)

    def validation(self, LR, HR, Z, it):
        self.compute_losses_and_optimize(LR, HR, Z, it, training_iteration=False)

    def make_new_labels(self):
        pred_real = True
        pred_fake = False

        if self.cfg.training.flip_labels:
            pred_real = False
            pred_fake = True

        real_label = 1.0
        fake_label = 0.0
        if (
            self.cfg.training.use_one_sided_label_smoothing
            and self.cfg.training.flip_labels
        ):
            real_label = 1.0
            fake_label = 0.1
        elif self.cfg.training.use_one_sided_label_smoothing:
            real_label = 0.9
            fake_label = 0.0

        if self.cfg.training.use_noisy_labels:
            self.HR_labels = (
                trainingtricks.noisy_labels(
                    pred_real,
                    self.batch_size,
                    true_label_val=real_label,
                    false_label_val=fake_label,
                )
                .to(self.device, non_blocking=True)
                .squeeze()
            )
            self.fake_HR_labels = (
                trainingtricks.noisy_labels(
                    pred_fake,
                    self.batch_size,
                    true_label_val=real_label,
                    false_label_val=fake_label,
                )
                .to(self.device, non_blocking=True)
                .squeeze()
            )
        else:  # no noise std dev -> no noise
            self.HR_labels = (
                trainingtricks.noisy_labels(
                    pred_real,
                    self.batch_size,
                    noise_stddev=0.0,
                    true_label_val=real_label,
                    false_label_val=fake_label,
                )
                .to(self.device, non_blocking=True)
                .squeeze()
            )
            self.fake_HR_labels = (
                trainingtricks.noisy_labels(
                    pred_fake,
                    self.batch_size,
                    noise_stddev=0.0,
                    true_label_val=real_label,
                    false_label_val=fake_label,
                )
                .to(self.device, non_blocking=True)
                .squeeze()
            )

    def test(self):
        raise NotImplementedError("test has not been implemented.")

    def get_loss_dict_ref(self):
        return self.loss_dict

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


def compute_psnr_for_SR_and_trilinear(
    LR, HR: torch.Tensor, fake_HR: torch.Tensor, interpolate: bool = False, device=torch.device("cpu")
):
    w, h, l = HR.shape[2], HR.shape[3], HR.shape[4]
    SR_batch_average_MSE = torch.sum((HR - fake_HR) ** 2) / (w * h * l * HR.shape[0])
    SR_batch_average_MSE = SR_batch_average_MSE
    max_diff_squared = torch.tensor(4.0, device=device)  # HR is in [-1, 1]
    epsilon = torch.tensor(1e-8, device=device)  # PSNR is usually ~< 50 so this should not impact the result much
    val_PSNR = torch.tensor(10, device=device) * math.log10(max_diff_squared / (SR_batch_average_MSE + epsilon))
    if interpolate:
        interpolated_LR = nn.functional.interpolate(
            LR[:, :4, :, :, :],
            scale_factor=(4, 4, 1),
            mode="trilinear",
            align_corners=True,
        )
        interp_batch_average_MSE = torch.sum((HR - interpolated_LR) ** 2) / (
            w * h * l * HR.shape[0]
        )
        interp_batch_average_MSE = interp_batch_average_MSE
        val_trilinear_PSNR =  torch.tensor(10, device=device) * math.log10(
            max_diff_squared / (interp_batch_average_MSE + epsilon)
        )
        return val_PSNR, val_trilinear_PSNR
    else:
        return val_PSNR
