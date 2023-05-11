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
        self.memory_dict={}
        self.loss_dict = {
            "train_loss_D": 0.0,
            "train_loss_G": 0.0,
            "train_loss_G_adversarial": 0.0,
            "train_loss_G_pix": 0.0,
            "train_loss_G_xy_gradient": 0.0,
            "train_loss_G_z_gradient": 0.0,
            "train_loss_G_divergence": 0.0,
            "train_loss_G_xy_divergence": 0.0,
            "train_loss_G_feature_D": 0.0,
            "val_loss_G_xy_gradient": 0.0,
            "val_loss_G_z_gradient": 0.0,
            "val_loss_G_divergence": 0.0,
            "val_loss_G_xy_divergence": 0.0,
            "val_loss_G_feature_D": 0.0,
            "val_loss_D": 0.0,
            "val_loss_G": 0.0,
            "val_loss_G_adversarial": 0.0,
            "val_loss_G_pix": 0.0,
        }
        self.hist_dict = {
            "val_grad_G_first_layer": 0.0,
            "val_grad_G_last_layer": 0.0,
            "val_grad_D_first_layer": -1.0,
            "val_grad_D_last_layer": -1.0,
            "val_weight_G_first_layer": 0.0,
            "val_weight_G_last_layer": 0.0,
            "val_weight_D_first_layer": -1.0,
            "val_weight_D_last_layer": -1.0,
            "SR_pix_distribution": 0.0,
            "D_pred_HR": 0.0,
            "D_pred_SR": 0.0,
        }

        self.metrics_dict = {
            "val_PSNR": 0.0,
        }
        self.batch_size = 1
        self.make_new_labels()  # updates self.HR_labels, self.fake_HR_labels

        ###################
        # Define generator, discriminator, feature extractor
        ###################
        cfg_g: config.GeneratorConfig = cfg.generator
        cfg_gan: config.GANConfig = cfg.gan_config
        self.G = Generator_3D(
            cfg_g.in_num_ch + cfg_gan.include_pressure + cfg_gan.include_z_channel + cfg_gan.include_above_ground_channel,
            cfg_g.out_num_ch + cfg_gan.include_pressure,
            cfg_g.num_features,
            cfg_g.num_RRDB,
            upscale=cfg.scale,
            hr_kern_size=cfg_g.hr_kern_size,
            number_of_RDB_convs=cfg_g.num_RDB_convs,
            RDB_gc=cfg_g.RDB_growth_chan,
            lff_kern_size=cfg_g.lff_kern_size,
            RDB_residual_scaling=cfg_g.RDB_res_scaling,
            RRDB_residual_scaling=cfg_g.RRDB_res_scaling,
            act_type=cfg_g.act_type,
            device=self.device,
            number_of_z_layers=cfg_gan.number_of_z_layers,
            conv_mode=cfg_gan.conv_mode,
        ).to(cfg.device)
        initialization.init_weights(self.G, scale=cfg_g.weight_init_scale)
        if torch.cuda.is_available() and not self.memory_dict.get("G"):
            self.memory_dict["G"] =torch.cuda.memory_allocated(self.device) / 1024**2
        
        self.conv_mode = cfg_g.conv_mode
        self.use_D_feature_extractor_cost = cfg_gan.use_D_feature_extractor_cost

        if cfg.is_train:
            cfg_d: config.DiscriminatorConfig = cfg.discriminator
            if cfg.dataset_train.hr_img_size == 128:
                self.D = Discriminator_3D(
                    cfg_d.in_num_ch + cfg_gan.include_pressure,
                    cfg_d.num_features,
                    feat_kern_size=cfg_d.feat_kern_size,
                    normalization_type=cfg_d.norm_type,
                    act_type=cfg_d.act_type,
                    mode=cfg_d.layer_mode,
                    device=self.device,
                    number_of_z_layers=cfg_gan.number_of_z_layers,
                    conv_mode=cfg_gan.conv_mode,
                )
            else:
                raise NotImplementedError(
                    f"Discriminator for image size {cfg.image_size} har not been implemented.\
                                            Please train with a size that has been implemented."
                )

            # move to CUDA if available
            self.HR_labels = self.HR_labels.float().to(cfg.device)
            self.fake_HR_labels = self.fake_HR_labels.float().to(cfg.device)
            self.D = self.D.float().to(cfg.device)
            
            # self.F = self.F.to(cfg.device)
            initialization.init_weights(self.D, scale=cfg_d.weight_init_scale)
            if torch.cuda.is_available() and not self.memory_dict.get("G_and_D"):
                self.memory_dict["G_and_D"] =torch.cuda.memory_allocated(self.device) / 1024**2
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
            self.gradient_xy_criterion = nn.MSELoss()
            self.gradient_z_criterion = nn.MSELoss()
            self.divergence_criterion = nn.MSELoss()
            self.xy_divergence_criterion = nn.MSELoss()
            self.feature_D_criterion = nn.L1Loss()

            if cfg_t.pixel_criterion is None or cfg_t.pixel_criterion == "none":
                self.pixel_criterion = None
            elif cfg_t.pixel_criterion == "l1":
                self.pixel_criterion = nn.L1Loss()
            elif cfg_t.pixel_criterion == "l2":
                self.pixel_criterion = nn.MSELoss()
            else:
                raise NotImplementedError(
                    f"Only l1 and l2 (MSE) loss have been implemented for pixel loss, not {cfg_t.pixel_criterion}"
                )

            # GAN adversarial loss
            if cfg_t.gan_type == "relativistic" or cfg_t.gan_type == "relativisticavg":
                self.criterion = nn.BCEWithLogitsLoss().float().to(cfg.device)
            else:
                raise NotImplementedError(
                    f"Only relativistic and relativisticavg GAN are implemented, not {cfg_t.gan_type}"
                )
            return

    def feed_data(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor = None,
        x: torch.Tensor = torch.Tensor([]),
        y: torch.Tensor = torch.Tensor([]),
        Z: torch.Tensor = torch.Tensor([]),
    ):
        self.lr = lr
        self.hr = hr
        if x.any():
            self.x = x
            self.y = y
        if Z.any():
            self.Z = Z

    def compute_losses_and_optimize(
        self, it, training_epoch: bool = False, validation_epoch: bool = False
    ):
        """
        process_data
        computes losses, and if it is a training epoch, performs parameter optimization
        """
        if (not training_epoch and not validation_epoch) or (
            training_epoch and validation_epoch
        ):
            raise ValueError("process_data requires exactly one input as true")

        self.fake_hr = self.G(self.lr, self.Z).to(self.device)
        if torch.cuda.is_available() and not self.memory_dict.get("after_G_forward"):
            self.memory_dict["after_G_forward"] = torch.cuda.memory_allocated(self.device) / 1024**2

        # changes when going from train <-> val <-> test
        # (at least when data loader has drop_last=True )
        current_batch_size = self.hr.size(0)
        if current_batch_size != self.batch_size:
            self.batch_size = current_batch_size

        self.make_new_labels()

        ###################
        # Update G
        ###################
        if it % 2 == 0:  # self.cfg.d_g_train_ratio == 0:
            for param in self.D.parameters():
                param.requires_grad = False

            self.G.zero_grad()

            y_pred = None
            fake_y_pred = None
            if self.cfg.training.use_instance_noise:
                y_pred = (
                    self.D(
                        self.hr
                        + trainingtricks.instance_noise(
                            1.0, self.hr.size(), it, self.cfg.training.niter
                        )
                        .float()
                        .to(self.device)
                    )
                    .squeeze()
                    .detach()
                )
                fake_y_pred = self.D(
                    self.fake_hr
                    + trainingtricks.instance_noise(
                        1, self.hr.size(), it, self.cfg.training.niter
                    )
                    .float()
                    .to(self.device)
                ).squeeze()
            else:
                y_pred = self.D(self.hr).squeeze().detach()
                fake_y_pred = self.D(self.fake_hr).squeeze()

            # adversarial loss
            loss_G_adversarial = 0

            if self.cfg.training.gan_type == "dcgan":
                loss_D = self.criterion(fake_y_pred, self.HR_labels) + self.criterion(
                    y_pred, self.fake_HR_labels
                )
            if self.cfg.training.gan_type == "relativistic":
                loss_G_adversarial = self.criterion(
                    fake_y_pred - y_pred, self.HR_labels
                )

            elif self.cfg.training.gan_type == "relativisticavg":
                loss_G_adversarial = (
                    self.criterion(fake_y_pred - torch.mean(y_pred), self.HR_labels)
                    + self.criterion(
                        y_pred - torch.mean(fake_y_pred), self.fake_HR_labels
                    )
                ) / 2.0
            else:
                raise NotImplementedError(
                    f"Only relativistic and relativisticavg GAN are implemented, not {self.cfg.training.gan_type}"
                )

            loss_G_feature_D = 0
            if self.use_D_feature_extractor_cost:
                features = self.D.features(self.hr).detach()
                fake_features = self.D.features(self.fake_hr)
                loss_G_feature_D = self.feature_D_criterion(features, fake_features)

            loss_G_pix = 0
            if self.pixel_criterion:
                loss_G_pix = self.pixel_criterion(self.hr, self.fake_hr)

            HR_wind_gradient, HR_divergence, HR_xy_divergence = calculate_gradient_of_wind_field(
                self.hr[:, :3], self.x, self.y, self.Z
            )
            SR_wind_gradient, SR_divergence, SR_xy_divergence = calculate_gradient_of_wind_field(
                self.fake_hr[:, :3], self.x, self.y, self.Z
            )

            loss_G_xy_gradient = self.gradient_xy_criterion(
                SR_wind_gradient[:, :6] / torch.max(abs(SR_wind_gradient[:, :6])),
                HR_wind_gradient[:, :6] / torch.max(abs(HR_wind_gradient[:, :6])),
            )
            loss_G_z_gradient = self.gradient_z_criterion(
                SR_wind_gradient[:, 6:] / torch.max(SR_wind_gradient[:, 6:]),
                HR_wind_gradient[:, 6:] / torch.max(HR_wind_gradient[:, 6:]),
            )
            

            max_divergence = torch.max(
                abs(torch.cat((HR_divergence, SR_divergence), dim=0))
            )
            loss_G_divergence = self.divergence_criterion(
                HR_divergence / max_divergence, SR_divergence / max_divergence
            )

            max_xy_divergence = torch.max(
                abs(torch.cat((HR_xy_divergence, SR_xy_divergence), dim=0))
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

            # normalize by batch sz, this is not done in ESRGAN
            # loss_D.mul_(1.0 / current_batch_size)

            loss_G.backward()

            if torch.cuda.is_available() and not self.memory_dict.get("after_G_backward"):
                self.memory_dict["after_G_backward"] =torch.cuda.memory_allocated(self.device) / 1024**2

            if training_epoch:
                self.loss_dict["train_loss_G"] = loss_G.item()
                self.loss_dict["train_loss_G_adversarial"] = loss_G_adversarial.item()
                self.loss_dict["train_loss_G_xy_gradient"] = loss_G_xy_gradient.item()
                self.loss_dict["train_loss_G_z_gradient"] = loss_G_z_gradient.item()
                self.loss_dict["train_loss_G_divergence"] = loss_G_divergence.item()
                self.loss_dict["train_loss_G_xy_divergence"] = loss_G_xy_divergence.item()
                self.loss_dict["train_loss_G_feature_D"] = loss_G_feature_D.item()
                self.loss_dict["train_loss_G_pix"] = loss_G_pix.item()
                self.hist_dict["SR_pix_distribution"] = (
                    self.fake_hr.detach().cpu().numpy()
                )
                self.optimizer_G.step()
                if torch.cuda.is_available() and not self.memory_dict.get("after_G_step"):
                    self.memory_dict["after_G_step"] =torch.cuda.memory_allocated(self.device) / 1024 ** 2
                
            else:
                self.loss_dict["val_loss_G"] = loss_G.item()
                self.loss_dict["val_loss_G_adversarial"] = loss_G_adversarial.item()
                self.loss_dict["val_loss_G_xy_gradient"] = loss_G_xy_gradient.item()
                self.loss_dict["val_loss_G_z_gradient"] = loss_G_z_gradient.item()
                self.loss_dict["val_loss_G_divergence"] = loss_G_divergence.item()
                self.loss_dict["val_loss_G_xy_divergence"] = loss_G_xy_divergence.item()
                self.loss_dict["val_loss_G_feature_D"] = loss_G_feature_D.item()
                self.loss_dict["val_loss_G_pix"] = loss_G_pix.item()
                if self.conv_mode == "horizontal_3D":
                    grad_start = self.G.model[0].convs[0][0].weight.grad.cpu().detach()
                    grad_end = self.G.hr_convs[-1].convs[-1][-1].weight.grad.cpu().detach()
                    weight_start = self.G.model[0].convs[0][0].weight.cpu().detach()
                    weight_end = self.G.hr_convs[-1].convs[-1][-1].weight.cpu().detach()
                else:
                    grad_start = self.G.model[0][0].weight.grad.cpu().detach()
                    grad_end = self.G.hr_convs[-1].weight.grad.cpu().detach()
                    weight_start = self.G.model[0][0].weight.cpu().detach()
                    weight_end = self.G.hr_convs[-1].weight.cpu().detach()

                self.hist_dict["val_grad_G_first_layer"] = grad_start.numpy()
                self.hist_dict["val_grad_G_last_layer"] = grad_end.numpy()
                self.hist_dict["val_weight_G_first_layer"] = weight_start.numpy()
                self.hist_dict["val_weight_G_last_layer"] = weight_end.numpy()

        ###################
        # Update D
        ###################

        for param in self.D.parameters():
            param.requires_grad = True

        self.optimizer_D.zero_grad()

        # squeeze to go from shape [batch_sz, 1] to [batch_sz]
        y_pred = None
        fake_y_pred = None
        if self.cfg.training.use_instance_noise:
            y_pred = self.D(
                self.hr
                + trainingtricks.instance_noise(
                    1, self.hr.size(), it, self.cfg.training.niter
                )
                .float()
                .to(self.device)
            ).squeeze()
            fake_y_pred = self.D(
                self.fake_hr.detach()
                + trainingtricks.instance_noise(
                    1, self.hr.size(), it, self.cfg.training.niter
                )
                .float()
                .to(self.device)
            ).squeeze()  # detach -> avoid BP to G
        else:
            y_pred = self.D(self.hr).squeeze()
            fake_y_pred = self.D(
                self.fake_hr.detach()
            ).squeeze()  # detach -> avoid BP to G
        
        if torch.cuda.is_available() and not self.memory_dict.get("after_D_forward"):
            self.memory_dict["after_D_forward"] =torch.cuda.memory_allocated(self.device) / 1024 ** 2

        # D only has adversarial loss.
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

        # normalize by batch sz, this is not done in ESRGAN
        # loss_D.mul_(1.0 / current_batch_size)
        loss_D.backward()
        if torch.cuda.is_available() and not self.memory_dict.get("after_D_backward"):
            self.memory_dict["after_D_backward"] =torch.cuda.memory_allocated(self.device) / 1024 ** 2

        if training_epoch:
            self.loss_dict["train_loss_D"] = loss_D.item()
            # BCEWithLogitsLoss has sigmoid activation.
            self.optimizer_D.step()
            if torch.cuda.is_available() and not self.memory_dict.get("after_D_step"):
                self.memory_dict["after_D_step"] =torch.cuda.memory_allocated(self.device) / 1024 ** 2
        else:
            # features[0] is StridedDownConv2x, whose first elem is a nn.Conv2D
            if self.conv_mode == "horizontal_3D":
                grad_start = (
                    self.D.features[0][0].convs[0][0].weight.grad.detach().cpu()
                )
                weight_start = self.D.features[0][0].convs[0][0].weight.detach().cpu()
                grad_end = self.D.classifier[-1].weight.grad.detach().cpu()
                weight_end = self.D.classifier[-1].weight.detach().cpu()
            else:
                grad_start = self.D.features[0][0][0].weight.grad.detach().cpu()
                weight_start = self.D.features[0][0][0].weight.detach().cpu()
                grad_end = self.D.classifier[-1].weight.grad.detach().cpu()
                weight_end = self.D.classifier[-1].weight.detach().cpu()

            self.hist_dict["val_grad_D_first_layer"] = grad_start.numpy()
            self.hist_dict["val_grad_D_last_layer"] = grad_end.numpy()
            self.hist_dict["val_weight_D_first_layer"] = weight_start.numpy()
            self.hist_dict["val_weight_D_last_layer"] = weight_end.numpy()
            self.loss_dict["val_loss_D"] = loss_D.item()
            self.hist_dict["D_pred_HR"] = (
                torch.sigmoid(y_pred.detach()).cpu().numpy()[np.newaxis]
            )
            self.hist_dict["D_pred_SR"] = (
                torch.sigmoid(fake_y_pred.detach()).cpu().numpy()[np.newaxis]
            )

    def optimize_parameters(self, it):
        self.compute_losses_and_optimize(it, training_epoch=True)

    def validation(self, it):
        self.compute_losses_and_optimize(it, validation_epoch=True)
        self.compute_psnr_x_batch_size()

    def compute_psnr_x_batch_size(self):
        # zeros = torch.FloatTensor(self.batch_size).fill_(0.0).to(self.cfg.device)
        w, h, l = self.hr.shape[2], self.hr.shape[3], self.hr.shape[4]
        batch_MSE = torch.sum((self.hr - self.fake_hr) ** 2) / (w * h * l)
        batch_MSE = batch_MSE.item()
        R_squared = 1.0  # R is max fluctuation, and data is float [0, 1] -> R² = 1
        epsilon = (
            1e-8  # PSNR is usually ~< 50 so this should not impact the result much
        )
        self.metrics_dict["val_PSNR"] = (
            self.batch_size * 10 * math.log10(R_squared / (batch_MSE + epsilon))
        )

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
            fake_label = 0.1  # WAS 0.0 --Thomas

        if self.cfg.training.use_noisy_labels:
            self.HR_labels = (
                trainingtricks.noisy_labels(
                    pred_real,
                    self.batch_size,
                    true_label_val=real_label,
                    false_label_val=fake_label,
                )
                .float()
                .to(self.device)
                .squeeze()
            )
            self.fake_HR_labels = (
                trainingtricks.noisy_labels(
                    pred_fake,
                    self.batch_size,
                    true_label_val=real_label,
                    false_label_val=fake_label,
                )
                .float()
                .to(self.device)
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
                .float()
                .to(self.device)
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
                .float()
                .to(self.device)
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
