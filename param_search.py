from ray import tune
from ray.tune.schedulers import ASHAScheduler
from train import log_status_logs
import config.config as config
import logging
from ray.air import Checkpoint, session
import torch
import torch.cuda
import tensorboardX
from GAN_models.wind_field_GAN_3D import wind_field_GAN_3D
from functools import partial
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
import iocomponents.displaybar as displaybar
import copy
import os

def train_param_search(cfg:config.Config, cfg_env, cfg_gan, cfg_G, cfg_D, cfg_train, cfg_datatrain, cfg_dataval, dataset_train, dataset_validation, x, y, search_cfg:dict):

    cfg.env = cfg_env
    cfg.gan_config = cfg_gan
    cfg.generator=cfg_G
    cfg.discriminator=cfg_D
    cfg.training=cfg_train
    cfg.dataset_train=cfg_datatrain
    cfg.dataset_val=cfg_dataval

    # print(cfg, dataset_train, dataset_validation, x, y, search_cfg)
    cfg.training.gradient_xy_loss_weight = search_cfg["gradient_xy_loss_weight"]
    cfg.training.gradient_z_loss_weight = search_cfg["gradient_z_loss_weight"]
    cfg.training.xy_divergence_loss_weight = search_cfg["xy_divergence_loss_weight"]
    cfg.training.divergence_loss_weight = search_cfg["divergence_loss_weight"]
    cfg.training.pixel_loss_weight = search_cfg["pixel_loss_weight"]
    cfg.gpu_id = search_cfg["gpu_id"]
    cfg.name =  cfg.name+str(search_cfg.values())

    if torch.cuda.is_available():
        cfg.device = torch.device(f"cuda:{cfg.gpu_id}")
    
    cfg_t = cfg.training
    status_logger = logging.getLogger("status")
    if cfg.use_tensorboard_logger:
        os.makedirs(cfg.env.this_runs_tensorboard_log_folder+"/"+str(search_cfg.values()))
        tb_writer = tensorboardX.SummaryWriter(
            log_dir=cfg.env.this_runs_tensorboard_log_folder+"/"+str(search_cfg.values())
        )
    else:
        tb_writer = None

    train_logger = logging.getLogger("train")

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

    status_logger.debug(f"GAN:\n{str(gan)}\n")
    log_status_logs(status_logger, gan.get_new_status_logs())

    start_epoch = 0
    it = 0
    it_per_epoch = len(dataloader_train)
    count_train_epochs = 1 + cfg_t.niter // it_per_epoch
    loaded_it = 0
    
    # bar = displaybar.DisplayBar(
    #     max_value=len(dataloader_train),
    #     start_epoch=start_epoch,
    #     start_it=it,
    #     niter=cfg_t.niter,
    # )
    
    status_logger.info(
        "storing LR and HR validation images in run folder, for reference"
    )
    # store_LR_HR_in_runs_folder(cfg, dataloader_val)
    # checkpoint = session.get_checkpoint()

    # if checkpoint:
    #     checkpoint_state = checkpoint.to_dict()
    #     it = checkpoint_state["it"]
    #     start_epoch = checkpoint_state["epoch"]
    #     gan.G.load_state_dict(checkpoint_state["G_state_dict"])
    #     gan.optimizer_G.load_state_dict(checkpoint_state["G_optimizer_state_dict"])

    status_logger.info(f"beginning run from epoch {start_epoch}, it {it}")
    
    for epoch in range(start_epoch, count_train_epochs):
        status_logger.debug("epoch {epoch}")

        # dataloader -> (LR, HR, HR_img_name)
        for i, (LR, HR, Z) in enumerate(dataloader_train):

            if it > cfg_t.niter:
                break
            
            it += 1
            # bar.update(i, epoch, it)

            LR = LR.to(cfg.device, non_blocking=True)
            HR = HR.to(cfg.device, non_blocking=True)
            Z = Z.to(cfg.device, non_blocking=True)
            
            if it == loaded_it + 1:
                x = x.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                gan.feed_xy_niter(
                    x, y, torch.tensor(cfg_t.niter, device=cfg.device), cfg_t.d_g_train_ratio,
                )

            gan.optimize_parameters(LR, HR, Z, it)

            gan.update_learning_rate() if i > 0 else None

            l = gan.get_new_status_logs()

            if len(l) > 0:
                for log in l:
                    train_logger.info(log)

            if it % cfg_t.log_period == 0:
                if cfg.use_tensorboard_logger:
                    losses = dict(
                        (val_name, val.item())
                        for val_name, val in gan.get_G_train_loss_dict_ref().items()
                    )
                    tb_writer.add_scalars("G_loss/train", losses, it)
                # session.report({**(gan.get_G_train_loss_dict_ref()), "it": it})

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

                # checkpoint_data = {
                #     "it": it,
                #     "epoch": epoch,
                #     "G_state_dict": gan.G.state_dict(),
                #     "G_optimizer_state_dict": gan.optimizer_G.state_dict(),
                # }
                # checkpoint = Checkpoint.from_dict(checkpoint_data)

                session.report(
                    {"it":it, "PSNR": metrics_vals["val_PSNR"], "pix": metrics_vals["pix_loss_unscaled"]}
                )

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
    return


def param_search(num_samples=10, number_of_GPUs=1, cfg:config.Config=None, dataset_train=None, dataset_validation=None, x=None, y=None):
    search_space = {
        "gradient_xy_loss_weight": tune.loguniform(1.0, 50.0, 2),  #tune.choice([0.0, 1.0, 5.0, 10.0, 20.0, 50.0]) #1.0
        "gradient_z_loss_weight": tune.loguniform(1.0, 50.0, 2), #0.2
        "xy_divergence_loss_weight": tune.loguniform(0.4, 20.0, 2), #tune.choice([[0.0, 0.4, 2.0, 4.0, 7.0, 20 ]]) #0.25
        "divergence_loss_weight": tune.loguniform(0.7, 40.0, 2), #tune.choice([[0.0, 0.8, 4.0, 8.0, 15.0, 40.0 ]]) #0.25
        "pixel_loss_weight": tune.uniform(0.0, 1.0), #0.5
        'gpu_id': tune.sample_from(lambda _: torch.cuda.current_device() if torch.cuda.is_available() else -1),
    }
    scheduler = ASHAScheduler(
        time_attr="it",
        max_t=cfg.training.niter,
        grace_period=2,
        reduction_factor=2,
    )

    initial_search_config = [
    {
        "gradient_xy_loss_weight": 5.0,  #tune.choice([0.0, 1.0, 5.0, 10.0, 20.0, 50.0]) #1.0
        "gradient_z_loss_weight": 5.0, #0.2
        "xy_divergence_loss_weight": 2.0, #tune.choice([[0.0, 0.4, 2.0, 4.0, 7.0, 20 ]]) #0.25
        "divergence_loss_weight": 4.0, #tune.choice([[0.0, 0.8, 4.0, 8.0, 15.0, 40.0 ]]) #0.25
        "pixel_loss_weight": 0.2, #0.5
        'gpu_id': 0,
    },
    {
        "gradient_xy_loss_weight": 10.0,  #tune.choice([0.0, 1.0, 5.0, 10.0, 20.0, 50.0]) #1.0
        "gradient_z_loss_weight": 2.0, #0.2
        "xy_divergence_loss_weight": 5.0, #tune.choice([[0.0, 0.4, 2.0, 4.0, 7.0, 20 ]]) #0.25
        "divergence_loss_weight": 2.0, #tune.choice([[0.0, 0.8, 4.0, 8.0, 15.0, 40.0 ]]) #0.25
        "pixel_loss_weight": 0.2, #0.5
        'gpu_id': 0,
    },
    ]

    search_algorithm = OptunaSearch() #points_to_evaluate=initial_search_config
    search_algorithm = ConcurrencyLimiter(search_algorithm, max_concurrent=number_of_GPUs)

    

    # train_param_search(initial_search_config, cfg, dataset_train, dataset_validation, x, y)


    result = tune.run(
        partial(train_param_search, cfg, cfg.env, cfg.gan_config, cfg.generator, cfg.discriminator, cfg.training, cfg.dataset_train, cfg.dataset_val, copy.deepcopy(dataset_train), copy.deepcopy(dataset_validation), x, y),
        resources_per_trial={"cpu": cfg.dataset_train.num_workers, "gpu": 1 if torch.cuda.is_available() else 0},
        config=search_space,
        num_samples=num_samples,
        metric="PSNR",
        mode="max",
        scheduler=scheduler,
        search_alg=search_algorithm,
        local_dir=cfg.env.this_runs_folder,
        chdir_to_trial_dir=False,
        sync_config=tune.SyncConfig(
            syncer=None
        ),
    )

    best_trial = result.get_best_trial("PSNR", "max", "last")
    print(f"Best trial config: {best_trial.config}")


