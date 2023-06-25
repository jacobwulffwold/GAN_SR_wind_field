"""
run.py
Written by Eirik Vesterkjær, 2019, edited by Thomas Nakken Larsen 2020 and Jacob Wulff Wold 2023
Apache License

Entry point for training or testing wind_field_GAN_3D
Sets up environment/logging, and starts training/testing
Usage:
    python run.py < --train | --test | --use > [ --cfg path/to/config.ini ] [ -h ]

"""

import argparse
import logging
import os
import time

import torch
import torch.cuda

from config.config import Config
import train
import test
from datetime import date
from process_data import preprosess
import numpy as np
import random
from param_search import param_search

def main():
    cfg: Config = argv_to_cfg()
    # cfg.is_train = True
    # cfg.is_download = True
    # cfg.is_param_search = True
    # cfg.is_train = True
    if not cfg.is_test and not cfg.is_train and not cfg.is_use and not cfg.is_download and not cfg.is_param_search:
        print(
            "pass either --test, --download, --use or --train as args, and optionally --cfg path/to/config.ini if config/wind_field_GAN_2D_config.ini isn't what you're planning on using."
        )
        return
    
    if cfg.slurm_array_id == 1:
        cfg.name = cfg.name + "_2lr_clip1"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/G_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/D_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/state_60000.pth"
        cfg.load_model_from_save = True
        cfg.training.resume_training_from_save = True

    if cfg.slurm_array_id == 2:
        cfg.name = cfg.name + "_2lr_clipInf"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/G_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/D_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/state_60000.pth"
        cfg.load_model_from_save = True
        cfg.training.resume_training_from_save = True
        cfg.generator.max_norm = 10**10

    if cfg.slurm_array_id == 3:
        cfg.name = cfg.name + "_4lr_clip1"
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*2
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/G_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/D_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/state_60000.pth"
        cfg.load_model_from_save = True
        cfg.training.resume_training_from_save = True

    if cfg.slurm_array_id == 4:
        cfg.name = cfg.name + "_4lr_clipInf"
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*2
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2
        cfg.generator.max_norm = 10**10
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/G_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/D_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/state_60000.pth"
        cfg.load_model_from_save = True
        cfg.training.resume_training_from_save = True

    if cfg.slurm_array_id == 5: #stpped
        cfg.name = cfg.name + "_4lr_clip2_adv"
        cfg.training.adversarial_loss_weight = cfg.training.adversarial_loss_weight*4
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*2
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2

    if cfg.slurm_array_id == 6:
        cfg.name = cfg.name + "_8lr_clip1"
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*4
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*4
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/G_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/D_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/state_60000.pth"
        cfg.load_model_from_save = True
        cfg.training.resume_training_from_save = True

    if cfg.slurm_array_id == 7:
        cfg.name = cfg.name + "_8lr_clipInf"
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*4
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*4
        cfg.generator.max_norm = 10**10
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/G_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/D_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/state_60000.pth"
        cfg.load_model_from_save = True
        cfg.training.resume_training_from_save = True

    if cfg.slurm_array_id == 8: #stopped
        cfg.name = cfg.name + "_16lr_clipInf"
        cfg.generator.max_norm = 10**10
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*8
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2
    if cfg.slurm_array_id == 9: #stopped
        cfg.name = cfg.name + "_16lr_clipInf_10adv"
        cfg.generator.max_norm = 10**10
        cfg.training.adversarial_loss_weight = cfg.training.adversarial_loss_weight*10
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*8
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2

    if cfg.slurm_array_id == 10: 
        cfg.name = cfg.name + "_16lr_clip1"
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*8
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/G_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/D_60000.pth"
        cfg.env.discriminator_load_path = "./runs/"+cfg.name+"/state_60000.pth"
        cfg.load_model_from_save = True
        cfg.training.resume_training_from_save = True

    if cfg.slurm_array_id == 11: #stopped
        cfg.name = cfg.name + "_4lr_clipInf_10adv"
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*2
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2
        cfg.training.adversarial_loss_weight = cfg.training.adversarial_loss_weight*10
        cfg.generator.max_norm = 10**10
    if cfg.slurm_array_id == 12:
        cfg.name = cfg.name + "_4lr_clip1_10adv"
        cfg.training.adversarial_loss_weight = cfg.training.adversarial_loss_weight*10
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*2
        cfg.training.learning_rate_d = cfg.training.learning_rate_d*2
    if cfg.slurm_array_id == 13:
        cfg.name = cfg.name + "_4lr_clip1_4adv_D2"
        cfg.training.adversarial_loss_weight = cfg.training.adversarial_loss_weight*4
        cfg.training.learning_rate_g = cfg.training.learning_rate_g*2



        
  

    setup_ok: bool = safe_setup_env_and_cfg(cfg)
    if not setup_ok:
        print("Aborting")
        return

    setup_torch(cfg)

    if cfg.is_use:
        test.test(cfg)

    save_config(cfg, cfg.env.this_runs_folder)

    setup_logger(cfg)
    status_logger = logging.getLogger("status")
    status_logger.info(f"run.py: initialized with config:\n\n{cfg}")

    status_logger.info(f"run.py: running with device: {cfg.device}")

    if cfg.is_download:
        start_date = date(
            cfg.gan_config.start_date[0],
            cfg.gan_config.start_date[1],
            cfg.gan_config.start_date[2],
        )

        end_date = date(
            cfg.gan_config.end_date[0],
            cfg.gan_config.end_date[1],
            cfg.gan_config.end_date[2],
        )
        status_logger.info(
            "run.py: starting download of all data for Bessaker Wind farm from "
            + str(start_date)
            + " to "
            + str(end_date)
            + " not previously downloaded."
        )

    dataset_train, dataset_test, dataset_validation, x, y = prepare_data(cfg)

    status_logger.info(f"run.py: data prepared")
    
    if cfg.is_param_search:
        cfg.is_train = True
        status_logger.info("run.py: starting parameter search")
        param_search(num_samples=250, number_of_GPUs=cfg.slurm_array_id, cfg=cfg, dataset_train=dataset_train, dataset_validation=dataset_validation, x=x, y=y)
        status_logger.info("run.py: finished parameter search")
        return
    
    if cfg.is_train:
        status_logger.info(
            "run.py: starting training" + ("" if not cfg.is_test else " before testing")
        )
        train.train(cfg, dataset_train, dataset_validation, x, y)
        status_logger.info("run.py: finished training")
        cfg.is_train = False

    if cfg.is_test:
        status_logger.info("run.py: starting testing")
        test.test(cfg, dataset_test)
        status_logger.info("run.py: finished testing")

    status_logger.info(
        f"run.py: log file location: {cfg.env.status_log_file}  run file location: {cfg.env.train_log_file}"
    )


def argv_to_cfg() -> Config:
    parser = argparse.ArgumentParser(
        description="Set config, and set if we're doing training or testing."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        # default="config/wind_field_GAN_3D_config_cluster_short.ini",
        default="config/wind_field_GAN_3D_config_local.ini",
        help="path to config ini file (defaults to config/wind_field_GAN_3D_config_local.ini)",
    )
    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="run training with supplied config",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="run tests with supplied config",
    )
    parser.add_argument(
        "--param_search",
        default=False,
        action="store_true",
        help="run tests with supplied config",
    )
    
    parser.add_argument(
        "--use", default=False, action="store_true", help="use on LR images"
    )
    parser.add_argument(
        "--download",
        default=False,
        action="store_true",
        help="Only downloads data, does not train or test",
    )

    parser.add_argument(
        "--loglevel",
        default=False,
        action="store_true",
        help="run tests with supplied config",
    )

    parser.add_argument(
        "--slurm_array_id",
        type=int,
        default=1,
        help="ID for slurm job",
    )

    args = parser.parse_args()
    is_test = args.test
    is_train = args.train
    is_use = args.use
    is_download = args.download
    is_param_search = args.param_search
    cfg_path = args.cfg
    slurm_array_id = args.slurm_array_id

    if is_use:
        cfg_path = (
            os.path.dirname(os.path.realpath(__file__)) + "/config/config_use.ini"
        )
        print(cfg_path)

    cfg = Config(cfg_path)
    cfg.is_test = is_test
    cfg.is_use = is_use
    cfg.is_train = is_train
    cfg.is_download = is_download
    cfg.is_param_search = is_param_search
    cfg.slurm_array_id = slurm_array_id

    return cfg


def safe_setup_env_and_cfg(cfg: Config) -> bool:
    # store some useful paths in the cfg
    cfg.env.log_folder = cfg.env.root_path + cfg.env.log_subpath
    cfg.env.tensorboard_log_folder = cfg.env.root_path + cfg.env.tensorboard_subpath
    cfg.env.status_log_file = cfg.env.log_folder + "/" + cfg.name + ".log"
    cfg.env.this_runs_folder = cfg.env.root_path + cfg.env.runs_subpath + "/" + cfg.name
    cfg.env.this_runs_tensorboard_log_folder = (
        cfg.env.tensorboard_log_folder + "/" + cfg.name
    )
    cfg.env.train_log_file = cfg.env.this_runs_folder + "/" + cfg.name + ".train"

    # make necessary paths, but warn user if the run folder overlaps with existing folder.
    makedirs(cfg.env.log_folder)
    makedirs(cfg.env.tensorboard_log_folder)
    [
        makedirs(path)
        for path in [
            "./data/downloaded_raw_bessaker_data",
            "./data/full_dataset_files",
            "./data/interpolated_z_data",
        ]
    ]
    is_ok = makedirs_ensure_user_ok(cfg.env.this_runs_folder)
    makedirs(cfg.env.this_runs_folder + "/images")
    makedirs(cfg.env.this_runs_tensorboard_log_folder)
    setup_seed(cfg.env.fixed_seed)
    return is_ok


def setup_logger(cfg: Config):
    # root logger for basic messages
    timestamp = str(int(time.time()))

    root_logger = logging.getLogger("status")
    root_logger.setLevel(logging.DEBUG)
    root_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s: %(message)s"
    )

    if cfg.is_train:
        log_handler = logging.FileHandler(cfg.env.status_log_file, mode="a")
        log_handler.setFormatter(root_formatter)
        log_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(log_handler)

        # train logger for logging losses during training
        train_logger = logging.getLogger("train")
        train_logger.setLevel(logging.INFO)
        train_formatter = logging.Formatter("%(message)s")
        train_handler = logging.FileHandler(cfg.env.train_log_file, mode="a")
        train_logger.addHandler(train_handler)
        train_logger.info("Initialized train logger")

    if cfg.also_log_to_terminal:
        term_handler = logging.StreamHandler()
        term_handler.setFormatter(root_formatter)
        term_handler.setLevel(logging.INFO)
        root_logger.addHandler(term_handler)

    root_logger.info("Initialized status logger")

    return


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_torch(cfg: Config):
    # device="mps" if torch.backends.mps.is_available() else "cpu"
    cfg.device = (
        torch.device(f"cuda:{cfg.gpu_id}")
        if torch.cuda.is_available() and cfg.gpu_id is not None
        else torch.device("cpu")
    )


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def makedirs_ensure_user_ok(path) -> bool:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return True
    else:
        print(
            f"Folder {path} exists.\nAre you sure you want to run with the same run name? Files may be overwritten. [Y/n]"
        )
        return get_yes_or_no_input()


def get_yes_or_no_input() -> bool:
    # courtesy of https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    yes = {"yes", "y", "ye", ""}
    no = {"no", "n"}
    #    ans = None
    #    while True:
    #        choice = input("> ").lower()
    #        if choice in yes:
    #           return True
    #        elif choice in no:
    #           return False
    #        else:
    #           print("Please respond with 'yes' or 'no'")
    return True  # FIX


def save_config(cfg: Config, folder: str):
    filename = folder + "/config.ini"
    if cfg.env.discriminator_load_path == None:
        cfg.env.discriminator_load_path = folder + "/D_"+str(cfg.training.niter)+".pth"
        cfg.env.generator_load_path = folder + "/G_"+str(cfg.training.niter)+".pth"
        cfg.env.state_load_path = folder + "/state_"+str(cfg.training.niter)+".pth"
    with open(filename, "w") as ini:
        ini.write(cfg.asINI())


def prepare_data(cfg: Config):
    cfg_gan = cfg.gan_config
    Z_DICT = {"start": 0, "max": cfg_gan.number_of_z_layers, "step": 1}
    start_date = date(
        cfg_gan.start_date[0], cfg_gan.start_date[1], cfg_gan.start_date[2]
    )
    end_date = date(cfg_gan.end_date[0], cfg_gan.end_date[1], cfg_gan.end_date[2])

    return preprosess(
        Z_DICT=Z_DICT,
        start_date=start_date,
        end_date=end_date,
        include_pressure=cfg_gan.include_pressure,
        include_z_channel=cfg_gan.include_z_channel,
        interpolate_z=cfg_gan.interpolate_z,
        enable_slicing=cfg_gan.enable_slicing,
        slice_size=cfg_gan.slice_size,
        include_above_ground_channel=cfg_gan.include_above_ground_channel,
        train_aug_rot=cfg.dataset_train.data_aug_rot,
        train_aug_flip=cfg.dataset_train.data_aug_flip,
        val_aug_rot=cfg.dataset_val.data_aug_rot,
        val_aug_flip=cfg.dataset_val.data_aug_flip,
        test_aug_rot=cfg.dataset_test.data_aug_rot,
        test_aug_flip=cfg.dataset_test.data_aug_flip,
        train_eval_test_ratio=cfg.training.train_eval_test_ratio,
    )


if __name__ == "__main__":
    main()
