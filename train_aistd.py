import argparse
import os
import random
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import models
import datasets
from utils import *
from models import DenoisingDiffusion
import logging
import sys
import shutil


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Training DeS3")
    parser.add_argument(
        "--exp_name",
        default="train_aistd",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="AISTDshadow.yml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--log_dir",
        default="experiments/",
        type=str,
        help="log directory",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Path for checkpoint to load and resume",
    )
    parser.add_argument(
        "--image_folder",
        default="results/images/",
        type=str,
        help="Location to save restored validation image patches",
    )
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return args, config


def main():
    args, config = parse_args_and_config()
    config = init_experiment(args, config)

    logger = logging.getLogger(__name__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    config.device = device

    # set random seed
    torch.manual_seed(config.misc.seed)
    np.random.seed(config.misc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.misc.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    logger.info(f"Using dataset {config.data.dataset}")
    DATASET = datasets.__dict__[config.data.dataset](config)
    train_loader, val_loader = DATASET.get_loaders()

    # create model
    logger.info("Creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(config)
    diffusion.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
