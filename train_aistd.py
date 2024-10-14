import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion
import logging
import sys
import shutil


def init_logger(path=""):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(path, "train_logs.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def prepare_log_dir(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    exp_dir = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_dir_folder_ls = os.listdir(exp_dir)
    if not exp_dir_folder_ls:
        exp_log_dir = os.path.join(exp_dir, f"{0}")
        os.makedirs(exp_log_dir)
    else:
        ls = []
        for i in range(len(exp_dir_folder_ls)):
            try:
                ls.append(int(exp_dir_folder_ls[i]))
            except:
                continue
        exp_dir_folder_ls = ls
        exp_dir_folder_ls.sort()
        exp_log_dir = os.path.join(exp_dir, f"{int(exp_dir_folder_ls[-1]) + 1}")
        os.makedirs(exp_log_dir)

    config_file_path = os.path.join("configs", args.config)
    shutil.copy(config_file_path, os.path.join(exp_log_dir, args.config))
    return exp_log_dir


def set_guidance_type(logger):
    while True:
        logger.info(f"Choose guidance image type: input 1, 2, 3, 4")
        logger.info(f"1: RGB (shadow image)")
        logger.info(f"2: RGB concatenated with log gray (shadow image)")
        logger.info(f"3: Log gray")
        logger.info(f"4: Reprojected RGB concatenated wit log gray")
        guidance_type = input()
        if guidance_type not in ["1", "2", "3", "4"]:
            logger.info("Please choose from: 1, 2, 3, 4")
        else:
            break
    return guidance_type


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Training DeS3")
    parser.add_argument(
        "--exp_name",
        default="train_aistd",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--log_dir",
        default="/home/satviktyagi/Desktop/desk/project/github/des3_mine/exp",
        type=str,
        help="log directory",
    )
    parser.add_argument(
        "--num_last_weights",
        default=3,
        type=int,
        help="number of last weights to keep ex => default 3 means we're keeping last 3 weights in dir",
    )
    parser.add_argument(
        "--save_after_epoch",
        default=50,
        type=int,
        help="Save Checkpoint after N Epochs",
    )
    parser.add_argument(
        "--config", type=str, default="AISTDshadow.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Path for checkpoint to load and resume",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="AISTD",
        help="restoration test set options: ['SRD', 'AISTD', 'LRSS', 'UIUC']",
    )
    parser.add_argument(
        "--sampling_timesteps",
        type=int,
        default=25,
        help="Number of implicit sampling steps for validation image patches",
    )
    parser.add_argument(
        "--image_folder",
        default="results/images/",
        type=str,
        help="Location to save restored validation image patches",
    )
    parser.add_argument(
        "--seed",
        default=61,
        type=int,
        metavar="N",
        help="Seed for initializing training (default: 61)",
    )
    parser.add_argument(
        "--input_type",
        default="sf",
        type=str,
        help="Which input to use among -> sf (shadow free img), sf/s (shadow free img / shadow img), sf-s (shadow free - shadow)",
    )
    parser.add_argument(
        "--use_class",
        default=False,
        type=bool,
        help="Wether to use class embeddings or not",
    )
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logger = logging.getLogger(__name__)

    exp_log_dir = prepare_log_dir(args)
    init_logger(exp_log_dir)

    guidance_type = set_guidance_type(logger)
    config.guidance_type = guidance_type

    logger.info(f"Saving log files to dir:{exp_log_dir}")

    # Logging all args
    logger.info("\n=========================================")
    logger.info(f"Experiment Settings:")
    string = ""
    for arg, value in vars(args).items():
        string += f"({arg}: {value}) ; "
    logger.info(string[0:-2])
    logger.info("=========================================\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    logger.info(f"Using dataset {config.data.dataset}")
    DATASET = datasets.__dict__[config.data.dataset](args, config)

    # create model
    logger.info("Creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config, exp_log_dir)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
