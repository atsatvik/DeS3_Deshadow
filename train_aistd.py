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
        default="/home1/yeying/DeS3_Deshadow/ckpts/AISTDShadow_ddpm.pth.tar",
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
        default=None,
        type=str,
        help="Which input to use among -> sf (shadow free img), sf/s (shadow free img / shadow img), sf-s (shadow free - shadow)",
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](args, config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
