import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
from inference.utilities import *
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Restoring AISTD with DeS3")
    parser.add_argument(
        "--config", type=str, default="AISTDshadow.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--log_dir",
        default="inference_results/",
        type=str,
        help="log directory",
    )
    parser.add_argument(
        "--resume",
        default="/home1/yeying/DeS3_Deshadow/ckpts/AISTDShadow_ddpm.pth.tar",
        type=str,
        help="Path for the diffusion model checkpoint to load for evaluation",
    )
    parser.add_argument(
        "--grid_r",
        type=int,
        default=16,
        help="Grid cell width r that defines the overlap between patches",
    )
    parser.add_argument(
        "--exp_name",
        default="eval_aistd",
        type=str,
        help="experiment name",
    )

    parser.add_argument("--sid", type=str, default=None)
    args = parser.parse_args()
    print(args)

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
    config = init_experiment(args, config)

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    config.device = device

    if torch.cuda.is_available():
        print(
            "Note: Currently supports evaluations (restoration) when run only on a single GPU!"
        )

    # set random seed
    torch.manual_seed(config.misc.seed)
    np.random.seed(config.misc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.misc.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print(f"Using dataset {config.data.dataset}")
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False)

    # create model
    print("Creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(config)
    model = DiffusiveRestoration(diffusion, config)
    model.restore(val_loader, r=config.grid_r, sid=config.sid)


if __name__ == "__main__":
    main()
