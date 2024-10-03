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
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Restoring AISTD with DeS3")
    parser.add_argument(
        "--config", type=str, default="AISTDshadow.yml", help="Path to the config file"
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
        "--sampling_timesteps",
        type=int,
        default=25,
        help="Number of implicit sampling steps",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="AISTD",
        help="restoration test set options: ['SRD', 'AISTD', 'LRSS', 'UIUC']",
    )
    parser.add_argument(
        "--image_folder",
        default="test_results/",
        type=str,
        help="Location to save restored images",
    )
    parser.add_argument(
        "--seed",
        default=61,
        type=int,
        metavar="N",
        help="Seed for initializing training (default: 61)",
    )
    parser.add_argument(
        "--log_dir",
        default="/home/satviktyagi/Desktop/desk/project/github/des3_mine/exp",
        type=str,
        help="log directory",
    )
    parser.add_argument(
        "--exp_name",
        default="eval_aistd",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--num_last_weights",
        default=3,
        type=int,
        help="number of last weights to keep ex => default 3 means we're keeping last 3 weights in dir",
    )
    parser.add_argument(
        "--save_after_epoch",
        default=10,
        type=int,
        help="Save Checkpoint after N Epochs",
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
    parser.add_argument(
        "--prefix",
        default=None,
        type=str,
        help="name of folder to save the test images",
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

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    config.device = device

    if torch.cuda.is_available():
        print(
            "Note: Currently supports evaluations (restoration) when run only on a single GPU!"
        )

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"args.seed {args.seed}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print(f"Using dataset {config.data.dataset}")
    DATASET = datasets.__dict__[config.data.dataset](args, config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    # create model
    print("Creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config, run="test")
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(val_loader, validation=args.test_set, r=args.grid_r, sid=args.sid)


if __name__ == "__main__":
    main()
