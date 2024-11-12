import os
import shutil
import logging
import argparse
import sys
import cv2


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


def init_experiment(args, config):
    exp_log_dir = prepare_log_dir(args)
    args.log_dir = exp_log_dir

    for arg, value in vars(args).items():
        setattr(config, arg, value)

    logger = logging.getLogger(__name__)
    init_logger(path=config.log_dir)
    logger.info(f"Saving log files to dir: {config.log_dir}")

    logger.info("\n=========================================")
    logger.info("Experiment Settings:")
    string = ""
    for arg, value in vars(config).items():
        string += f"{arg}: {value}\n"

    logger.info(string[0:-2])
    logger.info("=========================================\n")

    return config


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(gray_image)
    return clahe_image
