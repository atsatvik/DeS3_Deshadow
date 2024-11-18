import os
import shutil
import logging
import argparse
import sys
import cv2
import numpy as np
import re


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


def parse_file(file_path):
    data_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            match = re.match(
                r"(\S+)\s+lit_pt, shadow_pt\s+:\s+\((\d+),\s*(\d+)\)\s+\((\d+),\s*(\d+)\)",
                line,
            )
            if match:
                # Extract the image name and points
                image_name = match.group(1)
                lit_pt = (int(match.group(2)), int(match.group(3)))
                shadow_pt = (int(match.group(4)), int(match.group(5)))

                # Store in the dictionary
                data_dict[image_name] = [lit_pt, shadow_pt]
    return data_dict


def getPaths(folder):
    names = [f for f in os.listdir(folder) if f.split(".")[-1] in ["png"]]
    paths = [os.path.join(folder, f) for f in names]
    paths.sort()
    return paths


def checksanity(input_paths, target_paths, logger):
    if len(input_paths) != len(input_paths):
        raise Exception("images and GT are inconsistent")
    input_paths = set([os.path.basename(pth) for pth in input_paths])
    target_paths = set([os.path.basename(pth) for pth in target_paths])
    for name in input_paths:
        if name not in target_paths:
            raise Exception("images and GT are inconsistent")
    logger.info("Input and GT folders are consistent")


def getPlane(normal_vector, point_on_plane):
    normal_vector = normal_vector.astype(np.float64)
    point_on_plane = point_on_plane.astype(np.float64)
    a, b, c = normal_vector
    x0, y0, z0 = point_on_plane
    d = -(a * x0 + b * y0 + c * z0)
    return a, b, c, d


def estimate3DTransformationMat(a, b, c, d):
    normal = np.array([a, b, c])
    normal_norm = normal / np.linalg.norm(normal)  # Normalize

    if a != 0:
        point_on_plane = np.array([-d / a, 0, 0])
    elif b != 0:
        point_on_plane = np.array([0, -d / b, 0])
    elif c != 0:
        point_on_plane = np.array([0, 0, -d / c])
    else:
        raise ValueError("Invalid plane equation coefficients.")

    target = np.array([0, 0, 1])

    v = np.cross(normal_norm, target)
    s = np.linalg.norm(v)
    c = np.dot(normal_norm, target)

    if s != 0:  # Handle the case where the vectors are already aligned
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + np.dot(K, K) * ((1 - c) / (s**2))
    else:
        R = np.eye(3)  # No rotation needed

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = -np.dot(R, point_on_plane)

    new_normal = np.dot(R, normal)
    plane_pt = np.append(point_on_plane, 1)
    plane_pt = np.dot(transformation_matrix, plane_pt)[:3]
    new_d = -np.dot(new_normal, plane_pt)

    return transformation_matrix, (*new_normal, new_d)


def transform3DPoints(array_3D, transformation_matrix):
    points = array_3D.reshape(-1, 3)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    transformed_homogeneous = np.dot(transformation_matrix, points_homogeneous.T).T

    transformed_points = (
        transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, np.newaxis]
    )

    return transformed_points.reshape(array_3D.shape)


def estimateTransformationMat(normal_vector):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    arbitrary_vector = (
        np.array([1, 0, 0])
        if not np.allclose(normal_vector, [1, 0, 0])
        else np.array([0, 1, 0])
    )
    v1 = arbitrary_vector - np.dot(arbitrary_vector, normal_vector) * normal_vector
    v1 = v1 / np.linalg.norm(v1)  # Normalize v1

    v2 = np.cross(normal_vector, v1)
    v2 = v2 / np.linalg.norm(v2)  # Normalize v2

    transformation_matrix = np.vstack([v1, v2])  # Stack v1 and v2 into a 2x3 matrix
    return transformation_matrix


def transformPoints(image, tf_mat):
    image = image.reshape(-1, 3)
    projected_points = np.dot(image, tf_mat.T)
    return projected_points


def getXYmap(log_img, ISD):
    tf_mat = estimateTransformationMat(ISD)
    projected_pts = transformPoints(log_img, tf_mat)
    return projected_pts.reshape((log_img.shape[0], log_img.shape[1], 2))


def normalize(xymap):
    min_val = np.min(xymap, axis=(0, 1), keepdims=True)
    max_val = np.max(xymap, axis=(0, 1), keepdims=True)
    norm_img = ((xymap - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
    norm_img = np.clip(norm_img, a_min=0, a_max=1)
    return norm_img


def fetchGrayImg(log_img, points):
    lit_shadow_pts = [
        log_img[points[-2][0], points[-2][1], :],
        log_img[points[-1][0], points[-1][1], :],
    ]
    ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]

    a, b, c, d = getPlane(ISD_vec, lit_shadow_pts[0])
    transformation_matrix, new_plane = estimate3DTransformationMat(a, b, c, d)
    ISD_vec_transformed = new_plane[:3]

    log_img_transformed = transform3DPoints(log_img, transformation_matrix)
    xymap = getXYmap(log_img_transformed, ISD_vec_transformed)
    xymap = normalize(xymap) * 255
    return np.mean(xymap, axis=2).astype(np.float32)
