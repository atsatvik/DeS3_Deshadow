import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
import cv2
import logging
from inference.logchromaticity import processLogImage

file_ext = ["png", "jpg"]
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class AISTDShadowDataset:
    def __init__(
        self,
        args,
        config,
        folders=["test_A", "test_C"],
        label="test",
    ):
        self.config = config
        self.args = args
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        self.logger = logging.getLogger(__name__)
        self.guidance_type = self.config.guidance_type
        self.folders = folders

        self.processlog = processLogImage()
        self.loggray_paths = None
        self.processed_indexes = set()
        self.label = label

    def populate_paths(self, random_shuffle=True):
        directory = os.path.join(self.config.data.data_dir)
        self.logger.info(f"Directory: {directory}")

        if not self.folders:
            raise Exception("Incorrect or missing images and GT folders")
        self.folders = [os.path.join(directory, self.label, f) for f in self.folders]
        image_folder, GT_folder = (self.folders[0], self.folders[1])

        img_names = [
            f for f in os.listdir(image_folder) if f.split(".")[-1] in file_ext
        ]
        GT_names = [f for f in os.listdir(GT_folder) if f.split(".")[-1] in file_ext]

        image_paths = [os.path.join(image_folder, f) for f in img_names]
        GT_paths = [os.path.join(GT_folder, f) for f in GT_names]

        x = list(enumerate(image_paths))
        if random_shuffle:
            random.shuffle(x)
        self.indices, image_paths = zip(*x)
        GT_paths = [GT_paths[idx] for idx in self.indices]

        self.image_paths = image_paths
        self.GT_paths = GT_paths
        self.indices = list(self.indices)

    def generate_index(self):
        return self.indices.pop() if self.indices else None

    def get_image(self, index):
        input_name = self.image_paths[index]
        gt_name = self.GT_paths[index]
        loggray_img = None
        labels = []
        datasetname = re.split("/", input_name)[-3]
        img_id = re.split("/", input_name)[-1][:-4]
        input_img = PIL.Image.open(input_name)
        gt_img = PIL.Image.open(gt_name)

        if self.guidance_type in ["2", "3", "4"]:
            input_img = np.array(input_img)
            rgb_img = input_img.astype(np.uint8)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            log_img = cv2.imread(
                os.path.join(self.args.logimgpath, f"{img_id}.exr"),
                cv2.IMREAD_UNCHANGED,
            )
            loggray_img = self.processlog.getGrayImagebyPointSelection(
                log_img, rgb_img, num_points=2, title=img_id
            )
            if self.guidance_type == "2":
                input_img = PIL.Image.fromarray(
                    np.concatenate([input_img, loggray_img[:, :, np.newaxis]], axis=2)
                )
            elif self.guidance_type == "3":
                input_img = PIL.Image.fromarray(loggray_img)

        wd_new = 256
        ht_new = 256
        input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
        return (
            torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0),
            img_id,
        )
