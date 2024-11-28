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
from utils import *


file_ext = ["png", "jpg"]
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class AISTDShadow:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        self.logger = logging.getLogger(__name__)

    def get_loaders(self, parse_patches=True, validation="AISTD"):
        self.logger.info("Evaluating AISTD test set...")
        train_dataset = AISTDShadowDataset(
            self.config,
            transforms=self.transforms,
            parse_patches=parse_patches,
            label="train",
        )

        val_dataset = AISTDShadowDataset(
            self.config,
            transforms=self.transforms,
            parse_patches=parse_patches,
            label="test",
        )

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader


class AISTDShadowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        transforms,
        parse_patches=True,
        label="",
    ):
        self.label = label
        self.config = config
        self.populatefromConfig()

        self.mask_paths = None
        self.loggray_paths = None
        self.rgb_reproj_paths = None

        self.counter = 0
        self.epsilon = 1e-8
        self.logger = logging.getLogger(__name__)
        super().__init__()
        self.logger.info(f"Directory: {self.directory}")

        imgpath = os.path.join(self.directory, "rgb_data/" + label + "/input")
        gtpath = os.path.join(self.directory, "rgb_data/" + label + "/gt")

        self.image_paths = getPaths(imgpath)
        self.GT_paths = getPaths(gtpath)

        checksanity(self.image_paths, self.GT_paths, self.logger)

        self.transforms = transforms
        self.parse_patches = parse_patches

        # if self.use_class:
        #     # self.mask_paths = getPaths(folders[2])
        #     exit()

        self.data_dict = parse_file(self.ISD_file_path)

    def populatefromConfig(self):
        self.directory = os.path.join(self.config.data.data_dir)
        self.n = self.config.training.patch_n
        self.patch_size = self.config.data.image_size
        self.input_type = self.config.misc.input_type
        self.use_class = self.config.misc.use_class
        self.guidance_type = self.config.misc.guidance_type
        self.ISD_file_path = os.path.join(
            self.config.data.data_dir, f"ISD_vec_{self.label}.txt"
        )
        self.apply_clahe = self.config.misc.apply_clahe
        self.factor1 = self.config.misc.factor1
        self.factor2 = self.config.misc.factor2

    def getInputBasedOnGuidance(self, input_img, input_name, guidance_type):
        input_img = np.array(input_img, dtype=np.float32)
        logpth = input_name.replace("rgb_data", "log_data")
        logpth = logpth.replace(".png", ".exr")
        log_image = cv2.imread(logpth, cv2.IMREAD_UNCHANGED)
        input_name = os.path.basename(logpth).split(".")[0]

        if guidance_type in ["2", "3", "4"]:
            loggray_img = fetchGrayImg(log_image, self.data_dict[input_name])
            if self.guidance_type == "2":
                input_img = np.concatenate(
                    [input_img, loggray_img[:, :, np.newaxis]], axis=2
                )
            elif self.guidance_type == "3":
                input_img = loggray_img
            elif self.guidance_type == "4":
                input_img = (
                    self.factor1 * input_img
                    + self.factor2 * loggray_img[:, :, np.newaxis]
                )

        elif guidance_type == "5":
            input_img = fetchReprojectedColorImg(
                log_image, self.data_dict[input_name], input_name, self.config
            )

        input_img = input_img.astype(np.uint8)
        # cv2.imshow("input_img", input_img)
        # cv2.imshow("loggray_img", loggray_img.astype(np.uint8))
        # cv2.imshow("gt_img", np.array(self.gt_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        input_img = PIL.Image.fromarray(input_img)
        return input_img

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return list(crops)

    def get_images(self, index):
        input_name = self.image_paths[index]
        gt_name = self.GT_paths[index]
        loggray_img = None
        labels = []
        # print('input_name,gt_name',input_name,gt_name)
        datasetname = re.split("/", input_name)[-3]
        # print('datasetname',datasetname)
        img_id = re.split("/", input_name)[-1][:-4]
        # print('img_id',img_id)
        input_img = PIL.Image.open(input_name)
        gt_img = PIL.Image.open(gt_name)

        input_img = self.getInputBasedOnGuidance(
            input_img, input_name, self.guidance_type
        )

        if self.parse_patches:
            wd_new = 512
            ht_new = 512
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            i, j, h, w = self.get_params(
                input_img, (self.patch_size, self.patch_size), self.n
            )
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)

            if self.input_type != "sf":
                gt_img = self.prepare_GT(input_img, gt_img, self.input_type)

            if self.use_class:
                mask_name = self.mask_paths[index]
                mask_img = PIL.Image.open(mask_name).convert("L")
                mask_img = mask_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
                mask_img = self.n_random_crops(mask_img, i, j, h, w)
                labels = self.prepare_labels(input_img, gt_img, mask_img)

            # self.counter += 1
            # exit()
            outputs = [
                torch.cat(
                    [self.transforms(input_img[i]), self.transforms(gt_img[i])],
                    dim=0,
                )
                for i in range(self.n)
            ]  ## range is from 0 to 1 here

            # print("labels ", labels)input_channels
            return torch.stack(outputs, dim=0), img_id, labels
        else:
            wd_new = 256
            ht_new = 256
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # print(input_img.shape,gt_img.shape)

            return (
                torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0),
                img_id,
            )

    def prepare_labels(self, input_img, gt_img, mask_img):
        labels = []
        for i, (img_mask) in enumerate(mask_img):
            if np.any(np.array(img_mask) > 0):
                ip_lb = 1
            else:
                ip_lb = 0
            gt_lb = 0
            labels.append((ip_lb, gt_lb))
        return torch.tensor(labels)

    def prepare_GT(self, input_img, gt_img, input_type):
        gt_img = list(gt_img)
        for i, (img_ip, img_gt) in enumerate(zip(input_img, gt_img)):
            img_ip = np.array(img_ip).astype(np.float32)
            img_gt = np.array(img_gt).astype(np.float32)
            if input_type == "sf/s":
                altered_img = img_gt / (
                    img_ip + self.epsilon
                )  # To avoid invalid values
            elif input_type == "sf-s":
                altered_img = img_gt - img_ip
            altered_img = cv2.normalize(altered_img, None, 0, 1, cv2.NORM_MINMAX)
            gt_img[i] = altered_img

        return tuple(gt_img)

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.image_paths)
