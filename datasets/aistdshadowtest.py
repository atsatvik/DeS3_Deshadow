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
from logchromaticity import processLogImage
from imageutils import Image

file_ext = ["png", "jpg"]


class AISTDShadowTest:
    def __init__(self, args, config):
        self.config = config
        self.args = args
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        self.logger = logging.getLogger(__name__)
        self.guidance_type = self.config.guidance_type

    def get_loaders(self, parse_patches=True, validation="AISTD"):
        self.logger.info("Evaluating AISTD test set...")
        val_dataset = AISTDShadowDataset(
            dir=os.path.join(self.config.data.data_dir),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=parse_patches,
            folders=["test_A", "test_C", "test_B"],
            label="test",
            input_type=self.args.input_type,
            use_class=self.args.use_class,
            guidance_type=self.guidance_type,
        )

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        return None, val_loader


class AISTDShadowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir,
        patch_size,
        n,
        transforms,
        parse_patches=True,
        folders=[],
        label="",
        input_type="sf",
        use_class=False,
        guidance_type="1",
    ):
        self.processlog = processLogImage
        self.Image_ = Image
        self.mask_paths = None
        self.loggray_paths = None

        self.guidance_type = guidance_type
        if input_type not in ["sf/s", "sf-s", "sf"]:
            raise Exception("Incorrect input_type choose from: <sf/s>, <sf-s>, <sf>")

        self.counter = 0
        self.input_type = input_type
        self.use_class = use_class
        self.epsilon = 1e-8
        self.logger = logging.getLogger(__name__)
        super().__init__()
        self.logger.info(f"Directory: {dir}")

        if not folders:
            raise Exception("Incorrect or missing images and GT folders")
        folders = [os.path.join(dir, label, f) for f in folders]
        image_folder, GT_folder = (folders[0], folders[1])

        img_names = [
            f for f in os.listdir(image_folder) if f.split(".")[-1] in file_ext
        ]
        GT_names = [f for f in os.listdir(GT_folder) if f.split(".")[-1] in file_ext]

        image_paths = [os.path.join(image_folder, f) for f in img_names]
        GT_paths = [os.path.join(GT_folder, f) for f in GT_names]

        x = list(enumerate(image_paths))
        random.shuffle(x)
        indices, image_paths = zip(*x)
        GT_paths = [GT_paths[idx] for idx in indices]
        self.dir = None

        self.image_paths = image_paths
        self.GT_paths = GT_paths
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

        if self.use_class:
            mask_folder = folders[2]
            mask_names = [
                f for f in os.listdir(mask_folder) if f.split(".")[-1] in file_ext
            ]
            mask_paths = [os.path.join(mask_folder, f) for f in mask_names]
            mask_paths = [mask_paths[idx] for idx in indices]
            self.mask_paths = mask_paths

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
        return tuple(crops)

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

        if self.guidance_type in ["2", "3", "4"]:
            input_img = np.array(input_img)
            rgb_img = input_img.astype(np.uint8)
            loggray_img = self.processlog.getGrayImagebyPointSelection(rgb_img)
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
        # print(input_img.shape,gt_img.shape)
        return (
            torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0),
            img_id,
        )

    def save_debug_img(self, img, imgname):
        img.save(
            f"/home/satviktyagi/Desktop/desk/project/github/DeS3_Deshadow/debug_imgs/{imgname}"
        )

    def show_img(self, img, name="img", destroy=False):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        if destroy:
            cv2.destroyAllWindows()

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.image_paths)
