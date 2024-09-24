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

file_ext = ["png"]


class AISTDShadow:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

    def get_loaders(self, parse_patches=True, validation="AISTD"):
        print("=> evaluating AISTD test set...")
        train_dataset = AISTDShadowDataset(
            dir=os.path.join(self.config.data.data_dir),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=parse_patches,
            folders=["train_A", "train_C"],
            label="train",
        )

        val_dataset = AISTDShadowDataset(
            dir=os.path.join(self.config.data.data_dir),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=parse_patches,
            folders=["test_A", "test_C"],
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
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader


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
    ):
        self.counter = 0
        super().__init__()
        print(f"Directory: {dir}")

        if not folders:
            raise Exception("Incorrect or missing images and GT folders")
        folders = [os.path.join(dir, label, f) for f in folders]
        image_folder, GT_folder = folders[0], folders[1]

        img_names = [
            f for f in os.listdir(image_folder) if f.split(".")[-1] in file_ext
        ]
        GT_names = [f for f in os.listdir(GT_folder) if f.split(".")[-1] in file_ext]

        if not img_names == GT_names:
            raise Exception("images and GT have inconsistency")

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
        # print('input_name,gt_name',input_name,gt_name)
        datasetname = re.split("/", input_name)[-3]
        # print('datasetname',datasetname)
        img_id = re.split("/", input_name)[-1][:-4]
        # print('img_id',img_id)
        input_img = PIL.Image.open(input_name)
        gt_img = PIL.Image.open(gt_name)

        if self.parse_patches:
            wd_new = 512
            ht_new = 512
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # print('-input_img.shape,gt_img.shape-',input_img.size,gt_img.size)
            i, j, h, w = self.get_params(
                input_img, (self.patch_size, self.patch_size), self.n
            )
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)

            label_ip = []
            label_gt = []
            for i, (img_ip, img_gt) in enumerate(zip(input_img, gt_img)):
                label_gt.append(0)

                ###DEBUG########################################
                # imgname_ip = f"{self.counter}_{i}_ip.png"
                # imgname_gt = f"{self.counter}_{i}_gt.png"
                # self.save_debug_img(img_ip, imgname_ip)
                # self.save_debug_img(img_gt, imgname_gt)
                ###DEBUG########################################

                if self.calculate_mse(img_ip, img_gt) < 20:
                    label_ip.append(0)
                else:
                    label_ip.append(1)
            # self.counter += 1
            # exit()
            outputs = [
                torch.cat(
                    [self.transforms(input_img[i]), self.transforms(gt_img[i])],
                    dim=0,
                )
                for i in range(self.n)
            ]
            label = label_ip + label_gt
            label = torch.tensor(label)
            # print("labellllll, ", label)
            return torch.stack(outputs, dim=0), img_id, label
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

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.image_paths)

    def calculate_mse(self, image1, image2):
        img1 = np.array(image1)
        img2 = np.array(image2)
        return np.mean((img1 - img2) ** 2)

    def save_debug_img(self, img, imgname):
        img.save(
            f"/home/satviktyagi/Desktop/desk/project/github/DeS3_Deshadow/debug_imgs/{imgname}"
        )
