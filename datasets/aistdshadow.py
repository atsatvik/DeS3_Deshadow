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

file_ext = ["png", "jpg"]


class AISTDShadow:
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
        train_dataset = AISTDShadowDataset(
            dir=os.path.join(self.config.data.data_dir),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=parse_patches,
            folders=["train_A", "train_C", "train_B", "loggray", "train_A_reproj"],
            label="train",
            input_type=self.args.input_type,
            use_class=self.args.use_class,
            guidance_type=self.guidance_type,
        )

        val_dataset = AISTDShadowDataset(
            dir=os.path.join(self.config.data.data_dir),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=parse_patches,
            folders=["test_A", "test_C", "test_B", "loggray", "train_A_reproj"],
            label="test",
            input_type=self.args.input_type,
            use_class=self.args.use_class,
            guidance_type=self.guidance_type,
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
        input_type="sf",
        use_class=False,
        guidance_type="1",
    ):
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

        # if not img_names == GT_names:
        #     raise Exception("images and GT are inconsist")

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

        if self.guidance_type in ["2", "3", "4"]:
            loggray_folder = folders[3]
            loggray_names = [
                f for f in os.listdir(loggray_folder) if f.split(".")[-1] in file_ext
            ]
            loggray_paths = [os.path.join(loggray_folder, f) for f in loggray_names]
            loggray_paths = [loggray_paths[idx] for idx in indices]
            self.loggray_paths = loggray_paths

        elif self.guidance_type == "5":
            rgb_reproj_folder = folders[4]
            rgb_reproj_names = [
                f for f in os.listdir(rgb_reproj_folder) if f.split(".")[-1] in file_ext
            ]
            rgb_reproj_paths = [
                os.path.join(rgb_reproj_folder, f) for f in rgb_reproj_names
            ]
            rgb_reproj_paths = [rgb_reproj_paths[idx] for idx in indices]
            self.rgb_reproj_paths = rgb_reproj_paths

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

        if self.guidance_type in ["2", "3", "4"]:
            loggray_name = self.loggray_paths[index]
            loggray_img = np.array(PIL.Image.open(loggray_name))
            if self.guidance_type == "2":
                input_img = np.array(input_img)
                input_img = PIL.Image.fromarray(
                    np.concatenate([input_img, loggray_img[:, :, np.newaxis]], axis=2)
                )
            elif self.guidance_type == "3":
                input_img = PIL.Image.fromarray(loggray_img)

        if self.parse_patches:
            wd_new = 512
            ht_new = 512
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            # rgb_reproj_name = self.rgb_reproj_paths[index]
            # rgb_reproj = PIL.Image.open(rgb_reproj_name)
            # rgb_reproj = rgb_reproj.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # rgb_reproj = np.array(rgb_reproj).astype(np.float32)
            # test_input_img = np.array(input_img).astype(np.float32)
            # final_img = 0.3 * test_input_img + 0.5 * rgb_reproj

            # loggray_img = PIL.Image.fromarray(loggray_img)
            # loggray_img = loggray_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # loggray_img = np.array(loggray_img).astype(np.float32)
            # loggray_img = np.stack([loggray_img] * 3, axis=-1)

            # test_input_img = np.array(input_img).astype(np.float32)
            # final_img = 0.7 * test_input_img + 0.5 * loggray_img
            # cv2.imshow(
            #     "input_img",
            #     cv2.cvtColor(
            #         np.array(test_input_img).astype(np.uint8), cv2.COLOR_BGR2RGB
            #     ),
            # )
            # cv2.imshow(
            #     "final_img",
            #     cv2.cvtColor(np.array(final_img).astype(np.uint8), cv2.COLOR_BGR2RGB),
            # )
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # exit()

            # print('-input_img.shape,gt_img.shape-',input_img.size,gt_img.size)
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

            if self.guidance_type == "4":
                loggray_img = PIL.Image.fromarray(loggray_img)
                loggray_img = loggray_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
                log_gray_cropped = self.n_random_crops(loggray_img, i, j, h, w)
                for i in range(self.n):
                    img = np.array(input_img[i]).astype(np.float32)
                    log_gray_img = np.array(log_gray_cropped[i]).astype(np.float32)
                    log_gray_img = np.stack([log_gray_img] * 3, axis=-1)
                    final_img = 0.7 * img + 0.5 * log_gray_img
                    # Normalize the values to be in the range [0, 1]
                    input_img[i] = final_img / 255
                # print(np.min(input_img[0]))
                # print(np.max(input_img[0]))
                # print()
                # exit()

            if self.guidance_type == "5":
                rgb_reproj_name = self.rgb_reproj_paths[index]
                rgb_reproj = PIL.Image.open(rgb_reproj_name)
                rgb_reproj = rgb_reproj.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
                rgb_reproj_cropped = self.n_random_crops(rgb_reproj, i, j, h, w)
                for i in range(self.n):
                    img = np.array(input_img[i]).astype(np.float32)
                    rgb_reproj_img = np.array(rgb_reproj_cropped[i]).astype(np.float32)
                    final_img = 0.3 * img + 0.5 * rgb_reproj_img
                    # Normalize the values to be in the range [0, 1]
                    input_img[i] = final_img / 255
                # print(np.min(input_img[0]))
                # print(np.max(input_img[0]))
                # print()
                # exit()

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

    def calculate_mse(self, image1, image2):
        img1 = np.array(image1)
        img2 = np.array(image2)
        return np.mean((img1 - img2) ** 2)

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

    def prepare_labels(self, input_img, gt_img, mask_img):
        labels = []
        # for i, (img_ip, img_gt) in enumerate(zip(input_img, gt_img)):
        #     ###DEBUG########################################
        #     # imgname_ip = f"{self.counter}_{i}_ip.png"
        #     # imgname_gt = f"{self.counter}_{i}_gt.png"
        #     # self.save_debug_img(img_ip, imgname_ip)
        #     # self.save_debug_img(img_gt, imgname_gt)
        #     ###DEBUG########################################
        #     gt_lb = 0
        #     if (
        #         self.calculate_mse(img_ip, img_gt) < 20
        #     ):  # DEAL WITH THIS MAGIC NUMBER IF YOU USE LABEL EMBEDDING
        #         ip_lb = 0
        #     else:
        #         ip_lb = 1
        #     labels.append((ip_lb, gt_lb))
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
