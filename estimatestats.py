import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
import torch.nn.functional as F
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from math import exp

from inference_utils import *


class Config(defaultdict):
    def __init__(self, default_factory=None):
        super().__init__(default_factory)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class SSIM(torch.nn.Module):
    def __init__(self, use_focal_ssim, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        self.use_focal_ssim = use_focal_ssim

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(
            _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        )
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.use_focal_ssim:
            ssim_map = ssim_map / torch.exp(1 - ssim_map)

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def getNumpyArr(tensor):
    return tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)


def main(out_imgs, gt_imgs, names, config):
    ssim_obj = SSIM(use_focal_ssim=False).to(config.device)

    ssim_ls = []
    psnr_ls = []
    MSE_ls = []
    RMSE_ls = []

    # Initialize text file
    txt_file_path = os.path.join(config.save_dir, "stats.txt")
    if not os.path.exists(txt_file_path):
        with open(txt_file_path, "w") as f:
            f.write(f"")
    ssim_type = "Focal " if config.use_focal_ssim else ""

    for i, (input_image, target_image, imgname) in enumerate(
        tqdm(zip(out_imgs, gt_imgs, names), desc="Validating")
    ):
        input_image = input_image.float().cuda()
        target_image = target_image.float().cuda()

        ssim = ssim_obj(input_image, target_image).detach().cpu()
        ssim_ls.append(ssim)

        mse = F.mse_loss(input_image, target_image).detach().cpu()
        MSE_ls.append(mse)

        psnr = calculate_psnr(mse, max_pixel=255.0)
        psnr_ls.append(psnr)

        rmse = calculate_rmse(input_image, target_image).detach().cpu()
        RMSE_ls.append(rmse)

        string = f"{imgname} {ssim_type}SSIM: {ssim_ls[-1]:.4f} || RMSE: {RMSE_ls[-1]:.4f} || PSNR: {psnr_ls[-1]:.4f}\n"
        # Saving per image SSIM
        with open(txt_file_path, "a") as f:
            f.write(string)
        print(string)

    # Saving mean SSIM
    string = f"Mean {ssim_type}SSIM: {np.mean(np.array(ssim_ls), axis=0):.4f} || "
    string += f"Mean RMSE: {np.mean(np.array(RMSE_ls), axis=0):.4f} || "
    string += f"Mean PSNR: {np.mean(np.array(psnr_ls), axis=0):.4f}"
    with open(txt_file_path, "a") as f:
        f.write(string)
    print(string)


def readimgs(path):
    names = os.listdir(path)
    names.sort()
    imgs = [
        cv2.imread(os.path.join(path, name), cv2.IMREAD_UNCHANGED) for name in names
    ]
    imgs = [torch.from_numpy(img).unsqueeze(0).permute([0, 3, 1, 2]) for img in imgs]
    return imgs, names


if __name__ == "__main__":
    config = Config(lambda: None)
    config.device = "cuda"
    config.save_dir = (
        "/home/satviktyagi/Desktop/desk/project/to_show/for_doc/diff_results/quant_diff"
    )

    path = "/path/to/images"  # folder structure should be <path>/output and <path>/gt
    # output - network preds
    # gt - ground truth

    out_imgs, names = readimgs(os.path.join(path, "output"))
    gt_imgs, _ = readimgs(os.path.join(path, "gt"))

    main(out_imgs, gt_imgs, names, config)
