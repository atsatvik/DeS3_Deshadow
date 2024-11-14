import torch
import torch.nn as nn
import utils
import torchvision
import os
import numpy as np
import cv2
from utils.logging import save_image


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, config):
        super(DiffusiveRestoration, self).__init__()
        self.config = config
        self.diffusion = diffusion
        self.input_type = self.config.misc.input_type

        guidance_type_to_channels = {"1": 6, "2": 7, "3": 4, "4": 6, "5": 6}
        self.in_channels = guidance_type_to_channels[config.misc.guidance_type]
        self.slice_idx = self.in_channels - 3

        if os.path.isfile(self.config.resume):
            self.diffusion.load_ddm_ckpt(self.config.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print("Pre-trained diffusion model path is missing!")

    def transform_output(self, output, conditional):
        output = np.array(output.detach().cpu().numpy()).astype(np.float32)
        conditional = np.array(conditional.detach().cpu().numpy()).astype(np.float32)
        if self.input_type == "sf-s":
            output += conditional
        elif self.input_type == "sf/s":
            output *= conditional
        output = torch.tensor(output, device=self.diffusion.device)
        return output

    def restore(self, val_loader, validation="snow", r=None, sid=None):
        image_folder = self.config.log_dir
        print(f"Saving images to path {image_folder}")

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                print(f"Restoring image: size {x.shape} ; ID {y}")
                if sid:
                    y = y[0]
                    if sid + "__" in y:
                        print(self.args.image_folder, self.config.data.dataset)
                        print(i, y)
                        datasetname = y.split("__")[0]
                        id = y.split("__")[1]
                        frame = y.split("__")[2]
                        print(datasetname, id, frame)
                        # print(os.path.join(image_folder, id, f"{frame}_output.png"))
                        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x  ##1
                        x_cond = x[:, :3, :, :].to(self.diffusion.device)  ##2
                        x_gt = x[:, 3:, :, :].to(self.diffusion.device)
                        print("x_cond = ", x_cond.shape)

                        x_output = self.diffusive_restoration(x_cond, r=r)  ##3
                        x_output = inverse_data_transform(x_output)  ##4
                        save_image(
                            x_cond, os.path.join(image_folder, f"{frame}_in.png")
                        )
                        save_image(
                            x_output, os.path.join(image_folder, f"{frame}_out.png")
                        )
                        save_image(
                            x_gt, os.path.join(image_folder, f"{frame}_gt.png")
                        )  ##yy
                        x_output = x_output.to(self.diffusion.device)  ##yy
                        x_gt = x_gt.to(self.diffusion.device)  ##yy
                        saveimg = torch.cat((x_cond, x_output, x_gt), dim=3)  ##yy
                        save_image(
                            saveimg,
                            os.path.join(image_folder, f"{frame}_in_out_gt.png"),
                        )  ##yy
                else:
                    print(i, y)
                    y = y[0]
                    frame = y
                    print(f"starting processing from image {y}")

                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x  ##1
                    x_cond = x[:, : self.slice_idx, :, :].to(self.diffusion.device)  ##2
                    x_gt = x[:, self.slice_idx :, :, :].to(self.diffusion.device)
                    # print('x_cond, x_gt = ',x_cond.shape, x_gt.shape) #[1, 3, 256, 256]
                    x_output = self.diffusive_restoration(x_cond, x_gt.size(), r=r)  ##3
                    x_output = inverse_data_transform(x_output)  ##4
                    x_output = self.transform_output(x_output, x_cond)

                    save_image(
                        x_cond, os.path.join(image_folder, "input", f"{frame}.png")
                    )
                    save_image(
                        x_output, os.path.join(image_folder, "output", f"{frame}.png")
                    )
                    save_image(x_gt, os.path.join(image_folder, "gt", f"{frame}.png"))
                    x_output = x_output.to(self.diffusion.device)
                    x_gt = x_gt.to(self.diffusion.device)

                    if self.config.misc.guidance_type in ["1", "4", "5"]:
                        saveimg = torch.cat((x_cond, x_output, x_gt), dim=3)
                    else:
                        saveimg = getSaveImg(
                            x_cond, x_output, x_gt, self.config, self.slice_idx
                        )

                    save_image(
                        saveimg, os.path.join(image_folder, "in_out_gt", f"{frame}.png")
                    )

    def diffusive_restoration(self, x_cond, size=None, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        # random noise
        if size is None:
            size = x_cond.size()
        x = torch.randn(size, device=self.diffusion.device)
        # passing in random noise as input and shadow image as conditional
        x_output = self.diffusion.sample_image(
            x_cond, x, patch_locs=corners, patch_size=p_size
        )
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list


def getSaveImg(x_cond, x_output, x_gt, config, slice_idx):
    if config.misc.guidance_type == "2":
        x_gray = x_cond[:, slice_idx - 1, :, :]
        x_gray = x_gray.expand(x_gray.shape[0], 3, x_gray.shape[1], x_gray.shape[2])
        x_cond = x_cond[:, : slice_idx - 1, :, :]
        saveimg = torch.cat((x_cond, x_gray, x_output, x_gt), dim=3)
    elif config.misc.guidance_type == "3":
        x_gray = x_cond[:, slice_idx - 1, :, :]
        x_gray = x_gray.expand(x_gray.shape[0], 3, x_gray.shape[1], x_gray.shape[2])
        saveimg = torch.cat((x_gray, x_output, x_gt), dim=3)
    return saveimg
