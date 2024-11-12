import os
import time
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from utils.logging import save_image, save_checkpoint
import torchvision
from models.unet import ShadowDiffusionUNet
from models.transformer2d import My_DiT_test
from torch.utils.tensorboard import SummaryWriter
import shutil
import logging


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device
            )
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b, labels, slice_idx):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    gt = x0[:, slice_idx:, :, :]  # Ground truth image
    x = (
        gt * a.sqrt() + e * (1.0 - a).sqrt()
    )  # Adding noise to the GT to create the noisy image

    if len(labels) != 0:
        labels = labels.flatten(start_dim=0, end_dim=1)
        labels = labels[:, :1]

    # Forward pass: Model predicts the noise to be subtracted
    predicted_noise = model(
        torch.cat([x0[:, :slice_idx, :, :], x], dim=1), t.float(), labels
    )

    # Compute the diffusion loss (L2 loss on predicted noise)
    diffusion_loss = (e - predicted_noise).square().sum(dim=(1, 2, 3)).mean(dim=0)

    total_loss = diffusion_loss
    return total_loss


class DenoisingDiffusion(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.logger = logging.getLogger(__name__)
        self.exp_log_dir = self.config.log_dir

        if self.config.misc.use_class:
            num_classes = 2
            class_dropout_prob = 0.0
        else:
            num_classes = 4
            class_dropout_prob = 0.1

        guidance_type_to_channels = {"1": 6, "2": 7, "3": 4, "4": 6, "5": 6}
        self.in_channels = guidance_type_to_channels[self.config.misc.guidance_type]
        self.slice_idx = self.in_channels - 3

        self.logger.info(f"Using Input Image type {self.config.misc.input_type}")
        if self.config.model.model_type == "ShadowDiff":
            self.model = ShadowDiffusionUNet(config)
            self.logger.info("Using Diffusion model with U-net Backbone")
        elif self.config.model.model_type == "DiT":
            self.logger.info("Using Diffusion model with Transformer Backbone")
            self.model = My_DiT_test(
                input_size=config.data.image_size,
                class_dropout_prob=class_dropout_prob,
                num_classes=num_classes,
                in_channels=self.in_channels,
            )

        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(
            self.config, self.model.parameters()
        )
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        self.writer = SummaryWriter(log_dir=self.exp_log_dir)

        self.keep_last_weights = self.config.misc.num_last_weights
        self.save_every = self.config.misc.save_every

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.ema_helper.load_state_dict(checkpoint["ema_helper"])
        if ema:
            self.ema_helper.ema(self.model)
        self.logger.info(
            f"Loaded checkpoint {load_path} (epoch {self.start_epoch}, step {self.step})"
        )

    def train(self, train_loader, val_loader):
        cudnn.benchmark = True

        if os.path.isfile(self.config.resume):
            self.load_ddm_ckpt(self.config.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            self.logger.info(f"epoch: {epoch}")
            data_start = time.time()
            data_time = 0
            for i, (x, img_id, labels) in tqdm(
                enumerate(train_loader), desc=f"Training Epoch {epoch}: "
            ):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, self.slice_idx :, :, :])
                b = self.betas

                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(
                    self.model, x, t, e, b, labels, self.slice_idx
                )

                if self.step % 10 == 0:
                    self.logger.info(
                        f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )
                global_step = epoch * len(train_loader) + i
                self.writer.add_scalar("Loss/train", loss.item(), global_step)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

            if epoch % self.save_every == 0:
                self.model.eval()
                self.sample_validation_patches(val_loader, self.step)

                save_path = os.path.join(
                    self.exp_log_dir,
                    f"epoch_{epoch + 1}_ddpm",
                )
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "step": self.step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "ema_helper": self.ema_helper.state_dict(),
                        "params": self.config,
                    },
                    filename=save_path,
                )
                if self.keep_last_weights:
                    self.remove_extra_weight_files()

                self.logger.info(f"Saving checkpoint at {save_path}")

        self.writer.close()

    def remove_extra_weight_files(self):
        weight_files = os.listdir(self.exp_log_dir)
        weight_files = [elem for elem in weight_files if ".pth.tar" in elem]
        if len(weight_files) <= self.keep_last_weights:
            return

        min_epoch_in_filename = float("inf")
        min_epoch_filename = ""
        for filename in weight_files:
            if int(filename.split("_")[1]) < min_epoch_in_filename:
                min_epoch_in_filename = int(filename.split("_")[1])
                min_epoch_filename = filename

        os.remove(os.path.join(self.exp_log_dir, min_epoch_filename))

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = (
            self.config.diffusion.num_diffusion_timesteps
            // self.config.misc.sampling_timesteps
        )
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(
                x,
                x_cond,
                seq,
                self.model,
                self.betas,
                eta=0.0,
                corners=patch_locs,
                p_size=patch_size,
            )
        else:
            xs = utils.sampling.generalized_steps(
                x, x_cond, seq, self.model, self.betas, eta=0.0
            )
        if last:
            xs = xs[0][-1]
        return xs

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(
            self.exp_log_dir,
            self.config.data.dataset + str(self.config.data.image_size),
        )
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        with torch.no_grad():
            for i, (x, img_id, label) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, : self.slice_idx, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(
                n,
                3,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.device,
            )
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                save_image(
                    x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png")
                )
                save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
