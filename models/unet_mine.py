import math
import torch
import torch.nn as nn
import cv2
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# class Classifier(nn.Module):
#     def __init__(self, input_channels, num_classes=2):
#         super(Classifier, self).__init__()
#         self.input_channels = input_channels
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 512, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # For CAM

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         cam = self.conv2(x)  # Class Activation Map (CAM)
#         x = self.avg_pool(cam)
#         x = torch.flatten(x, 1)
#         return x, cam  # Return both the classification output and CAM


class Classifier(nn.Module):
    def __init__(self, input_channels):
        super(Classifier, self).__init__()
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        # Final convolution to get 1 channel (binary classification logit)
        self.conv5 = nn.Conv2d(
            32, 1, kernel_size=1
        )  # 1 channel for binary classification

        # Global average pooling layer to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        # Final convolution to produce the binary logit
        x = self.conv5(x)

        # Global average pooling to reduce spatial dimensions to 1x1
        x = self.global_avg_pool(x)

        # Flatten the output to get a scalar per batch sample (logit for binary classification)
        x = torch.flatten(x, 1)

        return x  # Return the classification logits


# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """

#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         # use_cfg_embedding = dropout_prob > 0
#         self.embedding_table = nn.Embedding(num_classes, hidden_size)
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#     # def token_drop(self, labels, force_drop_ids=None):
#     #     """
#     #     Drops labels to enable classifier-free guidance.
#     #     """
#     #     if force_drop_ids is None:
#     #         drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
#     #     else:
#     #         drop_ids = force_drop_ids == 1
#     #     labels = torch.where(drop_ids, self.num_classes, labels)
#     #     return labels

#     def forward(self, labels, train, force_drop_ids=None):
#         # use_dropout = self.dropout_prob > 0
#         # if (train and use_dropout) or (force_drop_ids is not None):
#         #     print("Token drop called")
#         #     labels = self.token_drop(labels, force_drop_ids)
#         embeddings = self.embedding_table(labels)
#         return embeddings


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ShadowDiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = (
            config.model.ch,
            config.model.out_ch,
            tuple(config.model.ch_mult),
        )
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = (
            config.model.in_channels * 2
            if config.data.conditional
            else config.model.in_channels
        )
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        # Print configuration summary
        print("Configuration Parameters:")
        print(f" - Base channels (ch): {ch}")
        print(f" - Output channels (out_ch): {out_ch}")
        print(f" - Channel multipliers (ch_mult): {ch_mult}")
        print(f" - Number of residual blocks: {num_res_blocks}")
        print(f" - Attention resolutions: {attn_resolutions}")
        print(f" - Dropout rate: {dropout}")
        print(f" - Input channels: {in_channels}")
        print(f" - Image resolution: {resolution}")
        print(f" - Use convolution for resampling: {resamp_with_conv}")

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                torch.nn.Linear(self.ch, self.temb_ch),
                torch.nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )
        # ch_mult[-1] is the deepest feature map size
        self.classifier = Classifier(input_channels=ch_mult[-1] * ch)

        # Add Sigmoid Conv layer to generate the residual map (mres)
        self.sigmoid_conv = nn.Conv2d(
            ch_mult[-1], 1, kernel_size=3, stride=1, padding=1
        )
        self.sigmoid = nn.Sigmoid()

        # self.label_embedder = LabelEmbedder(
        #     num_classes=2, hidden_size=128, dropout_prob=0
        # )
        # nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)

        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        print(
            "Input Convolution Layer: Conv2d({}, {}, kernel_size=3, stride=1, padding=1)".format(
                in_channels, self.ch
            )
        )

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            print(f"Downsampling Level {i_level}:")
            print(
                f" - Block In: {block_in}, Block Out: {block_out}, Residual Blocks: {len(block)}"
            )
            print(f" - Attention Blocks: {len(attn)}")

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        print("Middle Blocks:")
        print(
            f" - Block 1: ResnetBlock(in_channels={block_in}, out_channels={block_in})"
        )
        print(
            f" - Block 2: ResnetBlock(in_channels={block_in}, out_channels={block_in})"
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
            print(f"Upsampling Level {i_level}:")
            print(
                f" - Block In: {block_in}, Block Out: {block_out}, Residual Blocks: {len(block)}"
            )
            print(f" - Attention Blocks: {len(attn)}")

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
        print("Final Layers:")
        print(f" - Normalize Layer: Normalize(block_in={block_in})")
        print(
            f" - Output Convolution Layer: Conv2d({block_in}, {out_ch}, kernel_size=3, stride=1, padding=1)"
        )

        print("\nModel Summary:")
        print(f" - Input Channels: {in_channels}")
        print(f" - Output Channels: {out_ch}")
        print(f" - Total Number of Resolutions: {self.num_resolutions}")
        print(f" - Image Resolution: {resolution}")

        # Optionally print the complete architecture
        print("\nModel Architecture:")
        print(self)
        # exit()

    def forward(self, x, t, labels=None, shadow_free_imgs=None, img_id=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # Encoding portion of UNet or downsampling portion
        # print("Initial x shape:", x.shape)
        hs = [self.conv_in(x)]
        # print("x shape after first conv:", hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(
                # f"level : {i_level} x shape after {i_block} resnet block:",
                # hs[-1].shape,
                # )

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                    # print(
                    # f"x shape after {i_block} attention block at level {i_level}",
                    # hs[-1].shape,
                    # )

                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                # print("x shape after downsample block", hs[-1].shape)

        ############################## LOGS FROM ENCODE PORTION OF UNET FOR DEBUGGING #######################################
        # x shape after first conv: torch.Size([4, 128, 64, 64])
        # level : 0 x shape after 0 resnet block: torch.Size([4, 128, 64, 64])
        # level : 0 x shape after 1 resnet block: torch.Size([4, 128, 64, 64])
        # x shape after downsample block torch.Size([4, 128, 32, 32])
        # level : 1 x shape after 0 resnet block: torch.Size([4, 128, 32, 32])
        # level : 1 x shape after 1 resnet block: torch.Size([4, 128, 32, 32])
        # x shape after downsample block torch.Size([4, 128, 16, 16])
        # level : 2 x shape after 0 resnet block: torch.Size([4, 128, 16, 16])
        # x shape after 0 attention block at level 2 torch.Size([4, 128, 16, 16])
        # level : 2 x shape after 1 resnet block: torch.Size([4, 256, 16, 16])
        # x shape after 1 attention block at level 2 torch.Size([4, 256, 16, 16])
        # x shape after downsample block torch.Size([4, 256, 8, 8])
        # level : 3 x shape after 0 resnet block: torch.Size([4, 256, 8, 8])
        # level : 3 x shape after 1 resnet block: torch.Size([4, 256, 8, 8])
        # x shape after downsample block torch.Size([4, 256, 4, 4])
        # level : 4 x shape after 0 resnet block: torch.Size([4, 256, 4, 4])
        # level : 4 x shape after 1 resnet block: torch.Size([4, 512, 4, 4])
        # x shape after downsample block torch.Size([4, 512, 2, 2])
        # level : 5 x shape after 0 resnet block: torch.Size([4, 512, 2, 2])
        # level : 5 x shape after 1 resnet block: torch.Size([4, 512, 2, 2])
        ############################## LOGS FROM ENCODE PORTION OF UNET FOR DEBUGGING #######################################

        # exit()

        # Middle portion of Unet or bottleneck
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)

        # print("right before classifier h shape", h.shape) # (4, 512, 2, 2)
        # print("input_channels in self.classifier", self.classifier.input_channels)
        # exit()
        # Inject classifier here to generate CAM
        class_pred = self.classifier(h)
        class_pred = torch.sigmoid(class_pred)
        # print(class_pred)
        # exit()
        # print("class_pred.shape", class_pred.shape) (4, 512)

        # Fuse CAM with U-Net features (soft attention)
        # cam_attention = torch.sigmoid(cam)  # Convert CAM to a soft attention map
        # print(cam_attention)
        # print("h.shape", h.shape)
        # print("cam_attention.shape", cam_attention.shape)
        # print
        # exit()
        # h = h * cam_attention  # Apply CAM as attention

        h = self.mid.block_2(h, temb)

        # Decoding portion of UNet or downsampling portion
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # Add the Sigmoid Conv layer for residual map generation
        # mres = self.sigmoid(self.sigmoid_conv(h))  # Residual map: mres = sigma(x0 − x˜)

        ################## DONT NEED TO CALCULATE HERE DO IT IN THE LOSS FUNCTION ITSELF #######################
        diff_map = shadow_free_imgs - x[:, :3, :, :]  # shadow free - shadow img
        mres = self.sigmoid(diff_map)
        # self.debug_image(shadow_free_imgs, name="1_shadow_free_img", write=True)
        # self.debug_image(x, name="2_shadow img", write=True)
        # self.debug_image(diff_map, name="3_diff map", write=True)
        # self.debug_image(mres, name="4_sigmoided diff map", write=True)
        # exit()

        # exit()
        # print("mres.shape", mres.shape)
        # print("h.shape", h.shape)
        # # self.debug_image(h)
        # print("cam_attention.shape", cam_attention.shape)
        # print("img_id", img_id)
        # exit()

        # Compute residual difference (x0 - x˜) only during training
        # if shadow_free_imgs is not None:
        #     print(shadow_free_imgs.shape)
        #     # exit()
        #     diff_map = shadow_free_imgs - x[:, :3, :, :]  # x0 - x˜
        #     mres = self.sigmoid(diff_map)
        return h, class_pred, att

    def debug_image(self, x, name="img", write=False):
        img = x[0, :3, :, :].detach().cpu().numpy()  # * 255
        img = cv2.normalize(
            img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if write:
            cv2.imwrite(
                f"/home/satviktyagi/Desktop/desk/project/github/DeS3_Deshadow/debug_imgs/m_res/{name}.png",
                img,
            )
