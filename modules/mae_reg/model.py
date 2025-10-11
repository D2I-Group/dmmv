import os

import einops
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from . import models_mae, util

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"],
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"


class MAE_Reg(nn.Module):
    def __init__(
        self,
        context_len,
        pred_len,
        periodicity,
        norm_const,
        align_const,
        interpolation,
        arch,
        finetune_type,
        ckpt_dir,
        load_ckpt,
    ):
        super(MAE_Reg, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(
                f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}"
            )

        self.vision_model = MAE_ARCH[arch][0]()

        # load pretrained MAE
        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                self.vision_model.load_state_dict(checkpoint["model"], strict=True)
            except Exception as e:
                print(
                    f"Bad checkpoint file. Please delete {ckpt_path} and redownload! Error: {e}"
                )

        if finetune_type != "full":
            for n, param in self.vision_model.named_parameters():
                if "ln" == finetune_type:
                    param.requires_grad = "norm" in n
                elif "bias" == finetune_type:
                    param.requires_grad = "bias" in n
                elif "none" == finetune_type:
                    param.requires_grad = False
                elif "mlp" in finetune_type:
                    param.requires_grad = ".mlp." in n
                elif "attn" in finetune_type:
                    param.requires_grad = ".attn." in n

        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity

        input_ratio = (self.pad_left + self.context_len) / (
            self.pad_left + self.context_len + self.pad_right + self.pred_len
        )
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize(
            (self.image_size, int(self.image_size * adjust_input_ratio)),
            interpolation=interpolation,
        )
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (
            int(self.image_size * adjust_input_ratio)
        )
        self.output_resize = util.safe_resize(
            (self.periodicity, int(round(self.image_size * self.scale_x))),
            interpolation=interpolation,
        )
        self.norm_const = norm_const

        mask = torch.ones((self.num_patch, self.num_patch)).to(
            self.vision_model.cls_token.device
        )
        mask[:, : self.num_patch_input] = torch.zeros(
            (self.num_patch, self.num_patch_input)
        )
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()

    def forward(self, x, fp64=False):
        means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(
                x_enc.to(torch.float64) if fp64 else x_enc,
                dim=1,
                keepdim=True,
                unbiased=False,
            )
            + 1e-5
        )  # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev

        x_enc = einops.rearrange(x_enc, "b s n -> b n s")  # [bs x nvars x seq_len]
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode="replicate")  # [b n s]
        x_2d = einops.rearrange(
            x_pad, "b n (p f) -> (b n) 1 f p", f=self.periodicity
        )  # (bs * nvars, 1, period, patch)

        x_resize = self.input_resize(x_2d)
        masked = torch.zeros(
            (
                x_2d.shape[0],
                1,
                self.image_size,
                self.num_patch_output * self.patch_size,
            ),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, "b 1 h w -> b c h w", c=3)

        _, y, mask = self.vision_model(
            image_input,
            mask_ratio=self.mask_ratio,
            noise=einops.repeat(self.mask, "1 l -> n l", n=image_input.shape[0]),
        )
        image_reconstructed = self.vision_model.unpatchify(
            y
        )  # [(bs x nvars) x 3 x h x w]

        y_grey = torch.mean(
            image_reconstructed, 1, keepdim=True
        )  # [(bs x nvars) x 1 x h x w]
        y_segmentations = self.output_resize(y_grey)
        y_flatten = einops.rearrange(
            y_segmentations,
            "(b n) 1 f p -> b (p f) n",
            b=x_enc.shape[0],
            f=self.periodicity,
        )
        y = y_flatten[
            :,
            self.pad_left + self.context_len : self.pad_left
            + self.context_len
            + self.pred_len,
            :,
        ]  # extract the forecasting window

        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        return y


class MAE_Reg_BackCast_Forecast(nn.Module):
    def __init__(
        self,
        context_len,
        pred_len,
        periodicity,
        norm_const,
        align_const,
        interpolation,
        arch,
        finetune_type,
        ckpt_dir,
        load_ckpt,
    ):
        super(MAE_Reg_BackCast_Forecast, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(
                f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}"
            )
        self.vision_model = MAE_ARCH[arch][0]()
        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                util.download_file(MAE_DOWNLOAD_URL + MAE_ARCH[arch][1], ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.vision_model.load_state_dict(checkpoint["model"], strict=True)

        if finetune_type != "full":
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = (
                    ("norm" in name)
                    if finetune_type == "ln"
                    else ("bias" in name)
                    if finetune_type == "bias"
                    else False
                    if finetune_type == "none"
                    else (".mlp." in name)
                    if "mlp" in finetune_type
                    else (".attn." in name)
                    if "attn" in finetune_type
                    else True
                )

        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.norm_const = norm_const

        self.pad_left = (periodicity - context_len % periodicity) % periodicity
        self.pad_right = (periodicity - pred_len % periodicity) % periodicity

        total = self.pad_left + context_len + self.pad_right + pred_len
        input_ratio = (self.pad_left + context_len) / total
        self.num_patch_input = max(1, int(input_ratio * self.num_patch * align_const))
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interp = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        self.input_resize = util.safe_resize(
            (self.image_size, int(self.image_size * adjust_input_ratio)),
            interpolation=interp,
        )
        self.scale_x = ((self.pad_left + context_len) // periodicity) / (
            int(self.image_size * adjust_input_ratio)
        )
        self.output_resize = util.safe_resize(
            (periodicity, int(round(self.image_size * self.scale_x))),
            interpolation=interp,
        )

        mask = torch.ones((self.num_patch, self.num_patch))
        mask[:, : self.num_patch_input] = 0
        self.register_buffer("mask_forecast", mask.reshape(1, -1))
        self.mask_ratio_forecast = mask.mean().item()

        half = self.num_patch_input // 2
        mask_latter = torch.ones((self.num_patch, self.num_patch))
        mask_latter[:, :half] = 0
        self.register_buffer("mask_backcast_latter", mask_latter.reshape(1, -1))
        self.mask_ratio_backcast_latter = mask_latter.mean().item()

        mask_early = torch.ones((self.num_patch, self.num_patch))
        mask_early[:, half : self.num_patch_input] = 0
        self.register_buffer("mask_backcast_early", mask_early.reshape(1, -1))
        self.mask_ratio_backcast_early = mask_early.mean().item()

    def forward(self, x, fp64=False):
        B, S, N = x.shape
        means = x.mean(1, keepdim=True)
        x_enc = x - means
        var = torch.var(
            x_enc.to(torch.float64) if fp64 else x_enc,
            dim=1,
            keepdim=True,
            unbiased=False,
        )
        stdev = (torch.sqrt(var + 1e-5) / self.norm_const).to(x_enc.dtype)
        x_enc = x_enc / stdev

        x_enc = einops.rearrange(x_enc, "b s n -> b n s")
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode="replicate")
        x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=self.periodicity)

        x_resize = self.input_resize(x_2d)
        masked = torch.zeros(
            (
                x_2d.shape[0],
                1,
                self.image_size,
                self.num_patch_output * self.patch_size,
            ),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        x_cat = torch.cat([x_resize, masked], dim=-1)
        image_input = einops.repeat(x_cat, "b 1 h w -> b 3 h w", b=x_cat.shape[0])

        noise_f = einops.repeat(
            self.mask_forecast, "1 l -> n l", n=image_input.shape[0]
        )
        _, y_f, _ = self.vision_model(
            image_input, mask_ratio=self.mask_ratio_forecast, noise=noise_f
        )
        img_f = self.vision_model.unpatchify(y_f)
        grey_f = img_f.mean(1, keepdim=True)
        seg_f = self.output_resize(grey_f)
        flat_f = einops.rearrange(seg_f, "(b n) 1 f p -> b (p f) n", b=B)
        forecast = flat_f[
            :,
            self.pad_left + self.context_len : self.pad_left
            + self.context_len
            + self.pred_len,
            :,
        ]
        forecast = forecast * stdev.repeat(1, forecast.size(1), 1) + means.repeat(
            1, forecast.size(1), 1
        )

        noise_l = einops.repeat(
            self.mask_backcast_latter, "1 l -> n l", n=image_input.shape[0]
        )
        _, y_l, _ = self.vision_model(
            image_input, mask_ratio=self.mask_ratio_backcast_latter, noise=noise_l
        )
        img_l = self.vision_model.unpatchify(y_l)
        grey_l = img_l.mean(1, keepdim=True)
        seg_l = self.output_resize(grey_l)
        flat_l = einops.rearrange(seg_l, "(b n) 1 f p -> b (p f) n", b=B)
        half_len = self.context_len // 2
        back_l = flat_l[
            :, self.pad_left + half_len : self.pad_left + self.context_len, :
        ]
        back_l = back_l * stdev.repeat(1, back_l.size(1), 1) + means.repeat(
            1, back_l.size(1), 1
        )

        noise_e = einops.repeat(
            self.mask_backcast_early, "1 l -> n l", n=image_input.shape[0]
        )
        _, y_e, _ = self.vision_model(
            image_input, mask_ratio=self.mask_ratio_backcast_early, noise=noise_e
        )
        img_e = self.vision_model.unpatchify(y_e)
        grey_e = img_e.mean(1, keepdim=True)
        seg_e = self.output_resize(grey_e)
        flat_e = einops.rearrange(seg_e, "(b n) 1 f p -> b (p f) n", b=B)
        back_e = flat_e[:, self.pad_left : self.pad_left + half_len, :]
        back_e = back_e * stdev.repeat(1, back_e.size(1), 1) + means.repeat(
            1, back_e.size(1), 1
        )

        backcast = torch.cat([back_e, back_l], dim=1)  # [B, context_len, N]

        return backcast, forecast


class MAE_Reg_BackCast_Forecast_RandomMask(nn.Module):
    def __init__(
        self,
        context_len,
        pred_len,
        periodicity,
        norm_const,
        align_const,
        interpolation,
        arch,
        finetune_type,
        ckpt_dir,
        load_ckpt,
    ):
        super(MAE_Reg_BackCast_Forecast_RandomMask, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(
                f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}"
            )
        self.vision_model = MAE_ARCH[arch][0]()
        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                util.download_file(MAE_DOWNLOAD_URL + MAE_ARCH[arch][1], ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.vision_model.load_state_dict(checkpoint["model"], strict=True)

        if finetune_type != "full":
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = (
                    ("norm" in name)
                    if finetune_type == "ln"
                    else ("bias" in name)
                    if finetune_type == "bias"
                    else False
                    if finetune_type == "none"
                    else (".mlp." in name)
                    if "mlp" in finetune_type
                    else (".attn." in name)
                    if "attn" in finetune_type
                    else True
                )

        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.norm_const = norm_const

        self.pad_left = (periodicity - context_len % periodicity) % periodicity
        self.pad_right = (periodicity - pred_len % periodicity) % periodicity

        total = self.pad_left + context_len + self.pad_right + pred_len
        input_ratio = (self.pad_left + context_len) / total
        self.num_patch_input = max(1, int(input_ratio * self.num_patch * align_const))
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interp = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        self.input_resize = util.safe_resize(
            (self.image_size, int(self.image_size * adjust_input_ratio)),
            interpolation=interp,
        )
        self.scale_x = ((self.pad_left + context_len) // periodicity) / (
            int(self.image_size * adjust_input_ratio)
        )
        self.output_resize = util.safe_resize(
            (periodicity, int(round(self.image_size * self.scale_x))),
            interpolation=interp,
        )

        mask = torch.ones((self.num_patch, self.num_patch))
        mask[:, : self.num_patch_input] = 0
        self.register_buffer("mask_forecast", mask.reshape(1, -1))
        self.mask_ratio_forecast = mask.mean().item()

    def forward(self, x, fp64=False):
        B, S, N = x.shape
        means = x.mean(1, keepdim=True)
        x_enc = x - means
        var = torch.var(
            x_enc.to(torch.float64) if fp64 else x_enc,
            dim=1,
            keepdim=True,
            unbiased=False,
        )
        stdev = (torch.sqrt(var + 1e-5) / self.norm_const).to(x_enc.dtype)
        x_enc = x_enc / stdev

        x_enc = einops.rearrange(x_enc, "b s n -> b n s")
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode="replicate")
        x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=self.periodicity)

        x_resize = self.input_resize(x_2d)
        masked = torch.zeros(
            (
                x_2d.shape[0],
                1,
                self.image_size,
                self.num_patch_output * self.patch_size,
            ),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        x_cat = torch.cat([x_resize, masked], dim=-1)
        image_input = einops.repeat(x_cat, "b 1 h w -> b 3 h w", b=x_cat.shape[0])

        noise_f = einops.repeat(
            self.mask_forecast, "1 l -> n l", n=image_input.shape[0]
        )
        _, y_f, _ = self.vision_model(
            image_input, mask_ratio=self.mask_ratio_forecast, noise=noise_f
        )
        img_f = self.vision_model.unpatchify(y_f)
        grey_f = img_f.mean(1, keepdim=True)
        seg_f = self.output_resize(grey_f)
        flat_f = einops.rearrange(seg_f, "(b n) 1 f p -> b (p f) n", b=B)
        forecast = flat_f[
            :,
            self.pad_left + self.context_len : self.pad_left
            + self.context_len
            + self.pred_len,
            :,
        ]
        forecast = forecast * stdev.repeat(1, forecast.size(1), 1) + means.repeat(
            1, forecast.size(1), 1
        )

        total_patch = self.num_patch * self.num_patch
        rand_idx = torch.randperm(total_patch, device=image_input.device)
        mask1 = torch.ones(total_patch, device=image_input.device)
        mask1[rand_idx[: total_patch // 2]] = 0
        mask2 = 1 - mask1

        noise_1 = einops.repeat(
            mask1.unsqueeze(0), "1 l -> n l", n=image_input.shape[0]
        )
        noise_2 = einops.repeat(
            mask2.unsqueeze(0), "1 l -> n l", n=image_input.shape[0]
        )

        _, y_1, _ = self.vision_model(image_input, mask_ratio=0.5, noise=noise_1)
        _, y_2, _ = self.vision_model(image_input, mask_ratio=0.5, noise=noise_2)

        img_1 = self.vision_model.unpatchify(y_1)
        img_2 = self.vision_model.unpatchify(y_2)

        grey_1 = img_1.mean(1, keepdim=True)
        grey_2 = img_2.mean(1, keepdim=True)

        mask1_upsampled = (
            mask1.view(1, 1, self.num_patch, self.num_patch)
            .repeat_interleave(self.patch_size, dim=2)
            .repeat_interleave(self.patch_size, dim=3)
        )
        mask2_upsampled = (
            mask2.view(1, 1, self.num_patch, self.num_patch)
            .repeat_interleave(self.patch_size, dim=2)
            .repeat_interleave(self.patch_size, dim=3)
        )

        grey_1 = grey_1 * mask1_upsampled
        grey_2 = grey_2 * mask2_upsampled

        seg_1 = self.output_resize(grey_1)
        seg_2 = self.output_resize(grey_2)

        flat_1 = einops.rearrange(seg_1, "(b n) 1 f p -> b (p f) n", b=B)
        flat_2 = einops.rearrange(seg_2, "(b n) 1 f p -> b (p f) n", b=B)

        back_1 = flat_1[:, self.pad_left : self.pad_left + self.context_len, :]
        back_2 = flat_2[:, self.pad_left : self.pad_left + self.context_len, :]

        backcast = back_1 + back_2

        backcast = backcast * stdev.repeat(1, backcast.size(1), 1) + means.repeat(
            1, backcast.size(1), 1
        )

        return backcast, forecast


class MAE_Reg_BackCast_Forecast_NoMask(nn.Module):
    def __init__(
        self,
        context_len,
        pred_len,
        periodicity,
        norm_const,
        align_const,
        interpolation,
        arch,
        finetune_type,
        ckpt_dir,
        load_ckpt,
    ):
        super(MAE_Reg_BackCast_Forecast_NoMask, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(
                f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}"
            )
        self.vision_model = MAE_ARCH[arch][0]()
        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                util.download_file(MAE_DOWNLOAD_URL + MAE_ARCH[arch][1], ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.vision_model.load_state_dict(checkpoint["model"], strict=True)

        if finetune_type != "full":
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = (
                    ("norm" in name)
                    if finetune_type == "ln"
                    else ("bias" in name)
                    if finetune_type == "bias"
                    else False
                    if finetune_type == "none"
                    else (".mlp." in name)
                    if "mlp" in finetune_type
                    else (".attn." in name)
                    if "attn" in finetune_type
                    else True
                )

        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.norm_const = norm_const

        self.pad_left = (periodicity - context_len % periodicity) % periodicity
        self.pad_right = (periodicity - pred_len % periodicity) % periodicity

        total = self.pad_left + context_len + self.pad_right + pred_len
        input_ratio = (self.pad_left + context_len) / total
        self.num_patch_input = max(1, int(input_ratio * self.num_patch * align_const))
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interp = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        self.input_resize = util.safe_resize(
            (self.image_size, int(self.image_size * adjust_input_ratio)),
            interpolation=interp,
        )
        self.scale_x = ((self.pad_left + context_len) // periodicity) / (
            int(self.image_size * adjust_input_ratio)
        )
        self.output_resize = util.safe_resize(
            (periodicity, int(round(self.image_size * self.scale_x))),
            interpolation=interp,
        )

        mask = torch.ones((self.num_patch, self.num_patch))
        mask[:, : self.num_patch_input] = 0
        self.register_buffer("mask_forecast", mask.reshape(1, -1))
        self.mask_ratio_forecast = mask.mean().item()

    def forward(self, x, fp64=False):
        B, S, N = x.shape
        means = x.mean(1, keepdim=True)
        x_enc = x - means
        var = torch.var(
            x_enc.to(torch.float64) if fp64 else x_enc,
            dim=1,
            keepdim=True,
            unbiased=False,
        )
        stdev = (torch.sqrt(var + 1e-5) / self.norm_const).to(x_enc.dtype)
        x_enc = x_enc / stdev

        x_enc = einops.rearrange(x_enc, "b s n -> b n s")
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode="replicate")
        x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=self.periodicity)

        x_resize = self.input_resize(x_2d)
        masked = torch.zeros(
            (
                x_2d.shape[0],
                1,
                self.image_size,
                self.num_patch_output * self.patch_size,
            ),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        x_cat = torch.cat([x_resize, masked], dim=-1)
        image_input = einops.repeat(x_cat, "b 1 h w -> b 3 h w", b=x_cat.shape[0])

        noise_f = einops.repeat(
            self.mask_forecast, "1 l -> n l", n=image_input.shape[0]
        )
        _, y_f, _ = self.vision_model(
            image_input, mask_ratio=self.mask_ratio_forecast, noise=noise_f
        )
        img_f = self.vision_model.unpatchify(y_f)
        grey_f = img_f.mean(1, keepdim=True)
        seg_f = self.output_resize(grey_f)
        flat_f = einops.rearrange(seg_f, "(b n) 1 f p -> b (p f) n", b=B)
        backcast = flat_f[:, self.pad_left : self.pad_left + self.context_len, :]

        forecast = flat_f[
            :,
            self.pad_left + self.context_len : self.pad_left
            + self.context_len
            + self.pred_len,
            :,
        ]
        forecast = forecast * stdev.repeat(1, forecast.size(1), 1) + means.repeat(
            1, forecast.size(1), 1
        )
        return backcast, forecast
