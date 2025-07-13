from typing import overload, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

from .sampler import SpacedSampler
from .util import VRAMPeakMonitor, wavelet_reconstruction



def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        out_h, out_w = size, size
    elif h < w:
        out_h, out_w = size, int(w * (size / h))
    else:
        out_h, out_w = int(h * (size / w)), size

    return F.interpolate(imgs, size=(out_h, out_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    ph, pw = map(lambda x: (x + multiple - 1) // multiple * multiple - x, (h, w))
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)



def DiffusionPipe(control_net, diffusion, cond_img, steps, pos_prompt, neg_prompt, device, strength=1.2, \
                vae_encoder_tiled=False,vae_encoder_tile_size=256,vae_decoder_tiled=False,vae_decoder_tile_size=256, \
                cldm_tiled=False,cldm_tile_size=512,cldm_tile_stride=256,cfg_scale=4.0,rescale_cfg=False):
    bs, _, h0, w0 = cond_img.shape
    cond_img = pad_to_multiples_of(cond_img, multiple=64)

    with VRAMPeakMonitor("encoding condition image"):
        cond = control_net.prepare_condition(
            cond_img,
            [pos_prompt] * bs,
            vae_encoder_tiled,
            vae_encoder_tile_size,
        )
        uncond = control_net.prepare_condition(
            cond_img,
            [neg_prompt] * bs,
            vae_encoder_tiled,
            vae_encoder_tile_size,
        )
    h1, w1 = cond["c_img"].shape[2:]
    
    cond["c_img"] = pad_to_multiples_of(cond["c_img"], multiple=8)
    uncond["c_img"] = pad_to_multiples_of(uncond["c_img"], multiple=8)


    cond["c_img"] = pad_to_multiples_of(cond["c_img"], multiple=8)
    uncond["c_img"] = pad_to_multiples_of(uncond["c_img"], multiple=8)
    h2, w2 = cond["c_img"].shape[2:]


    x_T = torch.randn((bs, 4, h2, w2), dtype=torch.float32, device=device)


    control_scales = control_net.control_scales
    control_net.control_scales = [strength] * 13


    betas = diffusion.betas
    parameterization = diffusion.parameterization
    sampler = SpacedSampler(betas, parameterization, rescale_cfg)
    with VRAMPeakMonitor("sampling"):
        z = sampler.sample(
            model=control_net,
            device=device,
            steps=steps,
            x_size=(bs, 4, h2, w2),
            cond=cond,
            uncond=uncond,
            cfg_scale=cfg_scale,
            tiled=cldm_tiled,
            tile_size=cldm_tile_size // 8,
            tile_stride=cldm_tile_stride // 8,
            x_T=x_T,
            progress=True,
        )

        z = z[..., :h1, :w1]
    with VRAMPeakMonitor("decoding generated latent"):
        x = control_net.vae_decode(
            z,
            vae_decoder_tiled,
            vae_decoder_tile_size // 8,
        )
    x = x[:, :, :h0, :w0]
    # import pdb;pdb.set_trace()
    control_net.control_scales = control_scales
    x = wavelet_reconstruction((x + 1) / 2, cond_img)
    return x

 