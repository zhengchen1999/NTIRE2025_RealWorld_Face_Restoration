import torch
from torch import nn as nn
from torch.nn import functional as F
from peft import LoraConfig, set_peft_model_state_dict
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers import (
    UNet2DConditionModel,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    LCMScheduler,
    AutoencoderKL,
)

def get_lora_config(rank, use_dora, target_modules):
    base_config = {
        "r": rank,
        "lora_alpha": rank,
        "init_lora_weights": "gaussian",
        "target_modules": target_modules,
    }

    return LoraConfig(**base_config)


class SDXL_Turbo(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, rank=32):
        super(SDXL_Turbo, self).__init__()
        self.unet = UNet2DConditionModel.from_pretrained("/data2/pretrained/sdxl-turbo", subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained("/data2/pretrained/sdxl-turbo", subfolder="vae")
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.timestep = torch.tensor(999., dtype=torch.float32)
        unet_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        unet_lora_config = get_lora_config(rank=rank, use_dora=False, target_modules=unet_target_modules)
        self.unet.add_adapter(unet_lora_config)

        vae_target_modules = ["to_k", "to_q", "to_v", "to_out.0", "conv1", "conv2"]
        vae_lora_config = get_lora_config(rank=rank, use_dora=False, target_modules=vae_target_modules)
        self.vae.add_adapter(vae_lora_config)

        self.add_text_embeds = torch.load('./models/team07_SDFace/add_text_embeds.pt', map_location='cpu')
        self.add_text_embeds = self.add_text_embeds.to(torch.float32)

        self.add_time_ids = torch.load('./models/team07_SDFace/add_time_ids.pt', map_location='cpu')
        self.add_time_ids = self.add_time_ids.to(torch.float32)

        self.prompt_embeds = torch.load('./models/team07_SDFace/prompt_embeds.pt', map_location='cpu')
        self.prompt_embeds = self.prompt_embeds.to(torch.float32)

        print("SDXL_Turbo initialized")
        print("add_text_embeds shape:", self.add_text_embeds.shape)  # pooled
        print("add_time_ids shape:", self.add_time_ids.shape)
        print("add_time_ids:", self.add_time_ids)
        print("prompt_embeds shape:", self.prompt_embeds.shape) # sdxl_text_embedding
        # add_text_embeds shape: torch.Size([1, 1280])
        # add_time_ids shape: torch.Size([1, 6]) 
        # add_time_ids: tensor([[512., 512.,   0.,   0., 512., 512.]])
        # prompt_embeds shape: torch.Size([1, 77, 2048])

    def forward(self, x):
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        batch, _ ,_ ,_ = latents.size()
        # print
        pred = self.unet(
            latents,
            self.timestep.to(latents.device),
            encoder_hidden_states=self.prompt_embeds.repeat(batch,1, 1).to(latents.device),
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs={"text_embeds": self.add_text_embeds.repeat(batch,1).to(latents.device), "time_ids": self.add_time_ids.repeat(batch,1).to(latents.device)},
            return_dict=False,
        )[0]
        # .repeat(batch,1): (n * batch, m * 1)，也就是 (n*batch, m)。

        out = self.vae.decode(pred / self.vae.config.scaling_factor, return_dict=False)[0]
        return out

