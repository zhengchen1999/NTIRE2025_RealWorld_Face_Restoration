# Imports
import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import numpy as np
from PIL import Image, ImageOps
import torch
from models.team03_PiSAMAP.ram.models.ram_lora import ram

torch_dtype = torch.float16
np_dtype = np.float16

import argparse


torch.set_printoptions(sci_mode=False)
device = 'cuda'
# Shoutout to  bloc97's https://github.com/bloc97/CrossAttentionControl

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# alpha and beta for DDIM
def get_alpha_and_beta(t, scheduler):
    # want to run this for both current and previous timnestep
    if t < 0:
        return scheduler.final_alpha_cumprod.item(), 1 - scheduler.final_alpha_cumprod.item()

    if t.dtype == torch.long or (t == t.long()):
        alpha = scheduler.alphas_cumprod[t.cpu().long()]
        return alpha.item(), 1 - alpha.item()

    low = t.floor().long()
    high = t.ceil().long()
    rem = t - low

    low_alpha = scheduler.alphas_cumprod[low]
    high_alpha = scheduler.alphas_cumprod[high]
    interpolated_alpha = low_alpha * rem + high_alpha * (1 - rem)
    interpolated_beta = 1 - interpolated_alpha
    return interpolated_alpha.item(), interpolated_beta.item()

#forward DDIM step
def forward_step(
        self,
        model_output,
        timestep: int,
        sample,
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

    if timestep > self.timesteps.max():
        raise NotImplementedError("Need to double check what the overflow is")

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)
    first_term = (1. / alpha_quotient) * sample
    second_term = (1. / alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third_term = ((1 - alpha_prod_t_prev) ** 0.5) * model_output

    return first_term - second_term + third_term


# reverse ddim step
def reverse_step(
        self,
        model_output,
        timestep: int,
        sample,
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

    if timestep > self.timesteps.max():
        raise NotImplementedError
    else:
        alpha_prod_t = self.alphas_cumprod[timestep]

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)

    first_term = alpha_quotient * sample
    second_term = ((beta_prod_t) ** 0.5) * model_output
    third_term = alpha_quotient * ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first_term + second_term - third_term


def prep_image_for_return(image):  # take torch image and return PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image


def center_crop(im):  # PIL center_crop
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


from typing import List, Optional, Tuple, Union


def randn_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def load_im_into_format_from_path(im_path):  # From path get formatted PIL image
    return center_crop(
        ImageOps.exif_transpose(Image.open(im_path).convert("RGB")) if isinstance(im_path, str) else im_path).resize(
        (512, 512))  
    # return center_crop(
    #     ImageOps.exif_transpose(Image.open(im_path).convert("RGB")) if isinstance(im_path, str) else im_path)
    #return ImageOps.exif_transpose(Image.open(im_path).convert("RGB")) if isinstance(im_path, str) else im_path


def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (batch_size, num_channels_latents, height, width)
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        if latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents

from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionUpscalePipeline, EulerDiscreteScheduler
from models.team03_PiSAMAP.seesr_models.controlnet import ControlNetModel
from models.team03_PiSAMAP.seesr_models.unet_2d_condition import UNet2DConditionModel

pretrained_model_path = 'pretrained/stable-diffusion-2-base'
seesr_model_path = 'model_zoo/team03_PiSAMAP/seesr/models/seesr'

scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
clip = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
clip_tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
#feature_extractor = CLIPImageProcessor.from_pretrained(f"{pretrained_model_path}/feature_extractor")
unet = UNet2DConditionModel.from_pretrained_orig(pretrained_model_path, seesr_model_path, subfolder="unet", use_image_cross_attention=False)
controlnet = ControlNetModel.from_pretrained(seesr_model_path, subfolder="controlnet")

weight_dtype = torch.float16

clip.to(device, dtype=weight_dtype)
vae.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)
controlnet.to(device, dtype=weight_dtype)

from torchvision import transforms
#tag related
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
ram_model = ram(pretrained='model_zoo/team03_PiSAMAP/seesr/models/ram_swin_large_14m.pth',
            pretrained_condition='model_zoo/team03_PiSAMAP/seesr/models/DAPE.pth',
            image_size=384,
            vit='swin_l')
ram_model.eval()
ram_model.to(device)

scheduler = None

def upcast_vae():
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = True

print("Loaded all models")

