import os
import math
from inspect import isfunction
import torch
import torch.nn as nn
import numpy as np
from einops import repeat
from typing import Any, Dict, Callable, Literal
from torch import Tensor
from torch.nn import functional as F


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# class CheckpointFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#         ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
#                                    "dtype": torch.get_autocast_gpu_dtype(),
#                                    "cache_enabled": torch.is_autocast_cache_enabled()}
#         with torch.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         with torch.enable_grad(), \
#                 torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
#             # Fixes a bug where the first op in run_function modifies the
#             # Tensor storage in place, which is not allowed for detach()'d
#             # Tensors.
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             output_tensors = ctx.run_function(*shallow_copies)
#         input_grads = torch.autograd.grad(
#             output_tensors,
#             ctx.input_tensors + ctx.input_params,
#             output_grads,
#             allow_unused=True,
#         )
#         del ctx.input_tensors
#         del ctx.input_params
#         del output_tensors
#         return (None, None) + input_grads


# Fixes: When we set unet parameters with requires_grad=False, the original CheckpointFunction
# still tries to compute gradient for unet parameters.
# https://discuss.pytorch.org/t/get-runtimeerror-one-of-the-differentiated-tensors-does-not-require-grad-in-pytorch-lightning/179738/6
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + [x for x in ctx.input_params if x.requires_grad],
            output_grads,
            allow_unused=True,
        )
        grads = list(grads)
        # Assign gradients to the correct positions, matching None for those that do not require gradients
        input_grads = []
        for tensor in ctx.input_tensors + ctx.input_params:
            if tensor.requires_grad:
                input_grads.append(grads.pop(0))  # Get the next computed gradient
            else:
                input_grads.append(None)  # No gradient required for this tensor
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + tuple(input_grads)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    # callback: Callable[[int, int, int, int], None] | None = None,
    progress: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Only split the first input of function.
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        out = torch.zeros(
            (b, out_channel, scale_fn(h), scale_fn(w)),
            dtype=out_dtype,
            device=out_device,
        )
        count = torch.zeros_like(out, dtype=torch.float32)
        weight_size = scale_fn(size)
        weights = (
            gaussian_weights(weight_size, weight_size)[None, None]
            if weight == "gaussian"
            else np.ones((1, 1, weight_size, weight_size))
        )
        weights = torch.tensor(
            weights,
            dtype=out_dtype,
            device=out_device,
        )

        indices = sliding_windows(h, w, size, stride)
        pbar = tqdm(
            indices, desc=f"Tiled Processing", disable=not progress, leave=False
        )
        for hi, hi_end, wi, wi_end in pbar:
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(
                scale_fn, (hi, hi_end, wi, wi_end)
            )
            if len(args) or len(kwargs):
                kwargs.update(dict(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (
                fn(x_tile, *args, **kwargs) * weights
            )
            count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        out = out / count
        return out

    return tiled_fn


TRACE_VRAM = int(os.environ.get("TRACE_VRAM", False))


def trace_vram_usage(tag: str) -> Callable:
    def wrapper_1(func: Callable) -> Callable:
        if not TRACE_VRAM:
            return func

        def wrapper_2(*args, **kwargs):
            peak_before = torch.cuda.max_memory_allocated() / (1024**3)
            ret = func(*args, **kwargs)
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated() / (1024**3)
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            print(
                f"{YELLOW}VRAM peak before {tag}: {peak_before:.5f} GB, "
                f"after: {peak_after:.5f} GB{RESET}"
            )
            return ret

        return wrapper_2

    return wrapper_1

class VRAMPeakMonitor:

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __enter__(self):
        self.peak_before = torch.cuda.max_memory_allocated() / (1024**3)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated() / (1024**3)
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        if TRACE_VRAM:
            print(
                f"{YELLOW}VRAM peak before {self.tag}: {self.peak_before:.2f} GB, "
                f"after: {peak_after:.2f} GB{RESET}"
            )
        return False




def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2**i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq