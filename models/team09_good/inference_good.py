import argparse
import cv2
import glob
import numpy as np
import shutil
import math
import random
import torch
import os
import torch
from basicsr.utils import imwrite
from torch import nn
from torch.nn import functional as F

from .model import GFPGANer
from accelerate.utils import set_seed
from .diffbir.inference import (
    BFRInferenceLoop,
)



def main(model_dir, input_path=None, output_path=None, device=None, args=None):
    """Inference demo for GFPGAN (for users).
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-i',
    #     '--input',
    #     type=str,
    #     default=input_path,
    #     help='Input image or folder. Default: inputs/whole_imgs')
    # parser.add_argument('-o', '--output', type=str, default=output_path, help='Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    # parser.add_argument(
    #     '-v', '--version', type=str, default='1.4', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    # parser.add_argument(
    #     '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')

    # parser.add_argument(
    #     '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    # parser.add_argument(
    #     '--bg_tile',
    #     type=int,
    #     default=400,
    #     help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    # parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    # parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    # parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    # parser.add_argument(
    #     '--ext',
    #     type=str,
    #     default='auto',
    #     help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    # parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    # args = parser.parse_args()

    # args = parser.parse_args()
    # ----------------------------------------------------------------
    # device = get_device()
    print(f'Running on device: {device}')
    # args.bg_upsampler = 'realesrgan'


    # ------------------------ input & output ------------------------
    if input_path.endswith('/'):
        input_path = input_path[:-1]
    if os.path.isfile(input_path):
        img_list = [input_path]
    else:
        img_list = sorted(glob.glob(os.path.join(input_path, '*')))

    os.makedirs(output_path, exist_ok=True)
    tmp_output = os.path.join(output_path, 'tmp')
    # print("+++++++++++++++++++++++++++++++",tmp_output)
    os.makedirs(tmp_output, exist_ok=True)
    # ------------------------ set up background upsampler ------------------------

    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.')
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True)  # need to set False in CPU mode


    # ------------------------ set up GFPGAN restorer ------------------------
    # if args.version == '1':
    #     arch = 'original'
    #     channel_multiplier = 1
    #     model_name = 'GFPGANv1'
    #     url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    # elif args.version == '1.2':
    #     arch = 'clean'
    #     channel_multiplier = 2
    #     model_name = 'GFPGANCleanv1-NoCE-C2'
    #     url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    # elif args.version == '1.3':
    #     arch = 'clean'
    #     channel_multiplier = 2
    #     model_name = 'GFPGANv1.3'
    #     url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    # elif args.version == '1.4':
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.4'
    url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    # elif args.version == 'RestoreFormer':
    #     arch = 'RestoreFormer'
    #     channel_multiplier = 2
    #     model_name = 'RestoreFormer'
    #     url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    # else:
    #     raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('pretrained/gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    # print("=======================",img_list)
    # ------------------------ restore ------------------------
    for img_path in img_list:
        # print("=++++++++++++++++++++++++++++++++",img_path)
        # read images
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        print(input_img.shape)
        # print(args.aligned)
        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5)

        extension = ext[1:]
        # print("+++++++++++++++++",output_path)
        save_restore_path = os.path.join(tmp_output, f'{basename}.{extension}')
        imwrite(restored_img, save_restore_path)

    DEFAULT_POS_PROMPT = (
        "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
        "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
        "skin pore detailing, hyper sharpness, perfect without deformations."
    )
    parser = argparse.ArgumentParser("NTIRE2025-RealWorld-Face-Restoration")
    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set")
    parser.add_argument("--input", default=None, type=str, help="input")
    parser.add_argument("--test_dir", default=None, type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="NTIRE2025-RealWorld-Face-Restoration/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument(
        "--output", type=str, help="Path to save restored results."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--task",
        type=str,
        default="face",
        choices=["sr", "face", "denoise", "unaligned_face"],
        help="Task you want to do. Ignore this option if you are using self-trained model.",
    )
    parser.add_argument(
        "--upscale", type=float, default=1, help="Upscale factor of output."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2", "v2.1", "custom"],
        help="DiffBIR model version.",
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="",
        help="Path to training config. Only works when version is custom.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to saved checkpoint. Only works when version is custom.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="spaced",
        # choices=[
        #     "dpm++_m2",
        #     "spaced",
        #     "ddim",
        #     "edm_euler",
        #     "edm_euler_a",
        #     "edm_heun",
        #     "edm_dpm_2",
        #     "edm_dpm_2_a",
        #     "edm_lms",
        #     "edm_dpm++_2s_a",
        #     "edm_dpm++_sde",
        #     "edm_dpm++_2m",
        #     "edm_dpm++_2m_sde",
        #     "edm_dpm++_3m_sde",
        # ],
        help="Sampler type. Different samplers may produce very different samples.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=75,
        help="Sampling steps. More steps, more details.",
    )
    parser.add_argument(
        "--start_point_type",
        type=str,
        choices=["noise", "cond"],
        default="noise",
        help=(
            "For DiffBIR v1 and v2, setting the start point types to 'cond' can make the results much more stable "
            "and ensure that the outcomes from ODE samplers like DDIM and DPMS are normal. "
            "However, this adjustment may lead to a decrease in sample quality."
        ),
    )
    parser.add_argument(
        "--cleaner_tiled",
        action="store_true",
        help="Enable tiled inference for stage-1 model, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cleaner_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cleaner_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--vae_encoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE encoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_encoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--vae_decoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE decoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_decoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tiled",
        action="store_true",
        help="Enable tiled sampling, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cldm_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--captioner",
        type=str,
        choices=["none", "llava", "ram"],
        default="none",
        help="Select a model to describe the content of your input image.",
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default=DEFAULT_POS_PROMPT,
        help=(
            "Descriptive words for 'good image quality'. "
            "It can also describe the things you WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default='low quality, blurry, low-resolution, noisy, unsharp, weird textures',
        help=(
            "Descriptive words for 'bad image quality'. "
            "It can also describe the things you DON'T WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--rescale_cfg",
        action="store_true",
        help="Gradually increase cfg scale from 1 to ('cfg_scale' + 1)",
    )
    parser.add_argument(
        "--noise_aug",
        type=int,
        default=0,
        help="Level of noise augmentation. More noise, more creative.",
    )
    parser.add_argument(
        "--s_churn",
        type=float,
        default=0,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmin",
        type=float,
        default=0,
        help="Minimum sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmax",
        type=float,
        default=300,
        help="Maximum  sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=1,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1,
        help="I don't understand this parameter. Leave it as default.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order of solver. Only works with edm_lms sampler.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="Control strength from ControlNet. Less strength, more creative.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Nothing to say.")
    # guidance parameters
    parser.add_argument(
        "--guidance", action="store_true", help="Enable restoration guidance."
    )
    parser.add_argument(
        "--g_loss",
        type=str,
        default="w_mse",
        choices=["mse", "w_mse"],
        help="Loss function of restoration guidance.",
    )
    parser.add_argument(
        "--g_scale",
        type=float,
        default=0.0,
        help="Learning rate of optimizing the guidance loss function.",
    )
    # common parameters

    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples for each image."
    )
    parser.add_argument("--seed", type=int, default=231)
    # mps has not been tested

    parser.add_argument(
        "--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--llava_bit", type=str, default="4", choices=["16", "8", "4"])
    ops = parser.parse_args()
    print(ops)


    set_seed(ops.seed)
    ops.input = tmp_output
    ops.output = output_path
    # print(args)
    loops = {
        "face": BFRInferenceLoop,
    }
    loops["face"](ops).run()

    print(f'Results are in the [{output_path}] folder.')
    shutil.rmtree(tmp_output, ignore_errors=True)
    # os.system(f'rm -rf {tmp_output}')


if __name__ == "__main__":
   main()