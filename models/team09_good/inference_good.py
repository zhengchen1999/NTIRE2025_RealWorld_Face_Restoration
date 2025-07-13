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
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
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


    set_seed(args.seed)
    args.input = tmp_output
    args.output = output_path
    # print(args)
    loops = {
        "face": BFRInferenceLoop,
    }
    loops["face"](args).run()

    print(f'Results are in the [{output_path}] folder.')
    shutil.rmtree(tmp_output, ignore_errors=True)
    # os.system(f'rm -rf {tmp_output}')


if __name__ == '__main__':
    main()
