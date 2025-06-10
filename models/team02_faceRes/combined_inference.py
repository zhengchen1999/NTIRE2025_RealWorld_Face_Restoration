import os
import torch
# from gfpgan import GFPGANer
import argparse
import cv2
import glob
import numpy as np
import os
import torch
import sys
import shutil

# sys.path.append(".")
# from test_tsdsr import main as tsdsr_main
# import test_tsdsr.main as tsdsr_main
import argparse
from models.team02_faceRes.test_tsdsr import main as  tsdsr_main
from models.team02_faceRes.gfpgan import GFPGANer



def run_inference(model_dir, input_path, output_path, device):
    gfpgan_model_path = os.path.join(model_dir, "GFPGANv1.3.pth")  # 根据实际路径修改
    restorer = GFPGANer(
        model_path=gfpgan_model_path,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None, )

    temp_output = os.path.join(output_path, 'temp_images')
    os.makedirs(temp_output, exist_ok=True)

    input_images = sorted(glob.glob(os.path.join(input_path, '*.png')))
    for img_path in input_images:
        img_name = os.path.basename(img_path)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _, _, restored_img = restorer.enhance(input_img, has_aligned=False, paste_back=True)
        if restored_img is not None:
            save_restore_path = os.path.join(temp_output, img_name)
            cv2.imwrite(save_restore_path, restored_img)


# TSD-SR 参数设置
    tsdsr_args = argparse.Namespace(
        pretrained_model_name_or_path= os.path.join(model_dir, "models--stabilityai--stable-diffusion-3-medium-diffusers") ,
        lora_dir= model_dir,
        embedding_dir=os.path.join(model_dir, "default"),
        input_dir=temp_output,       
        output_dir=output_path,
        rank=64,
        rank_vae=64,
        is_use_tile=False,
        vae_decoder_tiled_size=224,
        vae_encoder_tiled_size=1024,
        latent_tiled_size=64,
        latent_tiled_overlap=8,
        device=device,
        seed=42,
        upscale=1,
        process_size=512,
        mixed_precision="fp16",
        align_method='wavelet'
    )

    # 调用 TSD-SR 进行进一步增强
    tsdsr_main(tsdsr_args)
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)


