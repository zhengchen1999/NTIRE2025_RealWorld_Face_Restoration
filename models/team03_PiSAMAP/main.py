import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from .pisasr import PiSASR_eval
from models.team03_PiSAMAP.src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import glob
from models.team03_PiSAMAP.MAP_diffusion_latent_clean import gen, quality_loss_fn
from models.team03_PiSAMAP.helper_functions import clip_tokenizer, clip, device, torch_dtype
from torch import autocast

null_prompt = ''
cond_prompt = ''
with autocast(device):
    tokens_unconditional = clip_tokenizer(null_prompt, padding="max_length",
                                          max_length=clip_tokenizer.model_max_length,
                                          truncation=True, return_tensors="pt",
                                          return_overflowing_tokens=True)
    embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

    tokens_conditional = clip_tokenizer(cond_prompt, padding="max_length",
                                        max_length=clip_tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt",
                                        return_overflowing_tokens=True)
    embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

    embedding_unconditional = embedding_unconditional.to(torch_dtype)
    embedding_conditional = embedding_conditional.to(torch_dtype)


def main(model_dir, input_path, output_path, device):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets', help="path to the input image")
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test',
                        help="the directory to save the output")
    # parser.add_argument("--pretrained_model_path", type=str,
    #                     default='model_zoo/team99_PiSA-SR/stable-diffusion-2-1-base')
    # parser.add_argument('--pretrained_path', type=str, default='model_zoo/team99_PiSA-SR/preset/models/pisa_sr.pkl',
    #                     help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    # parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?")
    parser.add_argument("--default", type=bool, default=True, help="use default or adjustale setting?")

    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set")
    parser.add_argument("--test_dir", default='/home/redpanda/codebase/PiSA-SR-main/preset/NTIRE2025_TEST/test', type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="NTIRE2025-RealWorld-Face-Restoration/results", type=str)
    parser.add_argument("--model_id", default=9, type=int)

    parser.add_argument("--device", type=str,
                        default='cuda:0')

    args = parser.parse_args()
    args.pretrained_model_path = os.path.join('pretrained', 'stable-diffusion-2-1-base')
    args.pretrained_path = os.path.join(model_dir, 'preset/models/pisa_sr.pkl')

    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    # Set BIQA
    grad_scale = 1
    biqa = 'hybrid'
    model_guidance_dict = {'quality_model_str': biqa}
    loss_fn = quality_loss_fn(quality_model_str=model_guidance_dict.get('quality_model_str', biqa),
                              device=device, grad_scale=grad_scale)

    # Get all input images
    if os.path.isdir(input_path):
        image_names = sorted(glob.glob(f'{input_path}/*.png'))
    else:
        image_names = [input_path]

    # Make the output directory
    os.makedirs(output_path, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    #time_records = []
    for image_name in image_names:
        model = model.to(device)
        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        #################
        print('########### Stage 1: PiSA-SR ###########')

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        # print(f"Inference time: {inference_time:.4f} seconds")
        # time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        #output_pil.save(os.path.join(output_path, bname))
        model = model.to('cpu')
        print('########### Stage 2: MAP post-enhancement ###########')
        diffusion_steps = 20
        quality_lambda = 1

        output_pil = gen(
            quality_model_str=biqa,
            grad_scale=grad_scale,
            renormalize_latents=False,
            tied_latents=False,
            num_traversal_steps=30,
            steps=diffusion_steps,
            mix_weight=0.93 ** (50 / diffusion_steps),
            # mix_weight=0.99,
            source_im=output_pil,
            loss_fn=loss_fn,
            embedding_unconditional=embedding_unconditional,
            embedding_conditional=embedding_conditional,
            quality_lambda=quality_lambda,
            clip_grad_val=1e-3,
            )
        output_pil.save(os.path.join(output_path, bname))

    # # Calculate the average inference time, excluding the first few for stabilization
    # if len(time_records) > 3:
    #     average_time = np.mean(time_records[3:])
    # else:
    #     average_time = np.mean(time_records)
    # print(f"Average inference time: {average_time:.4f} seconds")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     #parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets', help="path to the input image")
#     #parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
#     parser.add_argument("--pretrained_model_path", type=str, default='model_zoo/team99_PiSA-SR/stable-diffusion-2-1-base')
#     parser.add_argument('--pretrained_path', type=str, default='model_zoo/team99_PiSA-SR/preset/models/pisa_sr.pkl', help="path to a model state dict to be used")
#     parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
#     parser.add_argument("--process_size", type=int, default=512)
#     parser.add_argument("--upscale", type=int, default=4)
#     parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
#     parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
#     parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
#     parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
#     parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
#     parser.add_argument("--latent_tiled_size", type=int, default=96)
#     parser.add_argument("--latent_tiled_overlap", type=int, default=32)
#     parser.add_argument("--mixed_precision", type=str, default="fp16")
#     #parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?")
#     parser.add_argument("--default", type=bool, default=True, help="use default or adjustale setting?")
#
#     parser.add_argument("--model_dir", type=str,
#                         default='model_zoo/team99_PiSA-SR')
#     parser.add_argument('--input_path', '-i', type=str, default='preset/test_datasets', help="path to the input image")
#     parser.add_argument('--output_path', '-o', type=str, default='experiments/test',
#                         help="the directory to save the output")
#     parser.add_argument("--device", type=str,
#                         default='cuda:0')
#
#     args = parser.parse_args()
#     args.pretrained_model_path = os.path.join(args.model_dir, 'stable-diffusion-2-1-base')
#     args.pretrained_path = os.path.join(args.model_dir, 'preset/models/pisa_sr.pkl')
#     # Call the processing function
#     main(args)