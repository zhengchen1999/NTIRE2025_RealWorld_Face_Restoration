import os
import cv2
import argparse
import glob
import re
import torch
from torchvision.transforms.functional import normalize

from omegaconf import OmegaConf
from torchvision import transforms
from accelerate.utils import set_seed

from .util import imwrite, img2tensor, tensor2img, load_model_from_url
from .FidelityGenerationModel import FidelityModel
from .ldm.cldm import ControlLDM
from .ldm.gaussian_diffusion import Diffusion
from .ldm.pipeline import DiffusionPipe
from .ram.caption import RAMCaptioner


def main(model_dir, input_path, output_path, device):

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-i', '--input_path', type=str, default='/home/work/TEST_SRCB/jx2018.zhang/dataset/NTIRE2025_Challenge/test_1folder', 
    #                     help='Input image, video or folder. Default: inputs/')
    # parser.add_argument('-o', '--output_path', type=str, default="/home/work/TEST_SRCB/jx2018.zhang/dataset/NTIRE2025_Challenge/result/final_test",
    #                     help='Output folder. Default: results/<input_name>')
    # parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])    
    # parser.add_argument('--model_dir', type=str, default='weights', help=' the pretrained model. Participants are expected to save their pretrained model in')

    pos_prompt = ''
    neg_prompt = 'low quality, blurry, low-resolution, noisy, unsharp, weird textures, artifacts'
    steps: int = 50

    fidelity_path = 'fidelity_model.pth'
    natural_path = 'naturalness_model.pt'
    sd_path = 'v2-1_512-ema-pruned.ckpt'
    control_path = 'v2.pth'
    caption_path = 'ram_plus_swin_large_14m.pth'
    text_encoder_type = 'bert-base-uncased'
    seed: int = 231

    # args = parser.parse_args()
    # print(args)    

    set_seed(seed)

    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    # ------------------------ input & output ------------------------
    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [input_path]
    else: # input img folder
        if input_path.endswith('/'):  # solve when path ends with /
            input_path = input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))


    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/ is found...\n')

        
    # ------------------ set up models -------------------
    # Fidelity Model
    fidelity_model = FidelityModel(
                            out_size=512,
                            num_style_feat=512,
                            channel_multiplier=1,
                            decoder_load_path=None,
                            fix_decoder=True,
                            num_mlp=8,
                            input_is_latent=True,
                            different_w=True,
                            narrow=1,
                            sft_half=False)
    checkpoint = torch.load(model_dir+"/"+fidelity_path)
    fidelity_model.load_state_dict(checkpoint)
    fidelity_model.eval().to(device)
    print(f"load fidelity model weight")

    # Enhance Model
    enhance_model = torch.load(model_dir+"/"+natural_path)
    enhance_model.eval().to(device)
    print(f"load naturalness model weight")

    # ControNet
    ldm_config = OmegaConf.load(model_dir.replace("_zoo", "s")+"/ldm/cldm.yaml").get("params", dict())    
    control_net = ControlLDM(ldm_config["unet_cfg"], ldm_config["vae_cfg"], ldm_config["clip_cfg"], \
                                    ldm_config["controlnet_cfg"], ldm_config["latent_scale_factor"])    
    sd_weight = load_model_from_url(model_dir+"/"+sd_path)
    control_net.load_pretrained_sd(sd_weight)
    control_weight = load_model_from_url(model_dir+"/"+control_path)
    control_net.load_controlnet_from_ckpt(control_weight)
    control_net.eval().to(device)
    control_net.cast_dtype(torch.float32)    

    # Diffusion Model
    diffusion = Diffusion()
    diffusion.to(device)

    # Caption Model
    captioner = RAMCaptioner(pretrained_path = model_dir+"/"+caption_path, \
                             text_encoder_type = model_dir+"/"+text_encoder_type, \
                             device=device)
    print(f"load realness model weight")


    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        # clean all the intermediate results to process the next image

        img_name = os.path.basename(img_path)

        basename, ext = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)


        # the input faces are already cropped and aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)


        img = img2tensor(img / 255., bgr2rgb=True, float32=True)
        normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img = img.unsqueeze(0).to(device)


        with torch.no_grad():     

            s1, _  = fidelity_model(img)  
            s1 = torch.clamp(s1, min = -1, max = 1)

 
            prompt = captioner(transforms.ToPILImage()(s1.squeeze(0)))
            pos_prompt = ", ".join([text for text in [prompt, pos_prompt] if text])
            neg_prompt = neg_prompt
            s2 = DiffusionPipe(control_net, diffusion, (s1+1.0)/2.0, steps, pos_prompt, neg_prompt, device)
            s2 = torch.clamp(s2, min = 0, max = 1)

            s3 = enhance_model(s2) 
            s3 = torch.clamp(s3, min = 0, max = 1)
            restored_face = tensor2img(s3, rgb2bgr=True, min_max=(0, 1))

            save_face_name = f'{basename}.png'
            save_restore_path = os.path.join(output_path, save_face_name)
            imwrite(restored_face, save_restore_path)

    print(f'\nAll results are saved in {output_path}')

if __name__ == '__main__':
    main(args)