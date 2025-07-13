import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from glob import glob
from pathlib import Path

def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--is_module", default=False)
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_lpl", default=2.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_disc", type=float, default=0.04)

    # args for the `t` test
    parser.add_argument("--timesteps1", default=1, type=float)
    parser.add_argument("--timesteps2", default=-1, type=float)

    # args for the degrad/content optimization process
    parser.add_argument("--degrad_iter", default=5, type=float)
    parser.add_argument("--content_iter", default=12000, type=float)

    # args for the latent perceptual loss
    parser.add_argument("--is_middle_attn_lpl", action="store_true")
    parser.add_argument("--is_encoder_attn_lpl", action="store_true")
    parser.add_argument("--is_decoder_attn_lpl", action="store_true")
    parser.add_argument("--is_middle_blk_lpl", action="store_true")
    parser.add_argument("--is_encoder_blk_lpl", action="store_true")
    parser.add_argument("--is_decoder_blk_lpl", action="store_true")

    # args for the latent discriminator loss
    parser.add_argument("--is_middle_attn_disc", action="store_true")
    parser.add_argument("--is_encoder_attn_disc", action="store_true")
    parser.add_argument("--is_decoder_attn_disc", action="store_true")
    parser.add_argument("--is_middle_blk_disc", action="store_true")
    parser.add_argument("--is_encoder_blk_disc", action="store_true")
    parser.add_argument("--is_decoder_blk_disc", action="store_true")

    parser.add_argument("--eval_freq", type=int, default=10,)

    # dataset options
    parser.add_argument("--dataset_txt_paths", default='/home/notebook/data/personal/S9048296/Datasets/LSDIR_FFHQ_gt/musiq_76/txt/musiq_76.txt', type=str)
    parser.add_argument("--null_text_ratio", default=0., type=float)
    parser.add_argument("--tracker_project_name", type=str, default="osediff")

    parser.add_argument("--training_input_folder", default="")
    parser.add_argument("--training_target_folder", default="")
    parser.add_argument("--test_input_folder", default="")
    parser.add_argument("--test_target_folder", default="")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path", default='/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-1-base')
    parser.add_argument("--pretrained_model_path", default='/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-1-base')
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # resume
    parser.add_argument("--resume_ckpt", default=None, type=str)

    # training details
    parser.add_argument("--output_dir", default='experience/oup')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--use_online_deg", action="store_false",)
    parser.add_argument("--deg_file_path", default="params.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')

    # args for the vsd training
    parser.add_argument("--pretrained_model_name_or_path_vsd", default='/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-1-base', type=str)
    parser.add_argument("--snr_gamma_vsd", default=None)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--min_dm_step_ratio", default=0.02, type=float)
    parser.add_argument("--max_dm_step_ratio", default=0.98, type=float)
    parser.add_argument("--min_lpl_step_ratio", default=0.02, type=float)
    parser.add_argument("--max_lpl_step_ratio", default=0.98, type=float)
    parser.add_argument("--neg_prompt_vsd", default="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)
    parser.add_argument("--pos_prompt_vsd", default="", type=str)
    parser.add_argument("--cfg_vsd", default=1.0, type=float)
    parser.add_argument("--change_max_ratio_iter", default=100000, type=int)
    parser.add_argument("--change_max_dm_step_ratio", default=0.50, type=float)

    ## discriminator
    parser.add_argument("--disc_condition", type=float, default=499)
    parser.add_argument("--add_noise_con", default=True, help="")

    # # unet lora setting
    parser.add_argument("--use_unet_encode_lora", action="store_true",)
    parser.add_argument("--use_unet_decode_lora", action="store_true",)
    parser.add_argument("--use_unet_middle_lora", action="store_true",)
    parser.add_argument("--lora_rank_unet", default=4, type=int)
    parser.add_argument("--lora_rank_unet_degrad", default=4, type=int)
    parser.add_argument("--lora_rank_unet_content", default=4, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


import sys
import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation
# from src.datasets.realesrgan_light import RealESRGAN_degradation
class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            # self.crop_preproc = transforms.Compose([
            #     transforms.RandomCrop((args.resolution_ori_height, args.resolution_ori_width)),
            #     transforms.Resize((args.resolution_tgt_height, args.resolution_tgt_width)),
            #     transforms.RandomHorizontalFlip(),
            # ])
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((512, 512)),
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
            ])

            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]

        elif split == 'test':
            # dataset_folder = '/home/notebook/data/group/aigc_share_group_data/LowLevelLLM/DataSets/StableSR_testsets/RealSRVal_crop128'
            dataset_folder = '/home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/SunLingchen/onestep_2lora/output/project/30x_lr'
            self.input_folder = dataset_folder
            self.output_folder = dataset_folder
            # self.input_folder = os.path.join(dataset_folder, "test_SR_bicubic")
            # self.output_folder = os.path.join(dataset_folder, "test_HR")
            self.lr_list = []
            self.gt_list = []
            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))
            self.crop_preproc = transforms.Compose([
                transforms.Resize((768,576)),
            ])
            self.T = build_transform("no_resize")
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["degradation_prompt"] = "remove degradation"
            example["enhancement_prompt"] = "do enhancement"
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            input_img = self.crop_preproc(input_img)
            output_img = self.crop_preproc(output_img)
            # input images scaled to -1, 1
            img_t = self.T(input_img)
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = self.T(output_img)
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["degradation_prompt"] = "remove degradation"
            example["enhancement_prompt"] = "do enhancement"
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        self.image_size = args.image_size if hasattr(args, 'image_size') else 512
        
        if split == 'train':
            self.input_folder = Path(args.training_input_folder)
            self.target_folder = Path(args.training_target_folder)
        elif split == 'test':
            self.input_folder = Path(args.test_input_folder)
            self.target_folder = Path(args.test_target_folder)
        else:
            raise ValueError(f"Invalid split: {split}")

        self.input_files = sorted(self.input_folder.glob('*.png'))
        self.target_files = sorted(self.target_folder.glob('*.png'))
        assert len(self.input_files) == len(self.target_files), "Mismatch in number of input and target images"

        self.transform = self._get_transforms()

    def _get_transforms(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomCrop((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])


    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        target_path = self.target_files[idx]

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        if self.split == 'train':
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            torch.manual_seed(seed)
            target_img = self.transform(target_img)
        else:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        example = {
            "neg_prompt": self.args.neg_prompt_vsd,
            "null_prompt": "",
            "degradation_prompt": "remove degradation",
            "enhancement_prompt": "do enhancement",
            "output_pixel_values": target_img,
            "conditioning_pixel_values": input_img
        }

        if self.split == 'test':
            example["base_name"] = input_path.name

        return example