from types import SimpleNamespace

import os
import torch

from accelerate.utils import set_seed
from .diffbir.inference import (
    BSRInferenceLoop,
    BFRInferenceLoop,
    BIDInferenceLoop,
    UnAlignedBFRInferenceLoop,
    CustomInferenceLoop,
)


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    # print(f"using device {device}")
    return device


def main(model_dir=None, input_path=None, output_path=None, device='cuda'):
    args = SimpleNamespace()
    args.task = 'face'
    args.upscale = 1
    args.version = 'v2'
    args.train_cfg = 'models/team11_DiffBIR/configs/train.yaml'
    args.sampler = 'spaced'
    args.steps = 50
    args.start_point_type = 'noise'
    args.cleaner_tiled = False
    args.cleaner_tile_size = 512
    args.cleaner_tile_stride = 256
    args.vae_encoder_tiled = False
    args.vae_encoder_tile_size = 256
    args.vae_decoder_tiled = False
    args.vae_decoder_tile_size = 256
    args.cldm_tiled = False
    args.cldm_tile_size = 512
    args.cldm_tile_stride = 256
    args.captioner = 'none'
    args.pos_prompt = ''
    args.neg_prompt = 'low quality, blurry, low-resolution, noisy, unsharp, weird textures'
    args.cfg_scale = 4
    args.rescale_cfg = False
    args.noise_aug = 0
    args.s_churn = 0
    args.s_tmin = 0
    args.s_tmax = 300
    args.s_noise = 1
    args.eta = 1
    args.order = 1
    args.strength = 1
    args.batch_size = 1
    args.guidance = False
    args.g_loss = 'w_mse'
    args.g_scale = 0.0
    args.n_samples = 1
    args.seed = 231
    args.precision = 'fp32'
    args.llava_bit = '4'

    args.input = input_path
    args.output = output_path

    args.device = check_device(device)
    set_seed(args.seed)

    args.sd_path = os.path.join(model_dir, 'v2-1_512-ema-pruned.ckpt')
    args.ckpt = os.path.join(model_dir, '0080000.pt')
    args.swinir_path = os.path.join(model_dir, 'face_swinir_v1.ckpt')

    # CustomInferenceLoop(args).run()
    BFRInferenceLoop(args).run()
if __name__ == "__main__":
    main()
