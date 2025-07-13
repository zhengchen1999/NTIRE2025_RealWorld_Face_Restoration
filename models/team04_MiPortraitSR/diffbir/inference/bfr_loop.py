import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import torch


from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import MiFRPipeline
# from ..model import SwinIR
from ...GFPGAN.gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from ...PiSA.pisasr import PiSASR_eval
from ...FFANet.models import *
from ...FTN.face_tone_net import *


class MixedCleaner:
    def __init__(
        self,
        model_dir,
        pisa_cfg,
        gfpgan_cfg,
        ffanet_cfg,
        device
    ):
        if pisa_cfg is not None:
            self.pisa_cfg = pisa_cfg
            #self.pisa_cfg.pretrained_model_path = os.path.join(model_dir, self.pisa_cfg.pretrained_model_path)
            self.pisa_cfg.pretrained_path = os.path.join(model_dir, self.pisa_cfg.pretrained_path)
            self.pisa = PiSASR_eval(pisa_cfg)
            self.pisa.set_eval()
        
        if gfpgan_cfg is not None:
            self.gfpgan_cfg = gfpgan_cfg

            # from basicsr.archs.rrdbnet_arch import RRDBNet
            # from realesrgan import RealESRGANer
            # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            # self.bg_upsampler = RealESRGANer(
            #     scale=2,
            #     model_path='https://kkgithub.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            #     model=model,
            #     tile=self.gfpgan_cfg.bg_tile,
            #     tile_pad=10,
            #     pre_pad=0,
            #     half=True)  # need to set False in CPU mode
            
            if self.gfpgan_cfg.version == '1.2':
                channel_multiplier = 2
                model_name = 'GFPGANCleanv1-NoCE-C2'
            elif self.gfpgan_cfg.version == '1.3':
                channel_multiplier = 2
                model_name = 'GFPGANv1.3'
            elif self.gfpgan_cfg.version == '1.4':
                channel_multiplier = 2
                model_name = 'GFPGANv1.4'

            model_path = os.path.join(model_dir, model_name + '.pth')
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
            loadnet = torch.load(model_path)
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            else:
                keyname = 'params'
            self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
            self.gfpgan.eval()
            self.gfpgan = self.gfpgan.to(device)

        if ffanet_cfg is not None:
            self.ffanet_cfg = ffanet_cfg
            self.ffanet_cfg.model_path = os.path.join(model_dir, self.ffanet_cfg.model_path)
            ckp=torch.load(self.ffanet_cfg.model_path, map_location=device)
            net = FFA(gps=self.ffanet_cfg.gps, blocks=self.ffanet_cfg.blocks).to(device)
            net = torch.nn.DataParallel(net)
            net.load_state_dict(ckp['model'])
            net.eval()
            self.ffanet = net

class BFRInferenceLoop(InferenceLoop):

    def load_cleaner(self) -> None:
        self.cleaner = MixedCleaner(self.args.model_dir, self.pisa_args, self.gfpgan_args, self.ffanet_args, self.args.device)
    
    def load_pipeline(self) -> None:

        if self.ftn_args is not None:
            self.ftn = FaceToneNet(
                input_nc=self.ftn_args.input_nc,
                output_nc=self.ftn_args.output_nc,
                num_downs=self.ftn_args.num_downs,
                ngf=self.ftn_args.ngf,
                norm_layer=get_norm_layer(norm_type='instance'),
                use_dropout=False
            ).to(self.args.device)
            self.ftn.load_state_dict(torch.load(os.path.join(self.args.model_dir, self.ftn_args.model_path), map_location=self.args.device))
            self.ftn.eval()
        else:
            self.ftn = None

        self.pipeline = MiFRPipeline(
            self.cleaner, self.cldm, self.diffusion, self.cond_fn, self.args.device, 
            self.ftn, self.enhance
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
