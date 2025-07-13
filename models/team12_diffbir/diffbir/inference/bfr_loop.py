import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import SwinIRPipeline
from ..model import SwinIR
import torch


class BFRInferenceLoop(InferenceLoop):

    def load_cleaner(self) -> None:
        self.cleaner: SwinIR = instantiate_from_config(
            OmegaConf.load("models/team06_diffbir/configs/inference/swinir.yaml")
        )
        pt_number = self.args.swindegrad_pt
        
        # weight = load_model_from_url(MODELS["swinir_face"])
        if self.args.swindegrad == "v0":
            # weight = load_model_from_url(MODELS["swinir_face"])
            weight = torch.load(MODELS["swinir_face"], map_location="cpu")
        elif self.args.swindegrad == "v1":
            weight = torch.load(f"{MODELS['swinir_face_degrad1']}/{pt_number}.pt", map_location="cpu")
        elif self.args.swindegrad == "v2":
            weight = torch.load(f"{MODELS['swinir_face_degrad2']}/{pt_number}.pt", map_location="cpu")
        elif self.args.swindegrad == "v3":
            weight = torch.load(f"{MODELS['swinir_face_degrad3']}/{pt_number}.pt", map_location="cpu")
        elif self.args.swindegrad == "v4":
            weight = torch.load(f"{MODELS['swinir_face_degrad4']}/{pt_number}.pt", map_location="cpu")
        elif self.args.swindegrad == "v5":
            weight = torch.load(f"{MODELS['swinir_face_degrad5']}/{pt_number}.pt", map_location="cpu")
        else:
            raise ValueError("Unsupported degrad version!")
        
        if "state_dict" in weight:
            weight = weight["state_dict"]
        if list(weight.keys())[0].startswith("module"):
            weight = {k[len("module.") :]: v for k, v in weight.items()}
        self.cleaner.load_state_dict(weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        self.pipeline = SwinIRPipeline(
            self.cleaner, self.cldm, self.diffusion, self.cond_fn, self.args.device
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
