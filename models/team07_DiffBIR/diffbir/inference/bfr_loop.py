import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    load_model_from_path,
    trace_vram_usage,
)
from ..pipeline import SwinIRPipeline
from ..model import SwinIR


class BFRInferenceLoop(InferenceLoop):

    def load_cleaner(self) -> None:
        self.cleaner: SwinIR = instantiate_from_config(
            OmegaConf.load(self.args.train_cfg).model.swinir
        )
        # weight = load_model_from_url(MODELS["swinir_face"])
        weight = load_model_from_path(self.args.swinir_path)
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
