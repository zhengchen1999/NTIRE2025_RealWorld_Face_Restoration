from typing import overload, Literal
import re

from PIL import Image
import torch


from .models import ram_plus
from . import inference_ram as inference
from . import get_transform




class RAMCaptioner():
    def __init__(self, pretrained_path, text_encoder_type, device):
        super().__init__()
        image_size = 384
        self.device = device
        transform = get_transform(image_size=image_size)
        # pretrained = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
        # pretrained = "/home/work/TEST_SRCB/jx2018.zhang/weights/ram_plus_swin_large_14m.pth"
        model = ram_plus(pretrained=pretrained_path, text_encoder_type = text_encoder_type, image_size=image_size, vit="swin_l")
        model.eval()
        model = model.to(device)

        self.transform = transform
        self.model = model

    def __call__(self, image: Image.Image) -> str:
        image = self.transform(image).unsqueeze(0).to(self.device)
        res = inference(image, self.model)
        # res[0]: armchair | blanket | lamp | ...
        # res[1]: 扶手椅  | 毯子/覆盖层 | 灯  | ...
        return res[0].replace(" | ", ", ")
