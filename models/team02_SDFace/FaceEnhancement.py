import cv2
import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from thop import profile
from .utils_ import utils_faces
from .models.sdxl_arch import SDXL_Turbo

class FaceRestoration(object):
    def __init__(self, ModelDir, device='cuda'):
        
        self.ModelDir = ModelDir
        self.device = device

        self.modelFace = SDXL_Turbo()

        Num_Parameter = utils_faces.print_networks(self.modelFace)
        print('Total Number of Parameters : {:.2f} M'.format(Num_Parameter))
        self.modelFace.load_state_dict(torch.load(ModelDir)['params'], strict=False)
        self.modelFace.eval()
        
        input = torch.randn(1, 3, 512, 512)  # 假设输入尺寸为 (32, 32)

        # 计算 FLOPs
        # flops, params = profile(self.modelFace, inputs=(input,))
        # print(f"FLOPs: {flops}")

        self.modelFace = self.modelFace.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def handle_faces(self, img):
        height, width = img.shape[:2]
        
        LQ = transforms.ToTensor()(img)
        LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)
        LQ = LQ.unsqueeze(0).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            ef = self.modelFace(LQ)
        end_time = time.time()
        print('Single Inference Time: {:.4f}s'.format(end_time - start_time))
        ef = ef * 0.5 + 0.5
        ef = ef.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        ef = np.clip(ef.float().cpu().numpy(), 0, 1) * 255.0
        return ef
