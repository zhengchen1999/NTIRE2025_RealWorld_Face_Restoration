import os.path
import logging
import torch

from .utils_ import utils_logger
from .utils_ import utils_image as util
from PIL import Image, ImageOps
from collections import OrderedDict
from .FaceEnhancement import FaceRestoration
import cv2
import numpy as np
import time

def main(model_dir, input_path, output_path, device='cuda'):
  
    FaceModel = FaceRestoration(ModelDir=model_dir, device=device)###lxm
    logger = logging.getLogger('blind_face_sr_log')

    torch.cuda.empty_cache()
    idx = 0
    test_results = OrderedDict()
    test_results['runtime'] = []

    for img in util.get_image_paths(input_path):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = Image.open(img).convert('RGB')
        img_L = np.array(img_L)

        # --------------------------------
        # (2) inference
        # --------------------------------

        start_time = time.time()

        img_E = FaceModel.handle_faces(img=img_L)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        test_results['runtime'].append((end_time - start_time) * 1000)  # milliseconds

        cv2.imwrite(os.path.join(output_path, img_name+'.png'), img_E)

        ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
        logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(input_path, ave_runtime))

# Single Inference Time: 0.3790s