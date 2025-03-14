import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

# timestep = 0

def get_aligned_face(image_path, rgb_pil_image=None):
    # global timestep
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    
    # # Please just resize 512*512 image to 112*112
    # img_resized = img.resize((112, 112))
    # # img_resized = img_resized.convert('RGB')
    # img_resized.save(f'aligned{timestep}.jpg')
    # # return img_resized
    
    # find face
    try:
        bbox, facial5points, face = mtcnn_model.align(img)

    except Exception as e:
        # print('Face detection Failed due to error.')
        # print(e)
        # img.save(f'face_detection_failed{datetime.now()}.jpg')
        face = img
        bbox = None
        facial5points = None

    # print(face)
    # face.save(f'aligned_face{timestep}.jpg')
    # timestep += 1

    # return img_resized
    return bbox, facial5points, face


