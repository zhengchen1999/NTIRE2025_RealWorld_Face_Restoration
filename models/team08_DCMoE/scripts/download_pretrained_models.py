import argparse
import os
from os import path as osp

from basicsr.utils.download_util import load_file_from_url


def download_pretrained_models(method, file_urls):
    if method == 'CodeFormer_train':
        method = 'CodeFormer'
    save_path_root = f'./weights/{method}'
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_url in file_urls.items():
        save_path = load_file_from_url(url=file_url, model_dir=save_path_root, progress=True, file_name=file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'method',
        type=str,
        help=("Options: 'CodeFormer' 'facelib' 'dlib'. Set to 'all' to download all the models."))
    args = parser.parse_args()

    file_urls = {
        'CodeFormer': {
            'codeformer.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        },
        'CodeFormer_train': {
            'vqgan_code1024.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/vqgan_code1024.pth',
            'latent_gt_code1024.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/latent_gt_code1024.pth',
            'codeformer_stage2.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_stage2.pth',
            'codeformer.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        },
        'facelib': {
            # 'yolov5l-face.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5l-face.pth',
            'detection_Resnet50_Final.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
            'parsing_parsenet.pth': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'
        },
        'dlib': {
            'mmod_human_face_detector-4cb19393.dat': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/mmod_human_face_detector-4cb19393.dat',
            'shape_predictor_5_face_landmarks-c4b1e980.dat': 'https://kkgithub.com/sczhou/CodeFormer/releases/download/v0.1.0/shape_predictor_5_face_landmarks-c4b1e980.dat'
        }
    }

    if args.method == 'all':
        for method in file_urls.keys():
            download_pretrained_models(method, file_urls[method])
    else:
        download_pretrained_models(args.method, file_urls[args.method])