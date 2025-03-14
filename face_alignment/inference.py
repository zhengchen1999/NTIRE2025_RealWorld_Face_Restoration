from . import net
import torch
import os
from face_alignment import align
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from datetime import datetime
import argparse
import time
import glob
from tqdm import tqdm

from face_alignment.mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face


adaface_models = {
    'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], weights_only=False)['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    model.cuda()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(brg_img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def check_existing(filename, df):
    matched_rows = df[df['filename'] == filename]
    if not matched_rows.empty:
        return matched_rows.iloc[0].to_dict()
    return None

def inference(test_image_path, pred_image_path, points_csv, crop_size=(112, 112)):
    model = load_pretrained_model('ir_50')

    sim_dict = {
        'CelebA': [],
        'LFW-Test': [],
        'CelebChild-Test': [],
        'Wider-Test': [],
        'WebPhoto-Test': [],
    }

    df_proc = pd.read_csv(points_csv)
    refrence = get_reference_facial_points(default_square=crop_size[0] == crop_size[1])

    png_files = glob.glob(os.path.join(test_image_path, '**', '*.png'), recursive=True)

    png_filenames = [os.path.basename(f) for f in png_files]

    for i, fname in tqdm(enumerate(png_filenames)):

        lq_path = png_files[i]

        pred_paths = glob.glob(os.path.join(pred_image_path, '**', fname), recursive=True)
        assert fname in lq_path, f"{fname} not in {lq_path}"

        if not pred_paths:
            print(f"No match image: {fname}")
            continue
        pred_path = pred_paths[0]

        if existing_record := check_existing(fname.split('.')[0], df_proc):
            facial5points = [
                [existing_record["left_eye_x"], existing_record["left_eye_y"]],
                [existing_record["right_eye_x"], existing_record["right_eye_y"]],
                [existing_record["nose_x"], existing_record["nose_y"]],
                [existing_record["mouth_left_x"], existing_record["mouth_left_y"]],
                [existing_record["mouth_right_x"], existing_record["mouth_right_y"]]
            ]
            dataset = existing_record['dataset']

        else:
            print(f"Lost image: {fname}")
            break

        lq_img = Image.open(lq_path).convert('RGB')
        pred_img = Image.open(pred_path).convert('RGB')
        lq_aligned = warp_and_crop_face(np.array(lq_img), facial5points, refrence, crop_size=crop_size)
        pred_aligned = warp_and_crop_face(np.array(pred_img), facial5points, refrence, crop_size=crop_size)

        lq_input = to_input(lq_aligned).cuda()
        pred_input = to_input(pred_aligned).cuda()

        lq_feature, _ = model(lq_input)
        pred_feature, _ = model(pred_input)

        similarity = (lq_feature @ pred_feature.T).item()
        sim_dict[dataset].append([fname, similarity])

    return sim_dict

def inference_face(model, lq_path, pred_path, device, mode="test", crop_size=(112, 112)):

    if mode == "val":
        points_csv = "face_alignment/val_face.csv"
    elif mode == "test":
        points_csv = "face_alignment/test_face.csv"
    df_proc = pd.read_csv(points_csv)

    refrence = get_reference_facial_points(default_square=crop_size[0] == crop_size[1])

    lq_img = Image.open(lq_path).convert('RGB')
    pred_img = Image.open(pred_path).convert('RGB')

    fname = os.path.basename(lq_path)
    if existing_record := check_existing(fname.split('.')[0], df_proc):
        facial5points = [
            [existing_record["left_eye_x"], existing_record["left_eye_y"]],
            [existing_record["right_eye_x"], existing_record["right_eye_y"]],
            [existing_record["nose_x"], existing_record["nose_y"]],
            [existing_record["mouth_left_x"], existing_record["mouth_left_y"]],
            [existing_record["mouth_right_x"], existing_record["mouth_right_y"]]
        ]
        dataset_name = existing_record['dataset']
    else:
        print(f"Unkown image: {fname}")
        return None

    lq_aligned = warp_and_crop_face(np.array(lq_img), facial5points, refrence, crop_size=crop_size)
    pred_aligned = warp_and_crop_face(np.array(pred_img), facial5points, refrence, crop_size=crop_size)

    lq_input = to_input(lq_aligned).to(device)
    pred_input = to_input(pred_aligned).to(device)

    lq_feature, _ = model(lq_input)
    pred_feature, _ = model(pred_input)

    similarity = (lq_feature @ pred_feature.T).item()

    return similarity, dataset_name

def filter_images_by_threshold(dataset_name, data):
    thresholds = {
        'CelebA': 0.7,
        'LFW-Test': 0.7,
        'CelebChild-Test': 0.7,
        'debug': 0.7,
        'Wider-Test': 0.3,
        'WebPhoto-Test': 0.3
    }

    if dataset_name == 'all':
        for dataset in data.keys():
            threshold = thresholds.get(dataset, 0.5)
            low_score_images = []
            total_score = 0
            total_images = 0

            for image_name, score in data[dataset]:
                total_score += score
                total_images += 1

                if score < threshold:
                    low_score_images.append([image_name, score])

            average_score = total_score / total_images if total_images > 0 else 0

            print(f"Dataset: {dataset}")
            print(f"Total low score images: {len(low_score_images)}")
            print("Low score images:")
            for img_score in low_score_images:
                print(img_score[0], ":", img_score[1])
            print(f"Average score for {dataset}: {average_score:.4f}")
            print("=" * 50)
    else:
        threshold = thresholds.get(dataset_name, 0.5)
        low_score_images = []
        total_score = 0
        total_images = 0

        for image_name, score in data[dataset_name]:
            total_score += score
            total_images += 1

            if score < threshold:
                low_score_images.append([image_name, score])

        average_score = total_score / total_images if total_images > 0 else 0

        print(f"Dataset: {dataset_name}")
        print(f"Total low score images: {len(low_score_images)}")
        print("Low score images:")
        for img_score in low_score_images:
            print(img_score[0], ":", img_score[1])
        print(f"Average score for {dataset_name}: {average_score:.4f}")


if __name__ == '__main__':
    dict = inference(
        test_image_path=f'./NTIRE-FR/test',
        pred_image_path=f'/data/user/gj/DFOSD/test_ntire',
        points_csv='face_alignment/test_face.csv'
    )
    filter_images_by_threshold("all", dict)
