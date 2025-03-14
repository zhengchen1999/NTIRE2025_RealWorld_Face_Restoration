# This code is derived from the VQFR project:
# https://github.com/TencentARC/VQFR/blob/main/metric_paper/calculate_fid_folder.py

import argparse
import math
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import glob

from VQFR.vqfr.data import build_dataset
from VQFR.vqfr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3

# backend='disk' or 'lmdb', io backend for dataset.
def calculate_fid_folder(restored_folder, batch_size=64, num_sample=3000, num_workers=4, backend='disk'): 
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    # inception model
    inception = load_patched_inception_v3(device)
    print(f"Number of images: {len(glob.glob(os.path.join(restored_folder, '**', '*.png'), recursive=True))}")
    # create dataset
    opt = {}
    opt['name'] = 'SingleImageDataset'
    opt['type'] = 'SingleImageDataset'
    opt['dataroot_lq'] = restored_folder
    opt['io_backend'] = dict(type=backend)
    opt['mean'] = [0.5, 0.5, 0.5]
    opt['std'] = [0.5, 0.5, 0.5]
    dataset = build_dataset(opt)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        drop_last=False)
    num_sample = min(num_sample, len(dataset))
    total_batch = math.ceil(num_sample / batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['lq']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, device, total_batch)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:num_sample]
    print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load('pretrained/inception_FFHQ_512.pth', weights_only=False)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    return fid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored_folder', type=str, required=True)
    args = parser.parse_args()
    fid = calculate_fid_folder(args.restored_folder) 
    print(f'Restored Folder: {args.restored_folder}, FID: {fid}')