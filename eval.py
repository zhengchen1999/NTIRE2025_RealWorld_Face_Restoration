import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from torchvision import transforms
import torchvision.transforms.functional as F
import csv
import pyiqa
import cv2
import numpy as np
from einops import rearrange
import csv

from fid import calculate_fid_folder
from face_alignment.inference import *

def read_csv_to_dict(filename):
    data = {}

    with open(filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]

            data[key] = {
                field: (float(value) if is_number(value) else value)
                for field, value in row.items() if field != csv_reader.fieldnames[0]
            }

    return data

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def rgb_to_ycrcb(tensor):
    tensor_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ycrcb_np = cv2.cvtColor(tensor_np, cv2.COLOR_RGB2YCrCb)
    ycrcb_tensor = torch.tensor(ycrcb_np).permute(2, 0, 1).unsqueeze(0).float()
    return ycrcb_tensor

class IQA:
    def __init__(self, device=None, use_qalign=True):
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.use_qalign = use_qalign

        self.iqa_metrics = {
            'niqe': pyiqa.create_metric('niqe', device=self.device),
            'clipiqa': pyiqa.create_metric('clipiqa', device=self.device), 
            'maniqa': pyiqa.create_metric('maniqa', device=self.device),
            'musiq': pyiqa.create_metric('musiq', device=self.device),
            'qalign': pyiqa.create_metric('qalign', device=self.device) if use_qalign else None
        }
        self.adaface = load_pretrained_model('ir_50').to(self.device)
    
    def calculate_values(self, output_image):
        if type(output_image) == torch.Tensor or type(output_image) == np.ndarray:
            if type(output_image) == np.ndarray:
                output_image = torch.tensor(output_image).contiguous().float()
                
            if len(output_image.shape) == 3:

                output_image = output_image.unsqueeze(0)
                
            if output_image.shape[-1] == 3:
                assert output_image.shape[1] == output_image.shape[2] == 512
                print("Rearranging image dimensions from (N, W, H, C) to (N, C, W, H)")
                output_image = rearrange(output_image, "b h w c -> b c h w").contiguous().float()
            elif output_image.shape[-1] == 4:
                output_image = output_image[:, :, :, :3]
                assert output_image.shape[1] == output_image.shape[2] == 512
                print("Rearranging image dimensions from (N, W, H, C) to (N, C, W, H)")
                output_image = rearrange(output_image, "b h w c -> b c h w").contiguous().float()
                
            output_tensor = output_image.to(self.device)
        else:
            output_tensor = F.to_tensor(output_image).unsqueeze(0).to(self.device)

        try:
            niqe_value = self.iqa_metrics['niqe'](output_tensor)
            clipiqa_value = self.iqa_metrics['clipiqa'](output_tensor)
            maniqa_value = self.iqa_metrics['maniqa'](output_tensor)
            musiq_value = self.iqa_metrics['musiq'](output_tensor)
            qalign_value = self.iqa_metrics['qalign'](output_tensor) if self.use_qalign else None

            result = {}

            result['NIQE'] = niqe_value.item()
            result['CLIP_IQA'] = clipiqa_value.item()
            result['MANIQA'] = maniqa_value.item()
            result['MUSIQ'] = musiq_value.item()
            if qalign_value is not None:
                result['QALIGN'] = qalign_value.item()
        except Exception as e:
            print(f"Error: {e}")
            return None

        return result

def calculate_iqa_for_partition(output_folder, lq_ref_folder, output_files, use_qalign, device, mode, rank):
    iqa = IQA(device=device, use_qalign=use_qalign)
    local_results = {}
    for output_file in tqdm(output_files, total=len(output_files), desc=f"Processing images on GPU {rank}"):
        output_image_path = os.path.join(output_folder, output_file)
        output_image = Image.open(output_image_path)
        fname = os.path.basename(output_image_path).replace(".jpg", ".png")
        lq_path = glob.glob(os.path.join(lq_ref_folder, '**', fname), recursive=True)
        assert len(lq_path) >= 1, "No Matching Image Found"
        lq_path = lq_path[0]

        face_sim, dataset = (
            inference_face(model=iqa.adaface, lq_path=lq_path, pred_path=output_image_path, device=device, mode=mode))
        values = iqa.calculate_values(output_image)
        values['ID_Sim'] = face_sim
        values['Dataset'] = dataset
        if values is not None:
            local_results[os.path.basename(output_file)] = values

    
    return local_results

def main_worker(rank, gpu_id, output_folder, lq_ref_folder, output_files, use_qalign, return_dict, num_gpus, mode):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    partition_size = len(output_files) // num_gpus
    start_idx = rank * partition_size
    end_idx = (rank + 1) * partition_size if rank != num_gpus - 1 else len(output_files)
    
    output_files_partition = output_files[start_idx:end_idx]
    
    local_results = calculate_iqa_for_partition(output_folder, lq_ref_folder,
                                                output_files_partition, use_qalign, device, mode, rank)
    return_dict[rank] = local_results

import argparse
if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='test', choices=['test', 'val'])
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--lq_ref_folder", type=str, default="./NTIRE-FR")
    parser.add_argument("--metrics_save_path", type=str, default="./IQA_results")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--use_qalign", type=str, choices=['True', 'False'], default='True')
    args = parser.parse_args()

    args.use_qalign = args.use_qalign == 'True'
    args.lq_ref_folder = os.path.join(args.lq_ref_folder, args.mode)

    output_files = glob.glob(os.path.join(args.output_folder, '**', '*.png'), recursive=True)
    output_files = [os.path.relpath(file, args.output_folder) for file in output_files]
    print(f"Number of images ：{len(output_files)}")

    manager = mp.Manager()
    return_dict = manager.dict()

    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    print(f"Using GPUs: {args.gpu_ids}")
    num_gpus = len(args.gpu_ids)

    processes = []
    for rank, gpu_id in enumerate(args.gpu_ids):
        p = mp.Process(target=main_worker,
                       args=(rank, gpu_id, args.output_folder, args.lq_ref_folder,
                             output_files, args.use_qalign, return_dict, num_gpus, args.mode))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    results = {}
    for rank in return_dict.keys():
        results.update(return_dict[rank])

    folder_name = os.path.basename(args.output_folder)
    parent_folder = os.path.dirname(args.output_folder)
    next_level_folder = os.path.basename(parent_folder)
    os.makedirs(args.metrics_save_path, exist_ok=True)
    average_results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}_total.csv"
    results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}.csv"

    if results:
        all_keys = set()
        for values in results.values():
            try:
                all_keys.update(values.keys())
            except Exception as e:
                print(f"Error: {e}")

        all_keys = sorted(all_keys)

        thresholds = {
            'CelebA': 0.5,
            'LFW-Test': 0.6,
            'CelebChild-Test': 0.6,
            'Wider-Test': 0.3,
            'WebPhoto-Test': 0.3
        }
        
        dataset_counts = {
            'CelebA': 100,
            'LFW-Test': 100,
            'CelebChild-Test': 50,
            'WebPhoto-Test': 100,
            'Wider-Test': 100
        }

        average_results = {}
        total_below_threshold = 0
        total_metrics = {key: [] for key in next(iter(results.values())).keys() if key != 'Dataset'}

        for image_name, result in results.items():
            dataset = result['Dataset']
            assert dataset in dataset_counts.keys()

            if dataset not in average_results:
                average_results[dataset] = {key: [] for key in result if key != 'Dataset'}

            for key, value in result.items():
                if key != 'Dataset':
                    average_results[dataset][key].append(value)
                    total_metrics[key].append(value)

            if result['ID_Sim'] < thresholds[dataset]:
                if 'low_ID_Sim' not in average_results[dataset]:
                    average_results[dataset]['low_ID_Sim'] = 0
                average_results[dataset]['low_ID_Sim'] += 1
                total_below_threshold += 1

        for dataset, metrics in average_results.items():
            for metric, values in metrics.items():
                if metric != 'low_ID_Sim':
                    average_results[dataset][metric] = np.mean(values)

        total_images = sum(dataset_counts.values())
        average_results["All"] = {}
        average_results["All"]['FID'] = calculate_fid_folder(args.output_folder)
        for metric, values in total_metrics.items():
            weighted_sum = 0
            for dataset, count in dataset_counts.items():
                weighted_sum += np.mean(average_results[dataset][metric]) * count
            average_results["All"][metric] = weighted_sum / total_images

        average_results["All"]['low_ID_Sim'] = total_below_threshold

        average_results["Weighted Score"] = {}
        average_results["Weighted Score"]['Total'] = 0
        for metric, value in average_results['All'].items():
            if metric == 'low_ID_Sim' or metric == 'ID_Sim':
                continue
            if metric == 'FID':
                average_results["Weighted Score"]['FID'] = max(0, (100 - value) / 100)    # FID is a lower-is-better metric
            elif metric == 'NIQE':
                average_results["Weighted Score"]['NIQE'] = max(0, (10 - value) / 10)     # NIQE is a lower-is-better metric
            elif metric == 'CLIP_IQA':
                average_results["Weighted Score"]['CLIP_IQA'] = value
            elif metric == 'MANIQA':
                average_results["Weighted Score"]['MANIQA'] = value
            elif metric == 'MUSIQ':
                average_results["Weighted Score"]['MUSIQ'] = value / 100
            elif metric == 'QALIGN':
                average_results["Weighted Score"]['QALIGN'] = value / 5
            else:
                print(f"Unknown metric: {metric}")
        average_results["Weighted Score"]['Total'] = sum(average_results["Weighted Score"].values())
        
        for dataset, metrics in average_results.items():
            print(f"Average results for {dataset}:")
            for metric, avg_value in metrics.items():
                if metric == 'low_ID_Sim':
                    if dataset == 'All' :
                        print(f"  Low ID Similarity: {avg_value}/{total_images}")
                    else:
                        print(f"  Low ID Similarity: {avg_value}/{dataset_counts[dataset]}")
                else:
                    print(f"  {metric}: {avg_value:.4f}")
        
        with open(average_results_filename, 'w', newline='') as f:
            writer = csv.writer(f)

            header = ['Dataset', 'NIQE', 'CLIP_IQA', 'MANIQA', 'MUSIQ', 'QALIGN', 'FID', 'ID_Sim', 'low_ID_Sim', 'Total']
            writer.writerow(header)

            for dataset, metrics in average_results.items():
                row = [dataset] + [metrics.get(key, 0) for key in header[1:]]
                writer.writerow(row)
            print(f"Average IQA results and Weighted Score have been saved to {average_results_filename} file")

        with open(results_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename'] + list(all_keys))
            for filename, values in results.items():
                row = [filename] + [values.get(key, 0) for key in all_keys]
                writer.writerow(row)
            print(f"IQA results have been saved to {results_filename} file")
