import csv
import os
import subprocess
from pathlib import Path
import shutil  
from .usm_sharp_advanced import usm_sharp_advanced
import cv2
import numpy as np

# USM 应用函数
def apply_usm(img, params):
    """
    应用 USM 算法。如果 params 为 None 或不需要 USM，则直接返回原图。
    """
    if params is None or not params.get('apply', True):  # 判断是否应用 USM
        return img  # 不做 USM 处理，直接返回原图
    return usm_sharp_advanced(
        img,
        weight=params['weight'],
        contrast=params['contrast'],
        radius=params['radius'],
        threshold=params['threshold'],
        edges_only=params['edges_only'],
        edge_radius=params['edge_radius'],
        scale=params['scale']
    )

# 定义 USM 参数
usm_params = {
    "no_usm": None,
    "0.1": {'weight': 0.1, 'contrast': 0.7, 'radius': 20, 'threshold': 10, 'edges_only': True, 'edge_radius': 3, 'scale': 0.7},
    "0.3": {'weight': 0.3, 'contrast': 0.9, 'radius': 30, 'threshold': 15, 'edges_only': True, 'edge_radius': 3, 'scale': 0.6},
    "0.7": {'weight': 0.7, 'contrast': 1.2, 'radius': 50, 'threshold': 25, 'edges_only': True, 'edge_radius': 5, 'scale': 0.4},
    "1.0": {'weight': 1.0, 'contrast': 1.5, 'radius': 60, 'threshold': 20, 'edges_only': True, 'edge_radius': 6, 'scale': 0.3},
    "1.2": {'weight': 1.2, 'contrast': 2.5, 'radius': 70, 'threshold': 30, 'edges_only': True, 'edge_radius': 7, 'scale': 0.2},
}

def parse_usm_param_key(best_combination):
    """
    从 Best_Combination 中提取 USM 参数值或判断是否为 no_usm。
    
    Args:
        best_combination (str): 类似 "final_v1_usm_0.3_enhance_False" 或 "final_v1_no_usm_enhance_False"。
        
    Returns:
        str: 如果是 USM 参数值，则返回提取的值（如 "0.3")。
             如果是 no_usm，则返回 "no_usm"。
             如果无法解析，返回 None。
    """
    if "no_usm" in best_combination:
        return "no_usm"
    if "usm_" in best_combination:
        parts = best_combination.split("_")
        for i, part in enumerate(parts):
            if part == "usm":
                return parts[i + 1]  # 提取 usm 值
    return None  

# 定义基础命令
base_command = [
    "python", "-u", "models/team06_diffbir/inference.py",
    "--task", "face",
    "--upscale", "1",
    "--version", "v2",
    "--precision", "fp32",
    "--degrad", "v0"
]

# 定义 prompt_versions
prompt_versions = {
    1: {
        "pos": (
            "sufficient detail, high-resolution, detailed face, natural skin texture, sharp focus, "
            "realistic lighting, clear facial features, defined jawline, natural expression, "
            "proportional face, beautiful face, well-lit, high clarity, high aesthetic quality, "
            "natural proportions, detailed eyes, realistic shadows, natural colors, high-quality details, "
            "normal facial proportions IMG_123.CR2, 85mm lens, f/1.8"
        ),
        "neg": (
            "blurry, pixelated, low-resolution, distorted features, excessive noise, "
            "disproportionate facial features, plastic-like skin, cartoonish, unrealistic expressions, "
            "misshapen eyes or nose, blur hair, artifacts eyes, artifacts teeth, "
            "blurry, unnatural colors, exaggerated shadows, artifacts, overexposed areas, asymmetrical face, "
            "unnatural lighting, harsh shadows, warped facial proportions, noisy background, out-of-focus, "
            "unnatural textures, unrealistic proportions, strange or distorted facial features, blur hair, "
            "artifacts eyes, artifacts teeth"
        ),
    },
    2: {
        "pos": (
            "sufficient detail, high-resolution, feature-rich portrait, lifelike skin texture, sharp and clear focus, "
            "delicate lighting, well-defined facial details, natural expression, symmetrical features, "
            "aesthetically pleasing face, properly lit, vivid clarity, realistic color tones, refined details, "
            "expressive eyes, gentle shadows, photographic quality, balanced proportions, pure natural look"
            "high aesthetic quality, IMG_123.CR2, 85mm lens, f/1.8"
        ),
        "neg": (
            "blurry, pixelated, low-resolution, flattened or exaggerated features, heavy noise, over-smoothing, "
            "cartoon-like, plastic or waxy skin, inconsistent facial proportions, messy hair, artifacts eyes, "
            "artifacts teeth, unnatural highlights, harsh or uneven lighting, blurry edges, distorted or "
            "mismatched colors, strange facial expressions, out-of-focus elements, over-sharpened edges, warped geometry"
        ),
    },
    3: {
        "pos": (
            "sufficient detail, high-resolution, finely captured face, realistic pores and skin texture, crisp depth, "
            "soft and diffused lighting, precise facial landmarks, relaxed and natural pose, elegant expression, "
            "aesthetically unified proportions, carefully balanced highlights, high clarity and detail, immersive realism, "
            "nuanced color grading, well-defined eyes and eyelashes, distinct shadows, photorealistic finishing"
            "high aesthetic quality, IMG_123.CR2, 85mm lens, f/1.8"
        ),
        "neg": (
            "blurry, pixelated, low-resolution, blocky or compressed texture, incorrect facial alignment, "
            "overly bright or saturated areas, jarring color banding, cartoonish or painted look, unnatural glare, "
            "unnatural facial stretching, overly harsh shadows, artifacts eyes, artifacts teeth, blur hair, "
            "mismatched or dislocated facial features, skewed perspectives, unrealistic skin color, heavy noise or grain"
        ),
    },
}

def run_inference(csv_file, usm_csv_file, input_dir, output_dir, device="cuda"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取主 CSV 文件
    csv_data = {}
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data[row['Group']] = row  # 用 Group 作为键，存储整行数据

    # 读取 USM 参数对应的 CSV 文件
    usm_csv_data = {}
    with open(usm_csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            usm_csv_data[row['Group']] = row['Best_Combination']  # 提取 Best_Combination 列

    for root, _, files in os.walk(input_dir):
        for file_name in files:
            try:
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                group = os.path.splitext(file_name)[0]

                if group == "Jean-Marc_de_La_Sabliere_0001_0":
                    group = "Jean-Marc_de_La_Sabliere_0001_00"  # 映射到 CSV 中的键

                input_file = f"{group}.png"
                if input_file in ["0042.png", "0046.png"]:
                    input_path = os.path.join(root, input_file)
                    output_path = os.path.join(output_dir, input_file)
                    shutil.copy(input_path, output_path)
                    print(f"Copied {input_file} from {input_path} to {output_path}")
                    continue  

                if group not in csv_data:
                    print(f"Skipping {file_name}: not found in CSV.")
                    continue

                row = csv_data[group]
                best_combination = row['Best_Combination']

                params = parse_best_combination(best_combination)

                prompt_version = int(params['prompt_version'])
                pos_prompt = prompt_versions[prompt_version]["pos"]
                neg_prompt = prompt_versions[prompt_version]["neg"]

                command = base_command.copy()
                command.extend([
                    "--cfg_scale", params['cfg_scale'],
                    "--strength", params['strength'],
                    "--steps", params['steps'],
                    "--captioner", params['captioner'],
                    "--sampler", "spaced",
                    "--pos_prompt", f"'{pos_prompt}'",
                    "--neg_prompt", f"'{neg_prompt}'",
                    "--output", os.path.join(output_dir, f"{group}.png"),
                    "--device", device  # 添加设备参数
                ])
                
                if 'swindegrad' in params:
                    command.extend(["--swindegrad", params['swindegrad']])
                    if params['swindegrad'] != "v0" and 'swindegrad_pt' in params:
                        command.extend(["--swindegrad_pt", params['swindegrad_pt']])

                input_path = os.path.join(root, file_name)
                command.extend(["--input", input_path])

                print("Running command:", " ".join(command))
                subprocess.run(command)

                # 添加新的逻辑：处理 USM
                if group not in usm_csv_data:
                    print(f"Skipping USM processing for {group}: not found in USM CSV.")
                    continue  # 如果不在 USM CSV 中，则跳过

                best_combination = usm_csv_data[group]
                usm_param_value = parse_usm_param_key(best_combination)  # 提取 USM 参数值或判断是否为 no_usm
                if usm_param_value != "no_usm":  # 如果不是 no_usm，则应用 USM
                    usm_param = usm_params.get(usm_param_value)
                    output_image_path = os.path.join(output_dir, f"{group}.png")
                    if os.path.exists(output_image_path):
                        img = cv2.imread(output_image_path)
                        img = apply_usm(img.astype(np.float32) / 255.0, usm_param)
                        img = (img * 255).astype(np.uint8)
                        cv2.imwrite(output_image_path, img)
                        print(f"Applied USM with params {usm_param_value} to {output_image_path}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


def parse_best_combination(best_combination):
    params = {}

    keys = ['prompt_version', 'cfg_scale', 'strength', 'steps', 'captioner', 'swindegrad', 'swindegrad_pt']

    for key in keys:
        key_with_underscore = f"{key}_" 
        start_index = best_combination.find(key_with_underscore)  

        if start_index != -1:  
            start_index += len(key_with_underscore)  
            end_index = best_combination.find('_', start_index) 
            if end_index == -1:  
                end_index = len(best_combination)
            params[key] = best_combination[start_index:end_index]  

    return params

def run(model_dir, input_path, output_path, device):
    csv_file = "models/team06_diffbir/merged_combinations_bond05.csv"  # 替换为实际 CSV 文件路径
    usm_csv_file = "models/team06_diffbir/results_best_combinations_usm.csv"
    # input_path = "/root/autodl-tmp/data/face/test/test"  # 替换为实际输入图片路径
    # output_path = "output/output1"  # 替换为实际输出文件夹路径
    device = "cuda"  
    
    run_inference(csv_file, usm_csv_file, input_path, output_path, device=device)
    
if __name__ == "__main__":
#     csv_file = "/root/autodl-tmp/code/DiffBIR/result_final/merged_combinations1.csv"  # 替换为实际 CSV 文件路径
#     input_dir = "/root/autodl-tmp/data/face/test/test"  # 替换为实际输入图片路径
#     output_dir = "output/output1"  # 替换为实际输出文件夹路径
#     device = "cuda"  # 或 "cpu" 作为设备参数
    run()
#     run_inference(csv_file, input_dir, output_dir, device=device)
    
    