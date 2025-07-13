import os
import shutil
import subprocess
from basicsr.utils.misc import gpu_is_available, get_device

def partition_images(low_id_file, input_dir, dir1, dir2):
    # 创建目标文件夹
    dataset_name = os.path.basename(input_dir.rstrip('/'))
    target_dir1 = os.path.join(dir1, dataset_name)
    target_dir2 = os.path.join(dir2, dataset_name)
    os.makedirs(target_dir1, exist_ok=True)
    os.makedirs(target_dir2, exist_ok=True)

    # 读取low_ID.txt中的图片名称
    with open(low_id_file, 'r') as f:
        low_id_images = set(line.strip() for line in f)

    # 遍历input_dir中的所有图片
    for root, dirs, files in os.walk(input_dir):
        for img_name in files:
            img_path = os.path.join(root, img_name)
            if os.path.isfile(img_path):
                # 计算相对路径
                relative_path = os.path.relpath(root, input_dir)
                # 目标文件夹路径
                target_subdir1 = os.path.join(target_dir1, relative_path)
                target_subdir2 = os.path.join(target_dir2, relative_path)
                # 创建目标文件夹
                os.makedirs(target_subdir1, exist_ok=True)
                os.makedirs(target_subdir2, exist_ok=True)
                # 移动文件
                if img_name in low_id_images:
                    shutil.copy(img_path, os.path.join(target_subdir1, img_name))
                else:
                    shutil.copy(img_path, os.path.join(target_subdir2, img_name))
    return target_dir1, target_dir2
                    

def run_inference_codeformer(model_dir, input_dir, output_dir):
    subprocess.run(['python', './models/team16_DCMoE/inference_codeformer.py', '-i', input_dir, '-o', output_dir])

def run_inference_diffbir(model_dir, input_dir, output_dir):
    subprocess.run(['python', './models/team16_DCMoE/inference_diffbir.py', '--input', input_dir, '--output', output_dir])

def merge_results(output_dir1, output_dir2, final_output_dir):
    os.makedirs(final_output_dir, exist_ok=True)
    for root, dirs, files in os.walk(output_dir1):
        for file in files:
            src_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, output_dir1)
            dest_path = os.path.join(final_output_dir, relative_path, file)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)
    for root, dirs, files in os.walk(output_dir2):
        for file in files:
            src_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, output_dir2)
            dest_path = os.path.join(final_output_dir, relative_path, file)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)


def main(model_dir, input_path, output_path, device=None):

    device = get_device()
    print(f'Running on device: {device}')

    low_id_file = './models/team16_DCMoE/low_ID.txt'
    dir1 = './models/team16_DCMoE/data/dir1/test'
    dir2 = './models/team16_DCMoE/data/dir2/test'

    # output_dir1 = './data/output_dir1_test'
    # output_dir2 = './data/output_dir2_test'

    # 划分图片
    target_dir1, target_dir2 = partition_images(low_id_file, input_path, dir1, dir2)

    # 对dir1中的图片使用inference_codeformer.py处理
    run_inference_codeformer(model_dir, target_dir1, output_path)

    # 对dir2中的图片使用inference.py处理
    run_inference_diffbir(model_dir, target_dir2, output_path)

    # # 合并处理结果
    # merge_results(output_dir1, output_dir2, output_path)

if __name__ == "__main__":
    main()