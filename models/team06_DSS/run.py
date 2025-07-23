import subprocess
import os
import shutil

def main(model_dir, input_path, output_path, device):
#****************need provide*********************
    # # 获取当前工程的绝对路径（project 目录）
    base_dir = os.getcwd()

    # script1_path = os.path.join(base_dir, 'models/team06_DSS/DiffBIR', 'inference.py')
    # print(f"runing {script1_path} ...")    
    # # 以当前文件夹为工作目录执行 main.py
    cwd_diffbir = os.path.join(base_dir, 'models/team06_DSS/DiffBIR')
    # result1 = subprocess.run(["python", script1_path, '--device', device.type, '--model_dir', model_dir, '--input', input_path, '--output' , f"{cwd_diffbir}/{input_path.split('/')[-1]}"], cwd=cwd_diffbir)    
    # # 检查执行是否出错
    # if result1.returncode != 0:
    #     print(f"error: DiffBIR/inference.py exit!")
    # else:
    #     print("DiffBIR/inference.py done!")

    script2_path = os.path.join(base_dir, 'models/team06_DSS/StableSR/scripts', 'sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py')
    print(f"runing {script2_path} ...")    
    # 以当前文件夹为工作目录执行 main.py
    cwd_stablesr = os.path.join(base_dir, 'models/team06_DSS/StableSR')
    result2 = subprocess.run(["python", script2_path, '--device', device.type, '--model_dir', model_dir, '--init-img', f"{cwd_diffbir}/{input_path.split('/')[-1]}" ,'--outdir', f"{cwd_stablesr}/{input_path.split('/')[-1]}"], cwd=cwd_stablesr)    
    # 检查执行是否出错
    if result2.returncode != 0:
        print(f"error: StableSR/scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py exit!")
    else:
        print("StableSR/scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py done!")

    script3_path = os.path.join(base_dir, 'models/team06_DSS/SUPIR', 'test.py')
    print(f"runing {script3_path} ...")    
    # 以当前文件夹为工作目录执行 main.py
    cwd_supir = os.path.join(base_dir, 'models/team06_DSS/SUPIR')
    result3 = subprocess.run(["python", script3_path, '--save_dir', output_path, '--img_dir', f"{cwd_stablesr}/{input_path.split('/')[-1]}" , '--diffbir_out', f"{cwd_diffbir}/{input_path.split('/')[-1]}" ,'--oriimg_dir', input_path ,'--device', device.type, '--model_dir', model_dir], cwd=cwd_supir)    
    # 检查执行是否出错
    if result3.returncode != 0:
        print(f"error: SUPIR/test.py exit!")
    else:
        print("SUPIR/test.py done!")
