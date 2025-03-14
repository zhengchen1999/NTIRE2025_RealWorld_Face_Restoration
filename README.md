# [NTIRE 2025 Challenge on Real-World Face Restoration](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## How to test the baseline model?

1. `git clone https://github.com/zhengchen1999/NTIRE2025_RealWorld_Face_Restoration.git`
2. Select the model you would like to test:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
    - We provide a baseline (team00): CodeFormer (default). Switch models (default is CodeFormer) through commenting the code in [test.py](./test.py#L19). 

## How to add your model to this baseline?

Edit the `else` to `elif` in [test.py](./test.py#L24), and then you can add your own model with model id. 

`model_func` should be a function, which accept 4 params. 
- `model_dir`: the pretrained model. Participants are expected to save their pretrained model in `./model_zoo/` with in a folder named `teamID_MODELNAME`. 
- `input_path`: a folder contains several images in PNG format. 
- `output_path`: a folder contains restored images in PNG format. Please follow the section Folder Structure. 
- `device`: computation device. 

## How to eval images using NR-IQA metrics and facial ID?

### Environments

```sh
conda create -n NTIRE-FR python=3.8
conda activate NTIRE-FR
pip install -r requirements.txt
```

### Metrics include:
1. **NIQE**, **CLIP-IQA**, **MANIQA**, **MUSIQ**  
   - Provided by [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)

2. **QALIGN**  
   - Provided by [Q-Align](https://github.com/Q-Future/Q-Align)

3. **FID** (using FFHQ as reference)  
   - Provided by [VQFR](https://github.com/TencentARC/VQFR)

### Folder Structure
```
input_LQ_dir
├── test
│   ├── CelebA
│   ├── CelebChild-Test
│   ├── ...
├── val
│   ├── CelebA
│   ├── CelebChild-Test
│   ├── ...
    
output_dir_test
├── CelebA
├── CelebChild-Test
├──...
output_dir_val
├── CelebA
├── CelebChild-Test
├──...
```

### Command to calculate metrics

```sh
python eval.py \
--mode "test" \
--output_folder "/path/to/your/output_dir_test" \
--lq_ref_folder "/path/to/input_LQ_dir" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
--use_qalign True 
```

The `eval.py` file accepts the following 6 parameters:
- `mode`: Choose whether to test images from the `test` set or the `val` set.
- `output_folder`: Path where the restored images will be saved. Subdirectories should be organized by dataset names.
- `lq_ref_folder`: Path to the LQ images provided as input to the model. This path should be the parent directory of the `test` and `val` sets.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.
- `use_qalign`: Whether to use QALIGN or not, which will consume an additional 15GB of GPU memory.



## License and Acknowledgement

This code repository is release under [MIT License](LICENSE). 

Several implementations are taken from: [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), [VQFR](https://github.com/TencentARC/VQFR), [Q-Align](https://github.com/Q-Future/Q-Align). 