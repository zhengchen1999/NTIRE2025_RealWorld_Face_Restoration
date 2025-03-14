# [NTIRE 2025 Challenge on Real-World Face Restoration](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

------------------------------ *The repo is developing* ------------------------------

## How to test the baseline model?

1. `git clone https://github.com/zhengchen1999/NTIRE2025_RealWorld_Face_Restoration.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python eval.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
    - We provide a baseline (team00): CodeFormer (default). Switch models (default is CodeFormer) through commenting the code in [eval.py](./eval.py#L19). Three baselines are all test normally with `run.sh`.

## How to add your model to this baseline?

[TODO]

## How to test images using NR-IQA metrics and facial ID?

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
    ├── CelebA
    ├── CelebChild-Test
    ├──...
    
output_dir
├── test
│   ├── CelebA
│   ├── CelebChild-Test
│   ├──...
├── val
    ├── CelebA
    ├── CelebChild-Test
    ├──...
```

### Command to calculate metrics

```sh
python test_metrics.py \
--output_folder "/path/to/your/output_dir" \
--savelq_ref_folder "/path/to/input_LQ_dir" \
--gpu_ids 0 \
--use_qalign True 
```
The `--use_qalign` option is used to enable or disable QALIGN, which will consume 15GB of GPU memory.


## License and Acknowledgement

This code repository is release under [MIT License](LICENSE). 

Several implementations are taken from: [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), [VQFR](https://github.com/TencentARC/VQFR). 