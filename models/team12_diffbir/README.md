Run Commands

```python
# cd root path
cd NTIRE2025_RealWorld_Face_Restoration

# create environment
conda create -n diffbir python=3.10
conda activate diffbir
pip install -r requirements_team06.txt

# test(Your commands should not include valid_dir.)
CUDA_VISIBLE_DEVICES=0 python test.py --test_dir [path to test data dir] --save_dir [path to save dir] --model_id 1

```


- We provide our model (team06): DiffBir. We have added our models through commenting the code in [test.py](./test.py#L19)(model_id :1). Caution: You should download the pretrained model with the link in `model_zoo/team00_CodeFormer/team06_diffbir.txt` , and put the files in following structure: 
      ```shell
      model_zoo
      └── team06_diffbir
         ├── face_swinir_v1.ckpt
         ├── v2-1_512-ema-pruned.ckpt
         ├── v2.pth
         └── exp
            ├── train_stage1_degrad1
            ├── train_stage1_degrad2
            ├── train_stage1_degrad3
            ├── train_stage1_degrad4
            └── train_stage1_degrad5
      ```
  