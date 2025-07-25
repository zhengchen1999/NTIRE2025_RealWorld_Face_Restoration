# :milky_way: Training Procedures
[English](train.md) **|** [简体中文](train_CN.md)
## Preparing Dataset

- Download training dataset: [FFHQ](https://kkgithub.com/NVlabs/ffhq-dataset)

---

## Training
```
For PyTorch versions >= 1.10, please replace `python -m torch.distributed.launch` in the commands below with `torchrun`.
```

### 👾 Stage I - VQGAN
- Training VQGAN:
  > python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=4321 basicsr/train.py -opt options/VQGAN_512_ds32_nearest_stage1.yml --launcher pytorch

- After VQGAN training, you can pre-calculate code sequence for the training dataset to speed up the later training stages:
  > python scripts/generate_latent_gt.py

- If you don't require training your own VQGAN, you can find pre-trained VQGAN (`vqgan_code1024.pth`) and the corresponding code sequence (`latent_gt_code1024.pth`) in the folder of Releases v0.1.0: https://kkgithub.com/sczhou/CodeFormer/releases/tag/v0.1.0

### 🚀 Stage II - CodeFormer (w=0)
- Training Code Sequence Prediction Module:
  > python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=4322 basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher pytorch

- Pre-trained CodeFormer of stage II (`codeformer_stage2.pth`) can be found in the folder of Releases v0.1.0: https://kkgithub.com/sczhou/CodeFormer/releases/tag/v0.1.0

### 🛸 Stage III - CodeFormer (w=1)
- Training Controllable Module:
  > python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=4323 basicsr/train.py -opt options/CodeFormer_stage3.yml --launcher pytorch

- Pre-trained CodeFormer (`codeformer.pth`) can be found in the folder of Releases v0.1.0: https://kkgithub.com/sczhou/CodeFormer/releases/tag/v0.1.0

---

:whale: The project was built using the framework [BasicSR](https://kkgithub.com/XPixelGroup/BasicSR). For detailed information on training, resuming, and other related topics, please refer to the documentation: https://kkgithub.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md
