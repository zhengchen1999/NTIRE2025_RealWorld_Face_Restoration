import pyiqa
import torch

torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torchvision.utils import save_image
from .model import ZSSR_lightning
from .dataset import Single_Image_dataset, Pari_Image_dataset
from .config import set_config
import os
from tqdm import tqdm


class ZSSRWrapper:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if config.accelerator == "gpu" else "cpu"

        # 训练用
        self.clipiqa = pyiqa.create_metric("clipiqa", device=self.device, as_loss=True)
        self.maniqa = None
        # self.maniqa = pyiqa.create_metric("maniqa", device=self.device,as_loss=True)

        # 评分用
        self.musiq, self.niqe, self.qalign = None, None, None
        # self.musiq = pyiqa.create_metric("musiq", device=self.device,as_loss=True)
        self.niqe = pyiqa.create_metric("niqe", device=self.device, as_loss=True)
        # self.qalign = pyiqa.create_metric("qalign", device=self.device,as_loss=True)

    def perform_single(self, image_path, output_image_path):
        model = ZSSR_lightning(self.config, self.clipiqa, self.maniqa, self.musiq, self.niqe, self.qalign)
        # 从单张图片生成数据集
        train_dataset = Pari_Image_dataset(
            image_path=image_path,
            sr_factor=self.config.sr_factor,
            patch_size=self.config.patch_size,
            batch_size=self.config.batch_size,
            num_scale=self.config.num_scale
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            num_workers=self.config.num_workers,
            shuffle=False,
            persistent_workers=True
        )
        val_dataset = Single_Image_dataset(
            image_path=image_path,
            sr_factor=self.config.sr_factor
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=self.config.num_workers,
            shuffle=False,
            persistent_workers=True
        )

        trainer = pl.Trainer(
            max_epochs=self.config.num_epoch,
            log_every_n_steps=10,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            num_sanity_val_steps=2,
            accelerator=self.config.accelerator,
            devices=1
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        result = model.max_score_output
        save_image(result, output_image_path)

    def perform_multiple(self, input_dir, output_dir):
        filelist = os.listdir(input_dir)
        filelist.sort()

        for image_name in tqdm(filelist):
            image_path = os.path.join(input_dir, image_name)
            output_image_path = os.path.join(output_dir, image_name)

            if os.path.exists(output_image_path):
                print(f"Skipping {image_name}")
                continue
            self.perform_single(image_path, output_image_path)

    def wrapper(self, model_dir, input_path, output_path, device, args=None):
        self.perform_multiple(input_path, output_path)



