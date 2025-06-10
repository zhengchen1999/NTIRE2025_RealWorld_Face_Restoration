import torch
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torchvision.utils import save_image

import pyiqa
import os

from .resnet_model import ZSSR_RES
from  .model import ZSSR_Net

def batch_apply(module, x, y):
    print(x.shape,y.shape)
    batch_size = x.shape[0]
    outputs = [module(x[i:i+1], y[i:i+1]) for i in range(batch_size)]
    return torch.stack(outputs).mean()

class ZSSR_lightning(pl.LightningModule):
    def __init__(self, config, clipiqa, maniqa, musiq, niqe, qalign):
        super().__init__()
        self.model = ZSSR_RES(config.in_channels, config.channels, config.num_layer)
        self.optimizer = Adam(self.model.parameters(), config.lr)

        self.num_epoch = config.num_epoch
        self.lr = config.lr

        self.max_score, self.max_score_step, self.max_score_output = -1e10, 0, None

        # 其他
        self.save_hyperparameters(config)
        self.example_input_array = torch.randn(1, 3, 256, 256)

        # 评价指标
        self.clipiqa = clipiqa
        self.maniqa = maniqa
        self.musiq = musiq
        self.niqe = niqe
        self.qalign = qalign


    def forward(self, x):
        return self.model(x)
    
    def calculate_loss(self, x, y):
        loss1 = F.l1_loss(x, y)
        loss2 = 1-self.clipiqa(x, y)
        loss3 = 1-self.maniqa(x, y)
        return loss1 + 0.1*loss2 + 0.1*loss3
 
    def training_step(self, batch, batch_idx):
        HR, LR = batch
        HR, LR = HR[0], LR[0]

        r_HR = self.model(LR)
        loss = self.calculate_loss(r_HR, HR)
        self.log("loss", loss.item(), prog_bar=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return loss
    
    def calculate_score(self, x, y):
        clipiqa = self.clipiqa(x, y)
        maniqa = self.maniqa(x, y)
        musiq = self.musiq(x, y) if self.musiq is not None else 0
        niqe = self.niqe(x, y) if self.niqe is not None else 0
        qalign = self.qalign(x, y) if self.qalign is not None else 0

        return clipiqa + maniqa + musiq / 100 + max(0, (10 - niqe) / 10) + qalign / 5

    def validation_step(self, batch, batch_idx):
        LR_upscale, LR = batch
        r_HR = self.model(LR_upscale)

        score = self.calculate_score(r_HR, LR_upscale)

        # 记下得分最高的图片
        if score > self.max_score:
            self.max_score = score
            self.max_score_step = self.global_step
            self.max_score_output = r_HR

    def configure_optimizers(self):
        scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epoch,
            eta_min=self.lr / 1e3
        )
        return [self.optimizer], [scheduler]
