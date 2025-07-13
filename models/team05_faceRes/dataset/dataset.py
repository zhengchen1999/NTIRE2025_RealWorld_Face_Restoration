import cv2, imageio
import numpy as np
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader, Dataset


class Pari_Image_dataset(Dataset):
    '''
    训练的时候使用的自定义数据集
    基本功能：
    1. 读取一张图像的路径为图像：格式为numpy
    2. 根据论文的细节将图像进行不同比例的缩放：设定6个缩放比例，计算相应的概率密度pdf
    3. 随机从不同分辨率的缩放结果中根据概率分布pdf来采样batch_size数量的图像
    4. 在采样的batch_size张图像上进行随机裁剪128x128，随机水平翻转，随机旋转90度等数据增强
    5. 进一步转化为tensor类型便于送入神经网络
    '''
    def __init__(self, image_path, sr_factor, patch_size, batch_size, num_scale):
        super().__init__()
        self.image = imageio.v2.imread(image_path)
        self.sr_factor = sr_factor
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_scale = num_scale

        # 计算最短边长度
        h, w, _ = self.image.shape
        short_side = min(h, w)
        # 计算最大缩放系数：输入图像较大时，最大缩放系数为5， 否则为最短边长度除以裁剪大小patch_size
        max_scale = min(5, short_side / patch_size)
        # 计算缩放系数列表
        scale_list = np.linspace(1/max_scale, 1.0, num_scale+1)[1:]
        # 计算不同缩放系数在训练时被选择的概率
        pdf = np.power(scale_list, 2)
        self.pdf = pdf / np.sum(pdf)
        # 计算缩放后图像的宽和高
        h_list = (h * scale_list).astype(np.uint16)
        w_list = (w * scale_list).astype(np.uint16)
        # 生成不同缩放系数的图像
        self.image_list = []
        for i in range(self.num_scale - 1):
            self.image_list.append(A.resize(self.image, height=h_list[i], width=w_list[i]))
        self.image_list.append(self.image)

        # 数据增强：随机裁剪，随机水平翻转，随机旋转90度
        self.transform = A.Compose([
            A.RandomCrop(height=patch_size, width=patch_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
        # 将numpy数据转化为tensor图像
        self.to_tensor = A.Compose([
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ])

    def __getitem__(self, idx):
        # 根据概率密度选择batch_size个不同分辨率的训练图像
        sample_index = np.random.choice(range(self.num_scale), self.batch_size, p=self.pdf)
        image_sample = [self.image_list[i] for i in sample_index]
        # 生成相应的训练图像对
        hr_sample = [self.transform(image=img)['image'] for img in image_sample]
        lr_sample = [A.resize(img, self.patch_size // self.sr_factor, self.patch_size // self.sr_factor) for img in hr_sample]
        lrup_sample = [A.resize(img, self.patch_size, self.patch_size, interpolation=cv2.INTER_CUBIC) for img in lr_sample]
        # 将图像对转化为tensor数据
        hr_tensor = [self.to_tensor(image=img)['image'].unsqueeze(0) for img in hr_sample]
        lrup_tensor = [self.to_tensor(image=img)['image'].unsqueeze(0) for img in lrup_sample]
        hr_batch = torch.cat(hr_tensor, dim=0)
        lrup_batch = torch.cat(lrup_tensor, dim=0)

        return hr_batch, lrup_batch

    def __len__(self):
        return 1


class Single_Image_dataset(Dataset):
    '''
    在验证的时候使用的自定义数据集：
    适配pytorch-lightning框架
    读取一张图像并转化为能够送入神经网络的张量，
    '''
    def __init__(self, image_path, sr_factor):
        super().__init__()
        self.image = imageio.v2.imread(image_path)
        h, w, c = self.image.shape
        self.sr_factor = sr_factor
        self.image_upscale = A.resize(self.image, height=h*self.sr_factor, width=w*self.sr_factor, interpolation=cv2.INTER_CUBIC)

        self.transform = A.Compose([
            A.ToFloat(max_value=255),
            ToTensorV2()
        ])
        self.image_tensor = self.transform(image=self.image)['image']
        self.image_upscale_tensor = self.transform(image=self.image_upscale)['image']

    def __getitem__(self, idx):
        return self.image_upscale_tensor, self.image_tensor

    def __len__(self):
        return 1




