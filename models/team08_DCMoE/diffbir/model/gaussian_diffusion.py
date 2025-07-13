from functools import partial
from typing import Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np

#AdaFace_loss
class AdaFaceFeature:
    """
    @Time    :   2023/10/10 22:52:55
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   AdaFace 人脸特征值预测
    """
    __instance = None
    
    def __new__(cls, *args, **kwargs):

        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self,file_name="config/config.yaml") -> None:
        """
        @Time    :   2023/10/10 22:54:19
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   初始化配置
        """
        self.config = Yaml.get_yaml_config(file_name)
        self.adaface_config = self.config['adaface']['zero']
        self.adaface_models = {self.adaface_config['model']: self.adaface_config['model_file'],}
        pass

    def load_pretrained_model(self):
        """
        @Time    :   2023/10/10 23:03:07
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   加载模型
        """
        
        # load model and pretrained statedict
        architecture = self.adaface_config['model']
        assert architecture in self.adaface_models.keys()
        model = net.build_model(architecture)
        statedict = torch.load(
            self.adaface_models[architecture], map_location=torch.device('cpu'))['state_dict']
        model_statedict = {key[6:]: val for key,
                           val in statedict.items() if key.startswith('model.')}
        model.load_state_dict(model_statedict)
        model.eval()
        self.model = model
        return self



    def to_input(self,pil_rgb_image):
        """
        @Time    :   2023/10/10 23:08:09
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   PIL RGB图像对象转换为PyTorch模型的输入张量
        """
        tensor = None
        try:
            np_img = np.array(pil_rgb_image)
            brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
            #tensor = torch.tensor([brg_img.transpose(2, 0,1)]).float()
            tensor = torch.tensor(np.array([brg_img.transpose(2, 0,1)])).float()
        except Exception :
            return tensor    
        return tensor




    def b64_get_represent(self,path):
        """
        @Time    :   2023/10/10 23:12:19
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   获取脸部特征向量
        """
        
        feature = None
        
        aligned_rgb_img =  utils.get_base64_to_Image(path).convert('RGB')
        bgr_tensor_input = self.to_input(aligned_rgb_img)
        if bgr_tensor_input is not None:
            feature, _ = self.model(bgr_tensor_input)
        else:
           print(f"无法提取脸部特征向量 🥷🥷🥷")     
        return feature
    

    def byte_get_represent(self,path):
        """
        @Time    :   2023/10/10 23:12:19
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   获取脸部特征向量
        """
        
        feature = None
        
        aligned_rgb_img =  utils.get_byte_to_Image(path).convert('RGB')
        bgr_tensor_input = self.to_input(aligned_rgb_img)
        if bgr_tensor_input is not None:
            feature, _ = self.model(bgr_tensor_input)
        else:
           print(f"无法提取脸部特征向量 🥷🥷🥷")     
        return feature
    

    def findCosineDistance(self,source_representation, test_representation):
        """
        @Time    :   2023/06/16 12:19:27
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   计算两个向量之间的余弦相似度得分
        """
        import torch.nn.functional as F
        return F.cosine_similarity(source_representation, test_representation)
    


#新增模块感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, device, layer_weights=None):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features[:35].eval()
        self.vgg = vgg.to(device).requires_grad_(False)
        
        # 默认特征层权重配置
        self.layer_weights = {
            'conv1_2': 1.0,
            'conv2_2': 0.75,
            'conv3_2': 0.5,
            'conv4_2': 0.3
        } if layer_weights is None else layer_weights
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    def preprocess(self, x):
        # 输入范围转换 [-1,1] → [0,1] → VGG归一化
        x = (x + 1) * 0.5  # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self.preprocess(pred)
        target = self.preprocess(target.detach())
        
        total_loss = 0.0
        features = []
        x = pred
        for name, module in self.vgg.named_children():
            x = module(x)
            features.append((name, x))
            if name in self.layer_weights:
                # 计算特征图L1损失
                target_feat = self.vgg(target)[:len(features)]
                loss = F.l1_loss(x, target_feat[-1]) 
                total_loss += self.layer_weights[name] * loss
        return total_loss



def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "sqrt":
        betas = (
            np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]
) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# Copy from: https://kkgithub.com/Max-We/sf-zero-signal-to-noise/blob/main/common_diffusion_noise_schedulers_are_flawed.ipynb
# Original paper: https://arxiv.org/abs/2305.08891
def enforce_zero_terminal_snr(betas: np.ndarray) -> np.ndarray:
    betas = torch.from_numpy(betas)
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas.numpy()


class Diffusion(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",
        zero_snr=False,
        # 新增参数
        perceptual_weight: float = 0.1,
        perceptual_layers: str = "conv3_2,conv4_2",  # 选择特征层

    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], "currently only supporting 'eps' and 'x0' and 'v'"
        self.parameterization = parameterization
        self.zero_snr = zero_snr
        self.loss_type = loss_type
        # 初始化感知损失
        self.perceptual_weight = perceptual_weight
        self.perceptual_layers = {
            k: 1.0 for k in perceptual_layers.split(',')
        }


        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        if zero_snr:
            betas = enforce_zero_terminal_snr(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        self.betas = betas
        self.register("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def setup_perceptual(self, device):
        self.perceptual_loss = PerceptualLoss(
            device, 
            layer_weights=self.perceptual_layers
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                mse_loss = torch.nn.functional.mse_loss(target, pred)
                # 新增感知损失计算
                perc_loss = 0
                if hasattr(self, 'perceptual_loss'):
                    perc_loss = self.perceptual_loss(target, pred)
            
                    perc_loss = self.perceptual_weight * perc_loss
                loss = mse_loss + perc_loss
            else:
                mse_loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
                # 新增感知损失计算
                perc_loss = 0
                if hasattr(self, 'perceptual_loss'):
                    perc_loss = self.perceptual_loss(target, pred)
            
                    perc_loss = self.perceptual_weight * perc_loss
                loss = mse_loss + perc_loss
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, model, x_start, t, cond):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean()
        return loss_simple
