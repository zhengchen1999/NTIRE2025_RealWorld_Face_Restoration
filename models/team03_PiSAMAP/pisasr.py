import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules
from peft.utils.other import transpose
from .models.autoencoder_kl import AutoencoderKL
from .models.unet_2d_condition import UNet2DConditionModel

sys.path.append(os.getcwd())
from models.team09_PiSAMAP.src.my_utils.vaehook import VAEHook

## nonlinear mappings
# LIVE 2 param
#liqe
A1 = 100.0
B1 = 0.0
C1 = 2.077636606942305
S1 = 1.101934158051146
#musiq
A2 = 100.0
B2 = 0.0
C2 = 43.539140163575430
S2 = 16.786801106500807
#liqe_oppo_ms3
A3 = 100.0
B3 = 0.0
C3 = 1.984470232985137
S3 = 0.939839104077352
#clipiqa
A4= 100.0
B4 = 0.0
C4 = 0.383287415616992
S4 = 0.223077718907985
def logistic_mapping(x, model):
    A = 0
    B = 0
    C = 0
    S = 0
    if model == 'liqe':
        A = A1
        B = B1
        C = C1
        S = S1
    elif model == 'musiq':
        A = A2
        B = B2
        C = C2
        S = S2
    elif model == 'liqe_oppo_ms3':
        A = A3
        B = B3
        C = C3
        S = S3
    elif model == 'clipiqa':
        A = A4
        B = B4
        C = C4
        S = S4

    z = (x - C) / S
    return (A - B) / (1 + torch.exp(-z)) + B



# import pyiqa
# nriqa = pyiqa.create_metric('musiq', as_loss=True)

class PiSASR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        # Load pretrained weights
        self._load_pretrained_weights(args.pretrained_path)

        # Initialize VAE tiling
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        # Prepare LoRA adapters
        if not args.default:
            self._prepare_lora_deltas(["default_encoder_sem", "default_decoder_sem", "default_others_sem"])
        set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        # Move models to device and precision
        self._move_models_to_device_and_dtype()

        # Set parameters
        self.timesteps1 = torch.tensor([1], device=self.device).long()
        self.lambda_pix = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem = torch.tensor([args.lambda_sem], device=self.device)

    def _get_dtype(self, precision):
        """Get the appropriate data type based on precision."""
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _move_models_to_device_and_dtype(self):
        """Move models to the correct device and precision."""
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights and initialize LoRA adapters."""
        sd = torch.load(pretrained_path)
        self._load_and_save_ckpt_from_state_dict(sd)

    def _prepare_lora_deltas(self, adapter_names):
        """Precompute and store LoRA deltas for the given adapters."""
        self.lora_deltas_sem = {}
        key_list = [key for key, _ in self.unet.named_modules() if "lora_" not in key]

        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.unet, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    for active_adapter in adapter_names:
                        if active_adapter in target.lora_A.keys():
                            base_layer = target.get_base_layer()
                            weight_A = target.lora_A[active_adapter].weight
                            weight_B = target.lora_B[active_adapter].weight

                            s = target.get_base_layer().weight.size()
                            if s[2:4] == (1, 1):  # Conv2D 1x1
                                output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * target.scaling[active_adapter]
                            elif len(s) == 2:  # Linear layer
                                output_tensor = transpose(weight_B @ weight_A, False) * target.scaling[active_adapter]
                            else:  # Conv2D 3x3
                                output_tensor = F.conv2d(
                                    weight_A.permute(1, 0, 2, 3),
                                    weight_B,
                                ).permute(1, 0, 2, 3) * target.scaling[active_adapter]

                            key = key + ".weight"
                            self.lora_deltas_sem[key] = output_tensor.data.to(dtype=self.weight_dtype, device=self.device)

    def _apply_lora_delta(self):
        """Merge LoRA deltas into UNet weights."""
        for name, param in self.unet.named_parameters():
            if name in self.lora_deltas_sem:
                param.data = self.lora_deltas_sem[name] + self.ori_unet_weight[name]
            else:
                param.data = self.ori_unet_weight[name]

    def _apply_ori_weight(self):
        """Restore original UNet weights."""
        for name, param in self.unet.named_parameters():
            param.data = self.ori_unet_weight[name]

    def _load_and_save_ckpt_from_state_dict(self, sd):
        """Load checkpoint and initialize LoRA adapters."""
        # Define LoRA configurations
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        # Add and load adapters
        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        for name, param in self.unet.named_parameters():
            if "pix" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        # Merge and save unet weights
        set_weights_and_activate_adapters(self.unet, ["default_encoder_pix", "default_decoder_pix", "default_others_pix"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()
        self.ori_unet_weight = {}
        for name, param in self.unet.named_parameters():
            self.ori_unet_weight[name] = param.clone()
            self.ori_unet_weight[name] = self.ori_unet_weight[name].data.to(self.weight_dtype).to("cuda")
        
        # Add semantic adapters
        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")
        
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])


    def set_eval(self):
        """Set models to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def count_parameters(self, model):
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters()) / 1e9

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        """Forward pass for inference."""
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # Tile and process latents if necessary
        model_pred = self._process_latents(encoded_control, prompt_embeds, default)

        # Decode output
        x_denoised = encoded_control - model_pred
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        return total_time, output_image

    # def forward_iter(self, default, c_t, prompt=None):
    #     """Forward pass for inference."""
    #     torch.cuda.synchronize()
    #     start_time = time.time()
    #
    #     c_t = c_t.to(dtype=self.weight_dtype)
    #     prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
    #     encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
    #     original_encoded_control = encoded_control.clone().detach()
    #     prev_b_arr = 0
    #     with torch.enable_grad():
    #         for m in range(15):
    #             encoded_control = encoded_control.clone().detach().requires_grad_(True)
    #             with autocast(self.device ):
    #                 # Tile and process latents if necessary
    #                 model_pred = self._process_latents(encoded_control, prompt_embeds, default)
    #
    #                 # Decode output
    #                 x_denoised = original_encoded_control - model_pred
    #                 output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
    #                 loss = nriqa((output_image+1)/2)
    #                 print(loss.item())
    #                 loss = loss.half()
    #                 loss.backward()
    #                 g = encoded_control.grad
    #                 g = g.clip(1e-3, 1e-3)
    #                 mom = 0.9
    #                 if m == 0:
    #                     b = g
    #                 else:
    #                     b = mom * prev_b_arr + g
    #                 g = b
    #                 prev_b_arr = b.clone()
    #                 encoded_control = encoded_control + 10 * g
    #
    #     torch.cuda.synchronize()
    #     total_time = time.time() - start_time
    #
    #     return total_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds, default):
        """Process latents with or without tiling."""
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            print("[Tiled Latent]: Input size is small, no tiling required.")
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)

        print(f"[Tiled Latent]: Input size {h}x{w}, tiling required.")
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        """Predict on the entire latent without tiling."""
        if default:
            return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

        model_pred_sem = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        self._apply_ori_weight()
        model_pred_pix = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        self._apply_lora_delta()

        model_pred_sem -= model_pred_pix
        return self.lambda_pix * model_pred_pix + self.lambda_sem * model_pred_sem

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap):
        """Predict on the latent with tiling."""
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0
        cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # input tile dimensions
                input_tile = encoded_control[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)
                    # predict the noise residual
                    if default:
                        print(f"[0:Default setting]")
                        model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                    else:
                        print(f"[1:Adjustable setting]")
                        model_out_sem = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                        self._apply_ori_weight()
                        model_out_pix = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                        self._apply_lora_delta()
                        model_out_sem = model_out_sem - model_out_pix
                        model_out = self.lambda_pix * model_out_pix + self.lambda_sem * model_out_sem
                    # model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
                    input_list = []
                noise_preds.append(model_out)

        # Stitch noise predictions for all tiles
        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        model_pred = noise_pred
        return model_pred



    

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generate a Gaussian mask for tile contributions."""
        from numpy import pi, exp, sqrt
        import numpy as np

        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        """Initialize VAE with tiled encoding/decoding."""
        encoder, decoder = self.vae.encoder, self.vae.decoder

        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward

        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
