from torch import nn
from models.team09_PiSAMAP.helper_functions import *
import models.team09_PiSAMAP.memcnn as memcnn
from models.team09_PiSAMAP.LIQE_clean import LIQE_ms
import pyiqa
import torch
from torchvision import models, transforms
#from wavelet_color_fix import adaptive_instance_normalization, wavelet_reconstruction, wavelet_decomposition
from torch import autocast
import torch.nn.functional as F
from models.team09_PiSAMAP.mapping_clean import logistic_mapping
from tqdm import tqdm
torch.autograd.set_detect_anomaly(False)
torch.set_grad_enabled(False)
from models.team09_PiSAMAP.nlpd import NLPD
import os
from diffusers import DDIMScheduler
#from models.team99_PiSASR.ram import inference_ram as inference
import random
def Normalize(in_channels, num_groups=1):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class SteppingLayer(nn.Module):
    """
    This is a layer that performs DDIM stepping that will be wrapped
    by memcnn to be invertible
    """

    def __init__(self, unet,
                 controlnet,
                 embedding_uc,
                 embedding_c,
                 scheduler=None,
                 num_timesteps=50,
                 guidance_scale=7.5,
                 clip_cond_fn=None,
                 single_variable=False
                 ):
        super(SteppingLayer, self).__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.e_uc = embedding_uc
        self.e_c = embedding_c
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        if scheduler is None:
            self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                           beta_schedule="scaled_linear",
                                           num_train_timesteps=1000,
                                           clip_sample=False,
                                           set_alpha_to_one=False)
        else:
            self.scheduler = scheduler
        self.scheduler.set_timesteps(num_timesteps)
        #self.scheduler.set_timesteps(1, device="cuda")

        self.clip_cond_fn = clip_cond_fn

        self.single_variable = single_variable

    def forward(self, i, t, latent_pair, ram_encoder_hidden_states, image,
                reverse=False):
        """
        Run an EDICT step
        """
        for base_latent_i in range(2):
            # Need to alternate order compatibly forward and backward
            if reverse:
                orig_i = self.num_timesteps - (i + 1)
                offset = (orig_i + 1) % 2
                latent_i = (base_latent_i + offset) % 2
            else:
                offset = i % 2
                latent_i = (base_latent_i + offset) % 2

            # leapfrog steps/run baseline logic hardcoded here
            latent_j = ((latent_i + 1) % 2)

            latent_i = latent_i.long()
            latent_j = latent_j.long()

            if self.single_variable:
                # If it's the single variable baseline then just operate on one tensor
                latent_i = torch.zeros(1, dtype=torch.long).to(device)
                latent_j = torch.zeros(1, dtype=torch.long).to(device)

            # select latent model input
            if base_latent_i == 0:
                latent_model_input = latent_pair.index_select(0, latent_j)
            else:
                latent_model_input = first_output
            latent_base = latent_pair.index_select(0, latent_i)
        
            
            down_block_res_samples, mid_block_res_sample = [None]*10, None
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=self.e_uc,
                controlnet_cond=image,
                conditioning_scale=1,
                guess_mode=False,
                return_dict=False,
                image_encoder_hidden_states = ram_encoder_hidden_states,
            )
            
            noise_pred_uncond = self.unet(
                    latent_model_input,
                    t[0],
                    encoder_hidden_states=self.e_uc,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                    image_encoder_hidden_states = ram_encoder_hidden_states,
                )[0]

            noise_pred = noise_pred_uncond

            # incorporate classifier guidance if applicable
            if self.clip_cond_fn is not None:
                clip_grad = self.clip_cond_fn(latent_model_input, t.long(),
                                              scheduler=self.scheduler)
                alpha_prod_t, beta_prod_t = get_alpha_and_beta(t.long(), self.scheduler)
                fac = beta_prod_t ** 0.5
                noise_pred = noise_pred - fac * clip_grad

                # Going forward or backward?
            step_call = reverse_step if reverse else forward_step
            # Step
            new_latent = step_call(self.scheduler,
                                   noise_pred,
                                   t[0].long(),
                                   latent_base)
            new_latent = new_latent.to(latent_base.dtype)

            # format outputs using index order
            if self.single_variable:
                combined_outputs = torch.cat([new_latent, new_latent])
                break

            if base_latent_i == 0:  # first pass
                first_output = new_latent
            else:  # second pass
                second_output = new_latent
                if latent_i == 1:  # so normal order
                    combined_outputs = torch.cat([first_output, second_output])
                else:  # Offset so did in reverse
                    combined_outputs = torch.cat([second_output, first_output])

        return i.clone(), t.clone(), combined_outputs

    def inverse(self, i, t, latent_pair, ram_encoder_hidden_states, image):
        # Inverse method for memcnn
        output = self.forward(i, t, latent_pair, ram_encoder_hidden_states, image, reverse=True)
        return output


class MixingLayer(nn.Module):
    """
    This does the mixing layer of EDICT
    https://arxiv.org/abs/2211.12446
    Equations 12/13
    """

    def __init__(self, mix_weight=0.93):
        super(MixingLayer, self).__init__()
        self.p = mix_weight

    def forward(self, input_x):
        input_x0, input_x1 = input_x[:1], input_x[1:]
        x0 = self.p * input_x0 + (1 - self.p) * input_x1
        x1 = (1 - self.p) * x0 + self.p * input_x1
        return torch.cat([x0, x1])

    def inverse(self, input_x):
        input_x0, input_x1 = input_x.split(1)
        x1 = (input_x1 - (1 - self.p) * input_x0) / self.p
        x0 = (input_x0 - (1 - self.p) * x1) / self.p
        return torch.cat([x0, x1])


def quality_loss_fn(quality_model_str='liqe', device='cpu', grad_scale=0):
    model2 = None
    if quality_model_str == 'liqe':
        model = pyiqa.create_metric('liqe_mix', as_loss=True, num_patch=15)
        mono = 1
        fp32 = True
    elif quality_model_str == 'ms_liqe':
        model = LIQE_ms(ckpt='model_zoo/team09_PiSAMAP/MS_LIQE.pt', device=device, scale=[1,0.75])
        model.eval()
        mono = 1
        fp32 = True
    elif quality_model_str == 'musiq':
        model = pyiqa.create_metric('musiq', as_loss=True)
        model.eval()
        mono = 1
        fp32 = True
    elif quality_model_str == 'clipiqa':
        model = pyiqa.create_metric('clipiqa', as_loss=True)
        model.eval()
        mono = 1
        fp32 = True
    elif quality_model_str == 'hybrid':
        model = LIQE_ms(ckpt='model_zoo/team09_PiSAMAP/MS_LIQE.pt', device=device, scale=[1,0.75])
        model.eval()
        mono = 1
        model2 = pyiqa.create_metric('musiq', as_loss=True)
        model3 = pyiqa.create_metric('clipiqa', as_loss=True)
        fp32 = True
    else:
        raise NotImplementedError('Not now!')

    def loss_fn(im_pix, quality_model_str):
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
        if fp32:
            im_pix = im_pix.float()
        prediction = model(im_pix)
        print('quality prediction:{}'.format(prediction.item()))
        loss = mono * prediction
        if quality_model_str == 'hybrid':
            loss = logistic_mapping(loss, 'ms_liqe')
        else:
            loss = logistic_mapping(loss, quality_model_str)
        print('mapped prediction:{}'.format(loss.item()))
        
        if model2 is not None: #only support liqe_rect + musiq now
            loss2 = model2(im_pix)
            #loss2 = logistic_mapping(loss2, 'musiq')
            
            loss3 = model3(im_pix)
            loss3 = logistic_mapping(loss3, 'clipiqa')
            
            loss = (loss + loss2 + loss3)/3
            
            print('hybrid quality prediction:{} {} {}'.format(loss2.item(), loss3.item(), loss.item()))
        
        loss = loss.half()
        return loss * grad_scale

    return loss_fn


def gen(
        quality_model_str='hybrid',
        grad_scale=1,
        sd_guidance_scale=4,
        latent_seed=None,  # for starting point only
        seed=0,  # more general seed that both cuts and noise for traversal off of
        steps=50,  # 20
        mix_weight=0.93,  # .7
        single_variable=False,
        latent_traversal=True,
        num_traversal_steps=50,
        tied_latents=True,
        use_momentum=True,
        use_nesterov=False,
        renormalize_latents=True,
        optimize_first_edict_image_only=False,
        opt_t=None,
        clip_grad_val=1e-3,
        source_im=None,
        loss_fn=None,
        embedding_unconditional=None,
        embedding_conditional=None,
        quality_lambda=0
):
    # SGD logic
    if use_nesterov:
        use_momentum = True
    # Mem cnn arg
    keep_input = True
    # Latent setup
    if latent_seed is None:
        latent_seed = seed
    generator = torch.cuda.manual_seed(latent_seed)

    if source_im is None:  # novel generation, not editing
        latent = torch.randn((1, 4, 64, 64),
                             generator=generator,
                             device=device,
                             dtype=torch_dtype,
                             requires_grad=True)
        latent_pair = torch.cat([latent.clone(), latent.clone()])
    else:
        assert not renormalize_latents
    if renormalize_latents:  # if renormalize_latents then get original norm value
        orig_norm = latent.norm().item()

    # random generator boilerplate
    generator = torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    cond_fn = None

    fidelity_fn = NLPD(channels=3, k=5, test_y_channel=False, low_freq=True).to(device)


    # EDICT mixing layers
    if single_variable:
        mix = torch.nn.Identity()
    else:
        mix = MixingLayer(mix_weight)
        mix = memcnn.InvertibleModuleWrapper(mix, keep_input=keep_input,
                                             keep_input_inverse=keep_input,
                                             num_bwd_passes=1)

    if latent_traversal:
        if single_variable:
            raise NotImplementedError  # Doesn't work in memory, could try one-step traversal but would be bad

        # make diffusion steps with no model conditioning function (Still has text conditioning tho)
        s = SteppingLayer(unet,
                          controlnet,
                          embedding_unconditional,
                          embedding_conditional,
                          guidance_scale=sd_guidance_scale,
                          num_timesteps=steps,
                          clip_cond_fn=None,
                          scheduler=scheduler,
                          single_variable=single_variable)
        s = memcnn.InvertibleModuleWrapper(s, keep_input=keep_input,
                                           keep_input_inverse=keep_input,
                                           num_bwd_passes=1)
        timesteps = s._fn.scheduler.timesteps
        

        # SGD boiler plate
        # if use_momentum: prev_b_arr = [None, None]

        prev_b_arr = [None, None]

        if source_im is not None:  # image editng
            # assert opt_t is None  # assert we're optimizing at x_T
            lq = tensor_transforms(source_im).unsqueeze(0).to(device)
            lq = ram_transforms(lq)
            #res = inference(lq, ram_model)
            ram_encoder_hidden_states = ram_model.generate_image_embeds(lq)

            source_im = np.array(source_im) / 255.0 * 2.0 - 1.0
            source_im = torch.from_numpy(source_im[np.newaxis, ...].transpose(0, 3, 1, 2))
            if source_im.shape[1] > 3:
                source_im = source_im[:, :3] * source_im[:, 3:] + (1 - source_im[:, 3:])
            source_im = source_im.to(device).to(unet.dtype)
            
            control_im = source_im*0.5 + 0.5

            # # Peform reverse diffusion process
            with autocast(device):
                init_latent = vae.encode(source_im).latent_dist.sample(generator=generator) * 0.18215
                latent_pair = init_latent.repeat(2, 1, 1, 1)
                # Iterate through reversed tiemsteps using EDICT
                for i, t in tqdm(enumerate(timesteps.flip(0)), total=len(timesteps)):
                    i = torch.tensor([i],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    t = torch.tensor([t],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    latent_pair = mix.inverse(latent_pair)
                    i, t, latent_pair = s.inverse(i, t, latent_pair, ram_encoder_hidden_states, control_im)

        elif opt_t is not None:  # Partial optimization
            raise NotImplementedError('Not implemented!')

        early_flag = False
        """
        PERFORM GRADIENT DESCENT DIRECTLY ON LATENTS USING GRADIENT CALCUALTED THROUGH WHOLE CHAIN
        """
        # turn on gradient calculation
        with torch.enable_grad():  # important b/c don't have on by default in module
            for m in range(num_traversal_steps):  # This is # of optimization steps
                if early_flag:
                    print('Early stopping!')
                    break
                print(f"Optimization Step {m}")

                original_latent_pair = latent_pair.clone().detach()

                # Get orig_latent_pair of latent pair
                orig_latent_pair = latent_pair.clone().detach().requires_grad_(True)
                input_latent_pair = orig_latent_pair.clone()
                # input_latent_pair.retain_grad()
                # Full-chain generation using EDICT
                # optimizer = torch.optim.Adam([input_latent_pair], lr=1)
                with autocast(device):
                    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                        i = torch.tensor([i],
                                         dtype=torch_dtype,
                                         device=latent_pair.device)
                        t = torch.tensor([t],
                                         dtype=torch_dtype,
                                         device=latent_pair.device)
                        i, t, input_latent_pair = s(i, t, input_latent_pair, ram_encoder_hidden_states, control_im)
                        input_latent_pair = mix(input_latent_pair)

                    # Get the images that the latents yield
                    ims = [vae.decode(l.to(vae.dtype) / 0.18215).sample
                           for l in input_latent_pair.chunk(2)]

                # save images and compute loss
                losses = []
                for im_i, im in enumerate(ims):
                    # Get formatted images and save one of them
                    pil_im = prep_image_for_return(im.detach())
                    latent = input_latent_pair[im_i, ...]
                    source_latent = original_latent_pair[im_i, ...]
                    # If guiding then compute loss
                    if grad_scale != 0:
                        loss = loss_fn(im, quality_model_str)
                        score = loss.item()

    
                        loss = quality_lambda * loss - 20*fidelity_fn((0.5*im+0.5).clamp(0,1).float(),
                                                                        (0.5*source_im+0.5).clamp(0,1).float()).half()

                        loss = loss.half()

                        losses.append(loss)
                        if optimize_first_edict_image_only: break

                    # save
                    # if ((m % save_interval) == 0 or m == (num_traversal_steps - 1)) and im_i == 0:
                    if (score >= 90) & (im_i == 0):
                        early_flag = True
                    if ((m == (num_traversal_steps - 1)) | (early_flag)) and im_i == 0: #reaching max iter or early stop
                        return_im = pil_im

                sum_loss = sum(losses)
                # Backward pass
                sum_loss.backward()
                # Access latent gradient directly
                grad = 0.5 * orig_latent_pair.grad
                # Average gradients if tied_latents
                if tied_latents:
                    grad = grad.mean(dim=0, keepdim=True)
                    grad = grad.repeat(2, 1, 1, 1)

                new_latents = []
                # SGD step (S=stochastic from multicrop, can also just be GD)
                # Iterate through latents/grads
                for grad_idx, (g, l) in enumerate(zip(grad.chunk(2), orig_latent_pair.chunk(2))):

                    # Clip max magnitude
                    if clip_grad_val is not None:
                        g = g.clip(-clip_grad_val, clip_grad_val)

                    # SGD code
                    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
                    if use_momentum:
                        mom = 0.9
                        # LR is grad scale
                        # sticking with generic 0.9 momentum for now, no dampening
                        if m == 0:
                            b = g
                        else:
                            b = mom * prev_b_arr[grad_idx] + g
                        if use_nesterov:
                            g = g + mom * b
                        else:
                            g = b
                        prev_b_arr[grad_idx] = b.clone()

                    new_l = l + 2*g
                    new_latents.append(new_l.clone())
                if tied_latents:  # don't think is needed with other tied_latent logic but just being safe
                    combined_l = 0.5 * (new_latents[0] + new_latents[1])
                    latent_pair = combined_l.repeat(2, 1, 1, 1)
                else:
                    latent_pair = torch.cat(new_latents)

                if renormalize_latents:  # Renormalize latents
                    for norm_i in range(2):
                        latent_pair[norm_i] = latent_pair[norm_i] * orig_norm / latent_pair[norm_i].norm().item()
            # Once we've run all of our optimization steps we're done
            return return_im
    else:
        raise NotImplementedError('Not implemented!')
        
    return