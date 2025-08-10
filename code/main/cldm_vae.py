import einops
import torch
import torch as th
import torch.nn as nn
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm1_cide import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
# from tqlt.prior_model.registry_class import NORMAL_PRIOR
from sampler.registry_class import SAMPLER
from cldm.cldm import ControlNet
import pdb
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig


def pad_to_make_square(x):
    y = 255*((x+1)/2)
    y = torch.permute(y, (0,2,3,1))
    bs, _, h, w = x.shape
    if w>h:
        patch = torch.zeros(bs, w-h, w, 3).to(x.device)
        y = torch.cat([y, patch], axis=1)
    else:
        patch = torch.zeros(bs, h, h-w, 3).to(x.device)
        y = torch.cat([y, patch], axis=2)
    return y.to(torch.int)


class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=1024):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            # nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.bias)

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts   
    
class CIDE(nn.Module):
    def __init__(self, emb_dim, train_from_scratch):
        super().__init__()
        
        self.vit_processor = ViTImageProcessor.from_pretrained('/opt/data/private/yuancai/code/NormalDiffusion/models/vit-base-patch16-224')
        if train_from_scratch:
            vit_config = ViTConfig(num_labels=1000)
            self.vit_model = ViTForImageClassification(vit_config)
        else:
            self.vit_model = ViTForImageClassification.from_pretrained('/opt/data/private/yuancai/code/NormalDiffusion/models/vit-base-patch16-224')
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 400),
            # nn.BatchNorm1d(400),
            nn.GELU(),
            nn.Linear(400, 100)
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.bias)
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        
        self.embeddings = nn.Parameter(torch.randn(100, self.dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)
    
    def forward(self, x):
        y = pad_to_make_square(x)
        # use torch.no_grad() to prevent gradient flow through the ViT since it is kept frozen
        with torch.no_grad():
            inputs = self.vit_processor(images=y, return_tensors="pt").to(x.device)
            vit_outputs = self.vit_model(**inputs)
            vit_logits = vit_outputs.logits
            
        class_probs = self.fc(vit_logits)
        class_probs = self.m(class_probs)

        class_embeddings = class_probs @ self.embeddings
        conditioning_scene_embedding = self.embedding_adapter(class_embeddings, self.gamma) 
        return conditioning_scene_embedding

class ControlLDMVAE(LatentDiffusion):
    def __init__(self, control_stage_config, cide_stage_config,control_key, only_mid_control, prior_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.cide_module = instantiate_from_config(cide_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        # self.cide_module = CIDE(args, 1024, False)
        

        if prior_model is not None:
            self.prior = NORMAL_PRIOR.build(dict(type=prior_model.name))
        else:
            self.prior = None

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c,x0,xc = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)

        control_z = self.get_first_stage_encoding(control_posterior).detach()
        if self.prior is not None:
            # [-1,1] -> [0, 255]
            prior_hint = (batch['jpg']+1.) / 2 * 255.
            _, hint_h, hint_w, _ = prior_hint.shape
            size_info = [hint_w, hint_h, 0, 0]
            self.prior.to(self.device)
            # [-1, 1]
            prior_out = self.prior(prior_hint, size_info=size_info)['abs_vals']
            # prior embedding
            # prior_posterior = self.encode_first_stage(prior_out)
            prior_posterior = self.encode_first_stage_cond(control)
            prior_z = self.get_first_stage_encoding(prior_posterior).detach()
        return x, dict(c_crossattn=[c], c_concat=[control_z],rgb = [x0],txt=[xc])

    @torch.no_grad()
    def get_control(self, control):
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()


        return control_z


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        # print('main',cond.keys())
        diffusion_model = self.model.diffusion_model
        # print(cond)
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        
        # print(cond_txt.shape) [1,77,1024]
        x1 = torch.cat(cond['rgb'], 1)
        
        label =[]

        c1 = self.cide_module(x1)

        
        cond_txt = cond_txt+c1

        
        # import pdb;pdb.set_trace()
        if cond['c_concat'] is None:

            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            
        else:
            hint = torch.cat(cond['c_concat'], 1)

            control = self.control_model(x=x_noisy, hint=hint, timesteps=t, label= label,context=cond_txt)
            
            control = [c * scale for c, scale in zip(control, self.control_scales)]
           
            eps = diffusion_model(x=x_noisy, timesteps=t, label= label,context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)
    
    @torch.no_grad()
    def decode_first_stage_depth(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        out1 = self.first_stage_model.decode(z)
        out_mean = out1.mean(dim=1,keepdim=True)
        out = out_mean.expand(-1,3,-1,-1)
        return out
    
   
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        N = batch['hint'].size(0)
        # print(N) 
        # vkitti torch.Size([2, 375, 1242, 3])
        image = batch['hint']
        depth = batch['jpg']
        
        # print('1-',image.shape)
        if batch['hint'].size(2)/32 >33:
            image = F.pad(image.permute(0,3,1,2), (19, 19, 4, 5), mode='constant', value=0)
            depth = F.pad(depth.permute(0,3,1,2), (19, 19, 4, 5), mode='constant', value=0)
            # print('1--',image.shape)
            image = image.permute(0,2,3,1)
            depth = depth.permute(0,2,3,1)
        if batch['hint'].size(1)/32 == 15 :
            image = F.pad(image.permute(0,3,1,2), (0, 0, 16, 16), mode='constant', value=0)
            depth = F.pad(depth.permute(0,3,1,2), (0, 0, 16, 16), mode='constant', value=0)
            # print('1--',image.shape)
            image = image.permute(0,2,3,1)
            depth = depth.permute(0,2,3,1)
 
        xc = batch['txt']
        c = self.get_learned_conditioning(xc)
 
        
        control = image
 
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()

        c_crossattn = [c]
        c_concat = [control_z]


        c_cat = c_concat[0]
        c =  c_crossattn[0]
        # import pdb;pdb.set_trace()
        samples, z_denoise_row = self.sample_log1(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                            batch_size=N, ddim=True,
                                                            ddim_steps=50, eta=0.0)
        
        x_samples = self.decode_first_stage_depth(samples)

        
        pred_depth = x_samples  

  
        pred_depth = pred_depth.permute(0,2,3,1)

        metrics = self.compute_depth_metrics(pred_depth.squeeze(), depth.squeeze())
        self.log_dict(metrics, prog_bar=True,
                       logger=True, on_step=True, on_epoch=True)

        
        return  metrics
        
    
    # n=4 --- n=8
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None
        bs = batch["rgb"].shape[0]
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)  # bs valid for control.....
        
        # c_cat, c_prior, c = c["c_concat"][0][:N], c["prior_out"][0][:N], c["c_crossattn"][0][:N]
        c_cat, c,x0,txt= c["c_concat"][0][:N], c["c_crossattn"][0][:N],c["rgb"][0][:N],c["txt"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        # import pdb;pdb.set_trace()
        z = z[:N]
        log["reconstruction"] = self.decode_first_stage(z)
        # log["reconstruction"] = self.decode_first_stage_depth(z)
        log["control"] = self.decode_first_stage(c_cat)
        
        

        _,_, img_size_h, img_size_w = log['reconstruction'].shape
        log["conditioning"] = log_txt_as_img((img_size_w, img_size_h), batch[self.cond_stage_key][:N], size=16)

        log_sequence=['control', 'conditioning']

        # if self.prior is not None:
        #     log['prior'] = self.decode_first_stage(c_prior)

        #     log_sequence.append('prior')

        log_sequence.append('samples')


        if plot_diffusion_rows:
            
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
           
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],"rgb":[x0],
                                                           "txt":[txt]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            
            x_samples = self.decode_first_stage(samples)
            
            
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        
        log_sequence.append('reconstruction')
       
        
        log['visualized'] = torch.cat([log[key].detach().cpu() for key in log_sequence], dim = -2)

        for key in log_sequence:
            log.pop(key)

        return log


    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        ddim_sampler = SAMPLER.build(dict(type=self.sampler_type), model=self)

        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)

        if self.prior is not None:
            c_prior = cond.pop('c_prior')
            kwargs['x0'] = c_prior[0]

        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    

    @torch.no_grad()
    def sample_log1(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        ddim_sampler = SAMPLER.build(dict(type=self.sampler_type), model=self)

        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)

        if self.prior is not None:
            c_prior = cond.pop('c_prior')
            kwargs['x0'] = c_prior[0]
       
        time_range = range(ddim_steps)  # 假设 time_range 是一个整数序列
        total_steps = ddim_steps  # 进度条的总步数
       
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.cide_module.parameters())
        
        if not self.sd_locked:
           
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    
    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class ControlNetVAE(ControlNet):
    def forward(self, x, hint, timesteps,label, context,geo_embedding=None, **kwargs):
       
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
       
        emb = self.time_embed(t_emb)
        outs = []

        h = hint.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            outs.append(h)
        h = self.middle_block(h, emb, context)
        outs.append(h)

        return outs


