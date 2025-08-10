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
        # print('weight',self.fc[0].grad)
        
        # for name, param in self.fc[0].named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} has gradient(fc): {param.grad}")
        
        
        # total_memory_in_bytes = sum(param.numel() * param.storage().element_size() for param in self.fc.parameters())
        # print(f"Total memory in bytes: {total_memory_in_bytes}")
        # print(f"Total memory in MB: {total_memory_in_bytes / (1024 * 1024):.4f}") 
            
        # print('weight',self.fc[0].weight)
        class_embeddings = class_probs @ self.embeddings
        conditioning_scene_embedding = self.embedding_adapter(class_embeddings, self.gamma) 
        
        # for name, param in self.embedding_adapter.fc[0].named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} has gradient(em): {param.grad}")
        
        
        # total_memory_in_bytes = sum(param.numel() * param.storage().element_size() for param in self.embedding_adapter.parameters())
        # print(f"Total memory in bytes: {total_memory_in_bytes}")
        # print(f"Total memory in MB: {total_memory_in_bytes / (1024 * 1024):.4f}") 
        # print('em_weight',self.embedding_adapter.fc[0].weight) 
        return conditioning_scene_embedding

class ControlLDMVAE(LatentDiffusion):
    def __init__(self, control_stage_config, cide_stage_config,control_key, only_mid_control, prior_model=None, *args, **kwargs):
        # print(control_stage_config)
        # print(cide_stage_config)
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
        # print(batch)
        # print(batch['jpg'][0].shape,batch['jpg'][0].device,batch['jpg'][0].max(),batch['jpg'][0].min())
        # print(batch['hint'][0].shape,batch['hint'][0].max(),batch['hint'][0].min())
        # print(batch['txt'])
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.imshow(batch['hint'][0].cpu())
        # plt.axis('off')  # 关闭坐标轴显示
        # plt.subplot(2, 1, 2)
        # plt.imshow(batch['jpg'][0].cpu())
        # plt.axis('off')  # 关闭坐标轴显示
        # plt.show()
        #x[2, 4, 96, 96],c[2, 77, 1024],control [2, 3, 768, 768]  control_z [2, 4, 96, 96]
        # print(len(batch['hint']),batch['hint'][0].shape)
        
        # import pdb;pdb.set_trace()
        # print("batch is trainable:", batch['rgb'].requires_grad)
        x, c,x0,xc = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # c = c.repeat(2,1,1)
        # print(x.shape,x.device,x.max(),x.min())
        # plt.figure()
        # print(xc)
        # plt.imshow(x.cpu())
        # plt.axis('off')  # 关闭坐标轴显示
        # plt.show()
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        # control_posterior = self.encode_first_stage_cond(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()

        # control_z = control_z.repeat(2,1,1,1)
        # prior_z = x[:,0:3,:,:]
        # prior_z = self.encode_first_stage(prior_z)
        # prior_z = self.get_first_stage_encoding(prior_z).detach()
        
        # using prior model to capture x0
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
        # import pdb;pdb.set_trace()
        # return x, dict(c_crossattn=[c], c_concat=[control_z], prior_out=[prior_z])
        # x:encoder(depth/normal/depth+normal) c:txt control_z:encoder(rgb)
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
        
        
        # xc = torch.cat(cond['txt'], 1)
        # label0 = xc.split()[0]
        # label0 = cond['txt'][0].split()[0]
        # label0 = cond['txt'][0].split()[0]
        # print(label0)
        # if label0 == 'indoor':
        #     label  =[0,1]
        # if label0 == 'outdoor':
        #     label  =[1,0]
        label =[]
        # print(cond['txt'][0])
        
        # for s in cond['txt'][0]:
        #     if 'indoor' in s:
        #         label0 = torch.tensor([1, 0])
        #     elif 'outdoor' in s:
        #         label0 = torch.tensor([0, 1])
        #     else:
        #         label0 = torch.tensor([1, 1])
        #     label.append(label0)

        # label = torch.stack(label).float().to(self.device)
        
        # print(label)
        # print(label.shape)
        # b=  x0.shape[0]
        # label = label.unsqueeze(0).expand(b, -1)
        # label = None
        c1 = self.cide_module(x1)
        # print(cond_txt.shape)
        # print(c1.shape)
        # for name, param in self.cide_module.named_parameters():
        #     print(f"Parameter {name} is trainable: {param.requires_grad}")
        # print("x_noisy  is trainable:", x_noisy.requires_grad)
        # print("cond_txt  is trainable:", cond_txt.requires_grad)
        # print("x0 is trainable:", x0.requires_grad)
        # print("c1 is trainable:", c1.requires_grad)
        
        cond_txt = cond_txt+c1
        # cond_txt = c1

        # print("cond_txt  is trainable:", cond_txt.requires_grad)
        # cond_txt = cond_txt
        # geo_class = torch.tensor([[0., 1.], [1, 0]], device=self.device, dtype=self.dtype)
        # geo_embedding = torch.cat([torch.sin(geo_class), torch.cos(geo_class)], dim=-1)
        # bs = x_noisy.shape[0]/geo_embedding.shape[0]
        # geo_embedding = geo_embedding.repeat_interleave(int(bs),dim=0)
        
        # import pdb;pdb.set_trace()
        if cond['c_concat'] is None:
            # eps = diffusion_model(x=x_noisy.repeat(2,1,1,1), timesteps=t.repeat(2), context=cond_txt.repeat(2,1,1), control=None, only_mid_control=self.only_mid_control)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            
        else:
            hint = torch.cat(cond['c_concat'], 1)
            # print(hint.shape,cond_txt.shape)  torch.Size([1, 4, 64, 64]) torch.Size([1, 77, 1024]) 
            control = self.control_model(x=x_noisy, hint=hint, timesteps=t, label= label,context=cond_txt)
            # print("control  is trainable:", control[0].requires_grad)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            # eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            # control = [c.repeat(2,1,1,1) * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, label= label,context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            # for name, param in diffusion_model.named_parameters():
            #     print(f"Parameter {name} is trainable: {param.requires_grad}")
            
           
            # print("eps  is trainable:", eps.requires_grad)
            #eps [2, 4, 96, 96]
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
    def compute_depth_metrics(self,pred, gt, mask=None, thresholds=[1.25, 1.25**2, 1.25**3],device='cuda'):
        # pred = pred.to(device)
        # gt = gt.to(device)

        # Mask 有效区域
        if mask is None:
            mask = (gt > 0)  # 假设无效深度值为0
        # print('1',pred.shape)
        # print('1-',type(pred))
        # print('2',mask.shape)
        # print('2-',type(mask))
        pred = pred[mask]
        gt = gt[mask]

        # 计算有效像素的个数
        num_valid_pixels = mask.sum().item()  # 获取有效像素的数量
        # print(num_valid_pixels)
        # 计算 RMSE
        rmse = torch.sqrt(F.mse_loss(pred, gt, reduction='mean'))

        # 计算 REL
        rel = torch.mean(torch.abs(pred - gt) / gt)

        # 计算 log10
        log10 = torch.mean(torch.abs(torch.log10(pred + 1e-6) - torch.log10(gt + 1e-6)))  # 为避免log(0)加小常数
        # 计算 δ1, δ2, δ3
        ratios = torch.max(pred / gt, gt / pred)
        zero = torch.zeros(pred.shape)
        one = torch.ones(pred.shape)
        
        bit_mat01 = torch.where(ratios.cpu() < thresholds[0], one, zero)
        # count_mat01 = torch.sum(bit_mat01, (-1, -2))
        count_mat01 = torch.sum(bit_mat01)
        delta1 = count_mat01/num_valid_pixels
        
        bit_mat02 = torch.where(ratios.cpu() < thresholds[1], one, zero)
        count_mat02 = torch.sum(bit_mat02)
        delta2 = count_mat02/num_valid_pixels
        
        bit_mat03 = torch.where(ratios.cpu() < thresholds[2], one, zero)
        count_mat03 = torch.sum(bit_mat03)
        delta3 = count_mat03/num_valid_pixels
        
        delta1 = torch.mean(delta1)
        delta2 = torch.mean(delta2)
        delta3 = torch.mean(delta3)

        # # 计算绝对误差和对数误差
        # abs_diff = np.abs(pred - gt)
        # rel_diff = abs_diff / gt
        # log10_diff = np.abs(np.log10(pred) - np.log10(gt))

        # # 计算 RMSE
        # rmse = np.sqrt(np.mean((pred - gt) ** 2))

        # # 计算 REL
        # rel = np.mean(rel_diff)

        # # 计算 log10
        # log10 = np.mean(log10_diff)

        # # 计算 δ1, δ2, δ3
        # ratios = np.maximum(pred / gt, gt / pred)
        # delta_results = [np.mean(ratios < t) for t in thresholds]
        # dict = {
        #     "RMSE": rmse.item() / num_valid_pixels,
        #     "REL": rel.item() / num_valid_pixels,
        #     "log10": log10.item() / num_valid_pixels,
        #     "delta1": delta1.item(),
        #     "delta2": delta2.item(),
        #     "delta3": delta3.item()
        # }
        dict = {
            "RMSE": rmse.item() ,
            "REL": rel.item() ,
            "log10": log10.item(),
            "delta1": delta1.item(),
            "delta2": delta2.item(),
            "delta3": delta3.item()
        }
        print(dict)
        # 返回指标
        # return {
        #     "RMSE": rmse.item() / num_valid_pixels,
        #     "REL": rel.item() / num_valid_pixels,
        #     "log10": log10.item() / num_valid_pixels,
        #     "delta1": delta1.item(),
        #     "delta2": delta2.item(),
        #     "delta3": delta3.item()
        # }
        return dict

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
        # image = image.unsqueeze(0)  
        
        # print('1',image.shape)
        # image = image.to(device)  
        # depth = depth.to(device) 
        # print('2',depth.shape) 
        
        xc = batch['txt']
        c = self.get_learned_conditioning(xc)
        # print(c.device)
        # c = LatentDiffusion.get_learned_conditioning(model,c)
        # c = c.to(device)
        # print(c.device)
        
        control = image
        # print(control.device) 
        # control = control.to(device) 
        # print(control.device) 
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()

        c_crossattn = [c]
        c_concat = [control_z]

        # c_cat = c_concat[0][:4]
        # c =  c_crossattn[0][:4]
        c_cat = c_concat[0]
        c =  c_crossattn[0]
        # import pdb;pdb.set_trace()
        samples, z_denoise_row = self.sample_log1(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                            batch_size=N, ddim=True,
                                                            ddim_steps=50, eta=0.0)
        
        x_samples = self.decode_first_stage_depth(samples)
        # batch['jpg'] = batch['jpg'].to('cuda:0')
        # batch['txt'] = batch['txt'].to('cuda:0')
        
        pred_depth = x_samples  # 深度图是生成的样本

        # print('3',pred_depth.shape) 
        pred_depth = pred_depth.permute(0,2,3,1)
        # 计算指标
        # metrics = self.compute_depth_metrics(pred_depth.squeeze(), depth.squeeze(),device=device)  # 去掉批次维度
        metrics = self.compute_depth_metrics(pred_depth.squeeze(), depth.squeeze())
        self.log_dict(metrics, prog_bar=True,
                       logger=True, on_step=True, on_epoch=True)

        
        return  metrics
        
    def validation_epoch_end(self, outputs):
        # 计算各个指标的平均值
        print("end")
        all_metrics = {
            "RMSE": [],
            "REL": [],
            "log10": [],
            "delta1": [],
            "delta2": [],
            "delta3": []
            }
        for metrics in outputs:
            for key, value in metrics.items():
                all_metrics[key].append(value)
        for key in all_metrics:
            all_metrics[key] = torch.mean(torch.tensor(all_metrics[key]))
        # avg_metrics = {key: np.mean(val) for key, val in all_metrics.items()}
        self.log_dict(all_metrics, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return all_metrics

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
        
        # log["depth_reconstruction"] = self.decode_first_stage_depth(z[0:bs])
        # log["normal_reconstruction"] = self.decode_first_stage(z[bs:])
        # log['reconstruction'] = torch.cat((log["depth_reconstruction"],log["normal_reconstruction"]),dim=0)
        
        # log["control"] = self.decode_first_stage(c_cat)[0:bs]

        _,_, img_size_h, img_size_w = log['reconstruction'].shape
        log["conditioning"] = log_txt_as_img((img_size_w, img_size_h), batch[self.cond_stage_key][:N], size=16)

        log_sequence=['control', 'conditioning']

        if self.prior is not None:
            log['prior'] = self.decode_first_stage(c_prior)

            log_sequence.append('prior')

        log_sequence.append('samples')
        # log_sequence.append('depth_samples')
        # log_sequence.append('normal_samples')

        if plot_diffusion_rows:
            # get diffusion row
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
            # get denoise row
            # samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], 'c_prior': [c_prior]},
            #                                          batch_size=N, ddim=use_ddim,
            #                                          ddim_steps=ddim_steps, eta=ddim_eta)

            # samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
            #                                          batch_size=N, ddim=use_ddim,
            #                                          ddim_steps=ddim_steps, eta=ddim_eta)
            # # x_samples = self.decode_first_stage(samples)
            # x_samples = self.decode_first_stage_depth(samples)
            # log["samples"] = x_samples
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],"rgb":[x0],
                                                           "txt":[txt]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            
            x_samples = self.decode_first_stage(samples)
            # depth_samples = (depth_samples-depth_samples.min())/(depth_samples.max()-depth_samples.min())
            # depth_samples = depth_samples*2.0 -1.0
            
            
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        # if unconditional_guidance_scale > 1.0:
        #     uc_cross = self.get_unconditional_conditioning(N)
        #     uc_cat = c_cat  # torch.zeros_like(c_cat)
        #     uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        #     # samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], 'c_prior': [c_prior]},
        #     #                                  batch_size=N, ddim=use_ddim,
        #     #                                  ddim_steps=ddim_steps, eta=ddim_eta,
        #     #                                  unconditional_guidance_scale=unconditional_guidance_scale,
        #     #                                  unconditional_conditioning=uc_full,
        #     #                                  )
        #     samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
        #                             batch_size=N, ddim=use_ddim,
        #                             ddim_steps=ddim_steps, eta=ddim_eta,
        #                             unconditional_guidance_scale=unconditional_guidance_scale,
        #                             unconditional_conditioning=uc_full,
        #                             )
        #     # x_samples_cfg = self.decode_first_stage(samples_cfg)
        #     x_samples_cfg = self.decode_first_stage_depth(samples_cfg)
        #     log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

            # log_sequence.append(f"samples_cfg_scale_{unconditional_guidance_scale:.2f}")

        log_sequence.append('reconstruction')
        # log_sequence.append('depth_reconstruction')
        # log_sequence.append('normal_reconstruction')
        
        
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
        # import pdb;pdb.set_trace()
        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
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
        # import pdb;pdb.set_trace()
        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        time_range = range(ddim_steps)  # 假设 time_range 是一个整数序列
        total_steps = ddim_steps  # 进度条的总步数
        # iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.cide_module.parameters())
        
        if not self.sd_locked:
            # params += list(self.model.diffusion_model.class_embed.parameters())
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    # def configure_optimizers(self):
    #     params, params_embed = [], []
    #     lr = self.learning_rate
    #     for name, param in self.control_model.named_parameters():
    #         if 'class_embed' in name:
    #             params_embed.append(param)
    #         else:
    #             params.append(param)

    #     for name, param in self.cide_module.named_parameters():
    #         params_embed.append(param)
        
    #     if not self.sd_locked:
    #         for name, param in self.model.diffusion_model.named_parameters():
    #             if 'class_embed' in name:
    #                 params_embed.append(param)

    #         params += list(self.model.diffusion_model.output_blocks.parameters())
    #         params += list(self.model.diffusion_model.out.parameters())
    #     opt = torch.optim.AdamW([
    #         {"params": params, "lr": self.learning_rate},
    #         {"params": params_embed, "lr": self.learning_rate * 10.0}
    #     ],)
    #     return opt
    
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
        # import pdb;pdb.set_trace()
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        
        # geo_emb =self.geo_embed(geo_embedding)
        emb = self.time_embed(t_emb)
        # print('emb',emb.shape)
        
        # label_emb = self.class_embed(label)
        
        # print(label)
        # print(label.shape)
        # print('label_emb',label_emb.shape)
        
        # emb =emb + label_emb
        
        # print('emb1',emb.shape)
        # for name, param in self.time_embed[0].named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} has gradient: {param.grad}")
        # print(self.time_embed[0].weight)
        # emb = emb + geo_emb
        outs = []

        h = hint.type(self.dtype)
        for module in self.input_blocks:
            # import pdb;pdb.set_trace()
            # print(h.shape,emb.shape,context.shape)
            h = module(h, emb, context)
            outs.append(h)
        h = self.middle_block(h, emb, context)
        outs.append(h)

        return outs


