import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
from functools import reduce
from torch.nn.modules.utils import _pair
from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from functools import partial
from .base.sam_vit_vpt_attn import SAMViTVPTAttn
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

_logger = logging.getLogger(__name__)



class PriorPromptMaskAttn(nn.Module):
    def __init__(self, prompt_dim=768, prior_dim=768, inner_dim=768, head_num=8, bias=True, use_proj=[False, False, False, False]):
        super().__init__()
        self.inner_dim = inner_dim
        self.head_num = head_num
        self.to_q = nn.Linear(prompt_dim, inner_dim, bias=bias) if use_proj[0] else nn.Identity()
        self.to_k = nn.Linear(prior_dim, inner_dim, bias=bias) if use_proj[1] else nn.Identity()
        self.to_v = nn.Linear(prior_dim, inner_dim, bias=bias) if use_proj[2] else nn.Identity()
        self.to_out = nn.Linear(inner_dim, prompt_dim) if use_proj[3] else nn.Identity()
    
    def forward(self, prompt, prior, mask=None):
        n_q, _ = prompt.shape
        B, n_k, _ = prior.shape
        prompt = prompt.expand(B, -1, -1)

        # qkv with shape(B, nHead, N, C)
        k = self.to_k(prior).reshape(B, n_k, self.head_num, self.inner_dim // self.head_num).permute(0, 2, 1, 3)
        v = self.to_v(prior).reshape(B, n_k, self.head_num, self.inner_dim // self.head_num).permute(0, 2, 1, 3)
        q = self.to_q(prompt).reshape(B, n_q, self.head_num, self.inner_dim // self.head_num).permute(0, 2, 1, 3)
        mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.head_num, n_q, -1)
        scale = self.inner_dim ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)  # attn with shape(B, nHead, N_prompt, N_prior)
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, n_q, -1)
        x = self.to_out(x)

        return x
   

@BACKBONES.register_module()
class SAMAdapterVPTMaskAttn(SAMViTVPTAttn):
    def __init__(
        self,
        pretrain_size=1024,
        conv_inplane=64, 
        n_points=4,
        deform_num_heads=6, 
        init_values=0., 
        interaction_indexes=None, 
        with_cffn=True,
        cffn_ratio=0.25, 
        deform_ratio=1.0, 
        add_vit_feature=True, 
        pretrained=None,
        use_extra_extractor=True, 
        with_cp=False,
        drop_path_rate=0.,
        # SAM ViT parameter
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # VPT parameter
        prompt_dropout=0.,
        prompt_token_num=50,
        prompt_project=-1,
        ppa_prior_dim=768,
        ppa_inner_dim=768,
        ppa_head_num=8,
        ppa_use_proj=[False, False, False, False],
        ppa_prompt_norm=False
    ) :
        super().__init__(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=1024, # pretrain size, not input size
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=256, # no use
            pretrained=pretrained, 
        )

        embed_dim = self.embed_dim

        # prompt dropout setting
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        
        # prompt project setting
        if prompt_project > -1:
            prompt_dim = prompt_project
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
            
        else:
            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()

        self.ppa_prompt_norm = nn.LayerNorm(prompt_dim) if ppa_prompt_norm else nn.Identity()        

        self.ppa_list = nn.ModuleList()
        for _ in range(encoder_depth):
            self.ppa_list.append(
                PriorPromptMaskAttn(
                    prompt_dim=prompt_dim, 
                    prior_dim=ppa_prior_dim, 
                    inner_dim=ppa_inner_dim, 
                    head_num=ppa_head_num,
                    use_proj=ppa_use_proj
                )
            )

        # virtual prompt
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(self.patch_size), 1) + prompt_dim))  # noqa   
        self.prompt_embeddings = nn.Parameter(torch.zeros(encoder_depth, prompt_token_num, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings, -val, val)

        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H, W).permute(0, 2, 3, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x, aux_gt):
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.patch_embed(x) # (B, H, W, C)
        bs, H, W, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed, H, W)
        x = x + pos_embed # (B, H, W, C)
        x = x.view(bs, -1, dim)

        # Interaction
        outs = list()
        
        prior = c1 + self._get_pos_embed(self.pos_embed, c1.shape[-2], c1.shape[-1]).permute(0, 3, 1, 2)
        prior = prior.view(bs, dim, -1).transpose(1, 2)
        mask = F.interpolate(aux_gt.float(), size=(aux_gt.shape[-2] // 4, aux_gt.shape[-1] // 4), mode='nearest').long()
        mask = (mask != 0).view(bs, 1, -1).transpose(1, 2).squeeze(-1)

        wrapped_blocks = nn.ModuleList()
        for i, blk in enumerate(self.blocks):
            prompt_emb = self.ppa_prompt_norm(self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[i])))
            wrapped_blocks.append(
                WrappedBlock(blk, prompt_emb, prior, mask, self.ppa_list[i])
            )


        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, wrapped_blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W, )
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
        
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


class WrappedBlock(nn.Module):
    def __init__(self, block, prompt_emb, prior, mask, ppa):
        super().__init__()
        self.block = block
        self.prompt_emb = prompt_emb
        self.prior = prior
        self.mask = mask
        self.ppa = ppa
    
    def forward(self, x, H, W):
        """ block wrapper for dim adapting

        Args:
            x (torch.Tensor): input tensor, whose dim is (B, N, C)
        
        """
        bs, n, c = x.shape

        # new_prompt_emb with shape (B, N_Prompt, C)
        new_prompt_emb = self.ppa(self.prompt_emb, self.prior, self.mask)

        new_prompt_emb = self.prompt_emb.expand(bs, -1, -1) + new_prompt_emb

        return self.block(x.view(bs, H, W, c), new_prompt_emb).view(bs, n, c)
