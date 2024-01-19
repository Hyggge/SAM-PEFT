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
from .base.sam_vit_vpt import SAMViTVPT
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class SAMBaselineVPT(SAMViTVPT):
    def __init__(
        self,
        pretrain_size=1024,
        pretrained=None,
        # SAM ViT parameter
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # VPT parameter
        prompt_dropout=0.,
        prompt_token_num=50,
        prompt_project=-1
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

        # virtual prompt
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(self.patch_size), 1) + prompt_dim))  # noqa   
        self.prompt_embeddings = nn.Parameter(torch.zeros(encoder_depth, prompt_token_num, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings, -val, val)

        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)

        # self.norm1 = nn.SyncBatchNorm(embed_dim)
        # self.norm2 = nn.SyncBatchNorm(embed_dim)
        # self.norm3 = nn.SyncBatchNorm(embed_dim)
        # self.norm4 = nn.SyncBatchNorm(embed_dim)


    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H, W).permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x):
        # Patch Embedding forward
        x = self.patch_embed(x) # (B, H, W, C)
        bs, H, W, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed, H, W)
        x = x + pos_embed # (B, H, W, C)

        # Interaction
        outs = list()
    
        for i, blk in enumerate(self.blocks):
            prompt_input = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[i]))
            x = blk(x, prompt_input)
            outs.append(x.permute(0, 3, 1, 2)) # (B, C, H, W)

        
        # Split & Reshape
        x1, x2, x3, x4 = outs[2], outs[5], outs[8], outs[11]
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False) # (img_h // 4, img_w // 4)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False) # (img_h // 8, img_w // 8)
        x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False) # ((img_h // 32, img_w // 32))

        # Final Norm
        # f1 = self.norm1(x1)
        # f2 = self.norm2(x2)
        # f3 = self.norm3(x3)
        # f4 = self.norm4(x4)

        # return [f1, f2, f3, f4]
        return [x1, x2, x3, x4]
