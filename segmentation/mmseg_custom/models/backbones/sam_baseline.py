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
from .base.sam_vit import SAMViT
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class SAMBaseline(SAMViT):
    def __init__(
        self,
        pretrain_size=1024,
        pretrained=None,
        frozen=False,
        # SAM ViT parameter
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
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
            frozen=frozen
        )

        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)

        embed_dim = self.embed_dim
        self.norm1 = self.norm_layer(embed_dim)
        self.norm2 = self.norm_layer(embed_dim)
        self.norm3 = self.norm_layer(embed_dim)
        self.norm4 = self.norm_layer(embed_dim)

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        ])
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

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

    def forward(self, x):
        # Patch Embedding forward
        x = self.patch_embed(x) # (B, H, W, C)
        bs, H, W, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed, H, W)
        x = x + pos_embed # (B, H, W, C)

        # Interaction
        outs = list()
    
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            outs.append(x.permute(0, 3, 1, 2)) # (B, C, H, W)

        # Split & Reshape
        f1, f2, f3, f4 = outs[2], outs[5], outs[8], outs[11]
        f1 = self.norm1(f1.reshape(bs, dim, H * W).transpose(1, 2)).transpose(1, 2).reshape(bs, dim, H, W)
        f2 = self.norm2(f2.reshape(bs, dim, H * W).transpose(1, 2)).transpose(1, 2).reshape(bs, dim, H, W)
        f3 = self.norm3(f3.reshape(bs, dim, H * W).transpose(1, 2)).transpose(1, 2).reshape(bs, dim, H, W)
        f4 = self.norm4(f4.reshape(bs, dim, H * W).transpose(1, 2)).transpose(1, 2).reshape(bs, dim, H, W)
        
        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()

        return [f1, f2, f3, f4]
