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


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


@BACKBONES.register_module()
class SAMBaselineLora(SAMViT):
    def __init__(
        self,
        pretrain_size=1024,
        pretrained=None,
        # SAM ViT parameter
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # LoRA parameter
        lora_r=4,
        lora_layers=list(range(0, 12)),
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
            frozen=True,
        )

        self.w_As = []
        self.w_Bs = []

        for layer_i, blk in enumerate(self.blocks):
            # If we only want few lora layer instead of all
            if layer_i not in lora_layers:
                continue

            w_qkv_linear = blk.attn.qkv
            qkv_dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(qkv_dim, lora_r, bias=False)
            w_b_linear_q = nn.Linear(lora_r, qkv_dim, bias=False)
            w_a_linear_v = nn.Linear(qkv_dim, lora_r, bias=False)
            w_b_linear_v = nn.Linear(lora_r, qkv_dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
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
        self._init_lora_weights()

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
    
    def _init_lora_weights(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

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
