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
class SAMAdapterVPTSPMFuse(SAMViTVPT):
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
        # SPM fuse parameter
        spm_fuse_mode="concat"
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
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.spm_fuse_mode = spm_fuse_mode

        if self.spm_fuse_mode == "concat":
            self.fuse = nn.Linear(embed_dim * 2, embed_dim)
        elif self.spm_fuse_mode == "add":
            self.fuse = nn.Identity()
        else:
            raise ValueError("spm_fuse_mode must be 'concat' or 'add'")

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
        self.fuse.apply(self._init_weights)
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
        r1, r2, r3, r4 = self.spm(aux_gt.float().repeat(1, 3, 1, 1)) # feature of aux_gt
        t1, t2, t3, t4 = self.spm(x) # feature of target input
        bs, dim, H, W = r1.shape
        r1 = r1.view(bs, dim, -1).transpose(1, 2)
        t1 = t1.view(bs, dim, -1).transpose(1, 2)
        r = torch.cat([r1, r2, r3, r4], dim=1)
        t = torch.cat([t1, t2, t3, t4], dim=1)

        if self.spm_fuse_mode == "concat":
            c = self.fuse(torch.cat([r, t], dim=-1))
        elif self.spm_fuse_mode == "add":
            c = self.fuse(r + t)
        
        c1 = c[:, 0 : t1.size(1), :].transpose(1, 2).view(bs, dim, H, W)
        c2 = c[:, t1.size(1) : t1.size(1)+t2.size(1), :]
        c3 = c[:, t1.size(1)+t2.size(1) : t1.size(1)+t2.size(1)+t3.size(1), :]
        c4 = c[:, t1.size(1)+t2.size(1)+t3.size(1):, :]
        
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
        
        wrapped_blocks = nn.ModuleList()
        for i, blk in enumerate(self.blocks):
            wrapped_blocks.append(
                WrappedBlock(
                    blk, 
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[i])) 
                )
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
    def __init__(self, block, prompt_emb):
        super().__init__()
        self.block = block
        self.prompt_emb = prompt_emb
    
    def forward(self, x, H, W):
        """ block wrapper for dim adapting

        Args:
            x (torch.Tensor): input tensor, whose dim is (B, N, C)
        
        """
        bs, n, c = x.shape
        return self.block(x.view(bs, H, W, c), self.prompt_emb).view(bs, n, c)
