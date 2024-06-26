# Copyright (c) OpenMMLab. All rights reserved.
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .uper_head_v2 import UPerHeadV2
from .uper_head_v2_infer import UPerHeadV2Infer

__all__ = [
    'MaskFormerHead',
    'Mask2FormerHead',
    'UPerHeadV2',
    'UPerHeadV2Infer'
]
