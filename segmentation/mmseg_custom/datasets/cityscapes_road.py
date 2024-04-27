# Copyright (c) Hyggge. All rights reserved.

from mmseg.datasets.builder import DATASETS
from mmseg.datasets import CityscapesDataset


@DATASETS.register_module()
class CityscapesRoadDataset(CityscapesDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('road', 'non-road')

    PALETTE = [[0, 255, 0], [0, 0, 0]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds_road.png',
                 **kwargs):
        super(CityscapesRoadDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

   
