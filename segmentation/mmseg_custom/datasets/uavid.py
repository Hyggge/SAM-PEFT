from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class UAViDDataset(CustomDataset):
    """Mapillary dataset.
    """
    CLASSES = ('Clutter', 'Building', 'Road', 'Static_Car', 'Tree', 'Vegetation', 'Human', 'Moving_Car')
    PALETTE = [[0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192], [0, 128, 0], [128, 128, 0], [64, 64, 0], [64, 0, 128]]

    def __init__(self, **kwargs):
        super(UAViDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)