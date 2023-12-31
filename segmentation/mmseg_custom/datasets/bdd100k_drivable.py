from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class BDD100kDrivableDataset(CustomDataset):
    """Mapillary dataset.
    """
    CLASSES = ('background', 'drivable')

    PALETTE = [[0, 0, 0], [0, 255, 0]]

    def __init__(self, **kwargs):
        super(BDD100kDrivableDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_merge.png',
            reduce_zero_label=False,
            **kwargs)