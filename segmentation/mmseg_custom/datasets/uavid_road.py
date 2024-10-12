from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class UAViDRoadDataset(CustomDataset):
    """Mapillary dataset.
    """
    CLASSES = ('background', 'drivable')

    PALETTE = [[0, 0, 0], [0, 255, 0]]

    def __init__(self, **kwargs):
        super(UAViDRoadDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)