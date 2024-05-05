from mmseg.datasets.builder import DATASETS
from mmseg.datasets.cityscapes import CityscapesDataset


@DATASETS.register_module()
class BDD100kSemSegDataset(CityscapesDataset):
    """Mapillary dataset.
    """

    def __init__(self, **kwargs):
        super(BDD100kSemSegDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)