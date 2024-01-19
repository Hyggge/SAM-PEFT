# Copyright (c) Hyggge. All rights reserved.
import mmcv
import os
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.formatting import to_tensor



@PIPELINES.register_module()
class LoadAuxilliaryGT(object):
    """Transfer gt_semantic_seg to binary mask and generate gt_labels."""
    def __init__(self, aux_gt_dirs, suffixs):
        self.aux_gt_dirs = aux_gt_dirs
        self.suffixs = suffixs
        self.file_client = mmcv.FileClient(backend='disk')

    def __call__(self, results):
        
        for i, (dir, suffix) in enumerate(zip(self.aux_gt_dirs, self.suffixs)):
            filename = os.path.join(dir, results['img_info']['filename'])
            filename = os.path.splitext(filename)[0] + suffix
            img_bytes = self.file_client.get(filename)
            aux_gt = mmcv.imfrombytes(
                img_bytes, flag='unchanged', backend='cv2').squeeze().astype(np.uint8)
            aux_gt = aux_gt[:, :, 0]
            aux_gt[aux_gt == 255] = 1
            # aux_gt_list.append(aux_gt)
            results[f'aux_gt_{i+1}'] = aux_gt
            results['seg_fields'].append(f'aux_gt_{i+1}')
        
        # print(f"{results.keys()=}")
        # print(f"{results['img_info'].keys()=}")
        # print(results)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(ignore_index={self.ignore_index})'

