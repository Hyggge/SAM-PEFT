# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for name, param in model.named_parameters():
        print(name, ':', param.size())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    vit_params = sum(p.numel() for name, p in model.named_parameters() if ('pos_embed' in name or 'patch_embed' in name)) + \
                    sum(p.numel() for p in model.backbone.blocks.parameters())
    adapter_params = sum(p.numel() for p in model.backbone.spm.parameters()) + \
                    sum(p.numel() for p in model.backbone.interactions.parameters()) + \
                    sum(p.numel() for name, p in model.named_parameters() if ('level_embed' in name or 'backbone.up' in name or 'backbone.norm' in name) )
    decode_head_params = sum(p.numel() for p in model.decode_head.parameters())
    auxiliary_head_params = sum(p.numel() for p in model.auxiliary_head.parameters())

    print('Total:', total_params, ' Trainable:', trainable_params)
    print('Backbone:', backbone_params)
    print('Vit:', vit_params)
    print('Adapter:', adapter_params)
    print('Decode Head:', decode_head_params)
    print('Auxiliary Head:', auxiliary_head_params)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>') 


    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
        
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(args.img, result,
                            palette=get_palette(args.palette),
                            show=False, opacity=args.opacity)
    mmcv.mkdir_or_exist(args.out)
    out_path = osp.join(args.out, osp.basename(args.img))
    cv2.imwrite(out_path, img)
    print(f"Result is save at {out_path}")

if __name__ == '__main__':
    main()
