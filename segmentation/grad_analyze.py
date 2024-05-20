# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
from PIL import Image
import numpy as np

import mmcv
import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('out_dir', help='output result dir')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # assert cfg.model.type == 'EncoderDecoderMultiHeadV2', 'Only support EncoderDecoderMultiHeadV2 model'
    # cfg.model.type = "EncoderDecoderMultiHeadV2Infer"
    # assert cfg.model.decode_head.type == 'UPerHeadV2', 'Only support UPerHeadV2 decode head'
    # cfg.model.decode_head.type = "UPerHeadV2Infer"
    # build out dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # build dataset
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=2,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    # build the model
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # load ckpt
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    # Set an attribute for using the model in the following codes.
    model = MMDataParallel(model, device_ids=[0])
    # train
    model.train()
    model = revert_sync_batchnorm(model)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    cnt = 0
    for batch_indices, data in zip(loader_indices, data_loader):
        if cnt >= 10:
            break
        losses = model(**data)
        losses['decode.loss_ce'].backward()

        batch_size = batch_indices[1] - batch_indices[0] + 1
        for _ in range(batch_size):
            prog_bar.update()
        cnt += batch_size

    torch.save(model.module.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()


