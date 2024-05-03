# dataset settings
dataset_type = 'BDD100kDrivableDataset'
data_root = 'data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (512, 512)

id_map={
    0:0,
    **{k:1 for k in [200, 204, 213, 209, 206, 207]},
    **{k:2 for k in [201, 203, 211, 208]},
    **{k:3 for k in [216, 217, 215]},
    **{k:4 for k in [218, 219]},
    **{k:5 for k in [210, 232]},
    **{k:6 for k in [214]},
    **{k:7 for k in [202, 220, 221, 222, 231, 224, 225, 226, 230, 228, 229, 233]},
    **{k:8 for k in [205]},
    **{k:9 for k in [212]},
    **{k:10 for k in [227]},
    **{k:11 for k in [223, 250]},
    249:0,
    255:255
}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadADE20kGT', aux_gt_dirs=['data/bdd100k/labels/ade20k/masks/train'], suffixs=['.png'], selected_labels = [("car", "truck", "bus", "van")], aux_gt_ids=[1]),
    dict(type='LoadCustomRoadMarkGT', aux_gt_dirs=['data/bdd100k/labels/road_marking_seg/train'], suffixs=['.png'], id_map=id_map, aux_gt_ids=[2]),
    dict(type='Resize', img_scale=(640, 384), keep_ratio=False),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='MultiHeadCollect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 384),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/100k/train',
        ann_dir='labels/drivable/masks/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/100k/val',
        ann_dir='labels/drivable/masks/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/100k/val',
        ann_dir='labels/drivable/masks/val',
        pipeline=test_pipeline))
