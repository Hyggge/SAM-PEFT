# Copyright (c) Hyggge. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/bdd100k_drivable_multi_head_custom_road_mark_generated.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
pretrained = 'pretrained/sam_vit_b_01ec64.pth'
model = dict(
    type='EncoderDecoderMultiHead',
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='SAMAdapterLora',
        # SAM-B Parameters
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # Adapter Parameters
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        # LoRA Parameters
        lora_r=8,
        lora_layers=list(range(0, 12)),
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=768,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        ),
        dict(
            type='UPerHead',
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
    ],
    test_cfg=dict(mode='whole')
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95))
lr_config = dict(_delete_=True, 
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))

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
    dict(type='LoadCustomRoadMarkGT', aux_gt_dirs=['data/bdd100k/labels/road_marking_seg/train'], suffixs=['.png'], id_map=id_map),
]
