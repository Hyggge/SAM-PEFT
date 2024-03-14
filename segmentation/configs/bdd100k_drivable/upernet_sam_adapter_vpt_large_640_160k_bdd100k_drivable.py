# Copyright (c) Hyggge. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/bdd100k_drivable.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
pretrained = 'pretrained/sam_vit_l_0b3195.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='SAMAdapterVPT',
        # SAM-L Parameters
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        # Adapter Parameters
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        drop_path_rate=0.3,
        # VPT Paramters
        prompt_dropout=0.1,
        prompt_token_num=50,
        prompt_project=-1
        ),
    decode_head=dict(num_classes=150, in_channels=[768, 768, 768, 768]),
    auxiliary_head=dict(num_classes=150, in_channels=768),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
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
