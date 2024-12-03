# --------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------------------------------
#breakpoint()
# dataset settings
dataset_type = 'CityscapesDataset'
data_root = './data/cityscapes/'
IMG_MEAN = [v * 255 for v in [0.5, 0.5, 0.5]]
IMG_VAR = [v * 255 for v in [0.5, 0.5, 0.5]]
img_norm_cfg = dict(mean=IMG_MEAN , std=IMG_VAR , to_rgb=True) # original

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanopticAnnotations'),
    dict(type='LoadDepthAnnotations'),
    dict(type='Resize', img_scale=(1280, 760)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenPanopLabels', sigma=8, mode='train'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_center',
                               'center_weights', 'gt_offset', 'offset_weights',
                               'gt_instance_seg', 'gt_depth_map']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
        img_scale=(512,256),#(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='SynthiaDataset',
        data_root='./data/synthia/',
        img_dir='RGB',
        depth_dir='', #'Depth',
        ann_dir='panoptic-labels-crowdth-0-for-daformer/synthia_panoptic',
        pipeline=train_pipeline
       ),
    val=dict(
        type='CityscapesDataset',
        data_root='./data/cityscapes/',
        img_dir='leftImg8bit/val',
        test_mode='whole',
        depth_dir='', #'Depth', # not in use
        ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
        pipeline=test_pipeline),)