# ---------------------------------------------------------------
# Copyright (c) 2023-2024 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

source_dataset_type = 'CityscapesDataset'
source_data_root = './data/cityscapes/'
target_dataset_type = 'CityscapesFoggyDataset'
target_data_root = './data/cityscapesFoggy/'

img_dir='leftImg8bit/train'
img_dir_val='leftImg8bit/val'

beta='0.02' # 0.005, 0.01, 0.02
target_img_suffix=f'_leftImg8bit_foggy_beta_{beta}.png',

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
num_classes = 19

# synthia_train_pipeline = [
#                             dict(type='LoadImageFromFile'),
#                             dict(type='LoadPanopticAnnotations'),
#                             dict(type='Resize', img_scale=(1280, 760)),
#                             dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#                             dict(type='RandomFlip', prob=0.5),
#                             dict(type='GenPanopLabelsForMaskFormer',
#                                  sigma=8,
#                                  mode='train',
#                                  num_classes=num_classes,
#                                  gen_instance_classids_from_zero=True,
#                                  ),
#                             dict(type='Normalize', **img_norm_cfg),
#                             dict(type='DefaultFormatBundleMmdet'),
#                             dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_panoptic_only_thing_classes', 'max_inst_per_class']),
#                         ]

cityscapes_train_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadPanopticAnnotations'),
                    dict(type='Resize', img_scale=(1024, 512)),
                    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                    dict(type='RandomFlip', prob=0.5),
                    #dict(type='PhotoMetricDistortion'),
                    dict(type='GenPanopLabelsForMaskFormer',
                         sigma=8,
                         mode='train',
                         num_classes=num_classes,
                         gen_instance_classids_from_zero=True,
                         ),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='DefaultFormatBundleMmdet'),
                    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_panoptic_only_thing_classes', 'max_inst_per_class']),
                ]


cityscapes_foggy_train_pipeline = [
                            dict(type='LoadImageFromFile'),
                            # dict(type='LoadPanopticAnnotations'),
                            dict(type='Resize', img_scale=(1024, 512)),
                            dict(type='RandomCrop', crop_size=crop_size),
                            dict(type='RandomFlip', prob=0.5),
                            dict(type='Normalize', **img_norm_cfg),
                            dict(type='DefaultFormatBundleMmdet'),
                            dict(type='Collect', keys=['img']),
                        ]

test_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(type='MultiScaleFlipAug',
                    # img_scale=(1024, 512), # for actual training/val (2048, 1024)
                    img_scale=(2048, 1024),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img']),
                                ]
                         )
                ]

data = dict(
            samples_per_gpu=2,
            workers_per_gpu=4,
            train=dict(
                        type='UDADataset',
                        source=dict(
                            type=source_dataset_type,
                            data_root=source_data_root,
                            img_dir=img_dir,
                            depth_dir='',  # not in use
                            ann_dir='gtFine_panoptic/cityscapes_panoptic_train_trainId',
                            pipeline=cityscapes_train_pipeline,
                        ),
                        target=dict(
                            type=target_dataset_type,
                            data_root=target_data_root,
                            img_dir=img_dir,
                            depth_dir='', # not in use
                            ann_dir='gtFine_panoptic/cityscapes_panoptic_train_trainId',
                            pipeline=cityscapes_foggy_train_pipeline,
                            img_suffix=target_img_suffix,
                        )
                    ),
            val=dict(
                type=target_dataset_type,
                data_root=target_data_root,
                img_dir=img_dir_val,
                depth_dir='', # not in use
                ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
                pipeline=test_pipeline,
                img_suffix=target_img_suffix,
                    ),
            test=dict(
                type=target_dataset_type,
                data_root=target_data_root,
                img_dir=img_dir_val,
                depth_dir='',  # not in use
                ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
                pipeline=test_pipeline,
                img_suffix=target_img_suffix,
                    )
            )