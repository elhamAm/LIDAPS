# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

dataset_type = 'MapillaryDataset'
data_root = 'data/mapillary/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
num_classes = 19
cityscapes_train_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadPanopticAnnotations'),
                    dict(type='Resize', img_scale=(1024, 512)),
                    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                    dict(type='RandomFlip', prob=0.5),
                    dict(type='PhotoMetricDistortion'),
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
synthia_train_pipeline = [
                            dict(type='LoadImageFromFile'),
                            dict(type='LoadPanopticAnnotations'),
                            dict(type='Resize', img_scale=(1280, 760)),
                            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                            dict(type='RandomFlip', prob=0.5),
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

mapillary_train_pipeline = [
                            dict(type='LoadImageFromFile'),
                            dict(type='ResizeWithPad', img_scale=(1024, 768), img_pad_value=0, label_pad_value=0),
                            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0, dataset_name='mapillary'),
                            dict(type='RandomFlip', prob=0.5),
                            dict(type='Normalize', **img_norm_cfg),
                            dict(type='DefaultFormatBundleMmdet'),
                            dict(type='Collect', keys=['img']),
                        ]
test_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='MultiScaleFlipAug',
                                img_scale=(1024, 768),
                                flip=False,
                                transforms=[
                                            dict(type='ResizeWithPad', img_scale=(1024, 768), keep_ratio=True, img_pad_value=0, label_pad_value=0),
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
                            type='CityscapesDataset',
                            data_root='./data/cityscapes/',#'leftImg8bit/train',#'data/synthia/',
                            img_dir='leftImg8bit/train',
                            depth_dir='',  # not in use
                            ann_dir='gtFine_panoptic/cityscapes_panoptic_train_trainId',#'panoptic-labels-crowdth-0-for-daformer/synthia_panoptic',
                            pipeline=cityscapes_train_pipeline),#synthia_train_pipeline),
                        target=dict(
                            type=dataset_type,
                            data_root=data_root,
                            img_dir='train_imgs',#'training/images',#
                            depth_dir='', # not in use
                            ann_dir='train_panoptic_19cls',#'gtFine_panoptic/cityscapes_panoptic_train_trainId',#'train_panoptic_19cls',
                            pipeline=mapillary_train_pipeline)
                    ),
            val=dict(
                type=dataset_type,
                data_root=data_root,
                img_dir='val_imgs',#'testing/images',#
                depth_dir='',
                ann_dir='val_panoptic_19cls',
                pipeline=test_pipeline
                    ),
            test=dict(
                type=dataset_type,
                data_root=data_root,
                img_dir='val_imgs',#'testing/images',#
                depth_dir='',
                ann_dir='val_panoptic_19cls',
                pipeline=test_pipeline
                    )
            )