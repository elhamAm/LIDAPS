# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

dataset_type = 'CityscapesDataset'
data_root = './data/cityscapes'
IMG_MEAN = [v * 255 for v in [0.5, 0.5, 0.5]]
IMG_VAR = [v * 255 for v in [0.5, 0.5, 0.5]]
img_norm_cfg = dict(mean=IMG_MEAN , std=IMG_VAR, to_rgb=True)
#IMG_MEAN = [v * 255 for v in [0.5, 0.5, 0.5]]
#IMG_VAR = [v * 255 for v in [0.5, 0.5, 0.5]]
#img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)

crop_size = (512, 512)
num_classes = 19
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

cityscapes_train_pipeline = [
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
                    img_scale=(1024, 512),#(512, 256),#(512, 512),#
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        #dict(type='Crop', crop_size=(512,256)),##added by Elham
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
                            type='SynthiaDataset',
                            data_root='./data/synthia/',
                            img_dir='RGB',
                            depth_dir='',  # not in use
                            ann_dir='panoptic-labels-crowdth-0-for-daformer/synthia_panoptic',
                            pipeline=synthia_train_pipeline),
                        target=dict(
                            type=dataset_type,
                            data_root=data_root,
                            img_dir='leftImg8bit/train',
                            depth_dir='', # not in use
                            ann_dir='gtFine_panoptic/cityscapes_panoptic_train_trainId',
                            pipeline=cityscapes_train_pipeline,
                        )
                    ),
            val=dict(
                type=dataset_type,
                data_root=data_root,
                img_dir='leftImg8bit/val',
                depth_dir='', # not in use
                ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
                pipeline=test_pipeline
                    ),
            test=dict(
                type=dataset_type,
                data_root=data_root,
                img_dir='leftImg8bit/val',
                depth_dir='', # not in use
                ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
                pipeline=test_pipeline
                    )
            )