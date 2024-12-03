# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------------

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = './data/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#IMG_MEAN = [v * 255 for v in [0.5, 0.5, 0.5]]
#IMG_VAR = [v * 255 for v in [0.5, 0.5, 0.5]]
#img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)
crop_size = (512, 512)
num_classes = 19
#print('hiiiiiiiiiiiiiiiii')
synthia_train_pipeline = [
                            dict(type='LoadImageFromFile'),
                            dict(type='LoadPanopticAnnotations'),
                            dict(type='Resize', img_scale=(1280, 760)),
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
test_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(type='MultiScaleFlipAug',
                    img_scale=(1024,512),#(1024, 512),#(512,256),
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
    workers_per_gpu=2,
    train=dict(
        type='SynthiaDataset',
        data_root='./data/synthia/',
        img_dir='RGB',
        depth_dir='',
        ann_dir='panoptic-labels-crowdth-0-for-daformer/synthia_panoptic',
        pipeline=synthia_train_pipeline
       ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        depth_dir='', # not in use
        ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        depth_dir='', # not in use
        ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
        pipeline=test_pipeline)
)