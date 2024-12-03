# -----------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -----------------------------------------------------------------------------------

# model settings
num_instance_classes=8
#breakpoint()
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
maskrcnn_losses_weights = 1.0
model = dict(
    type='MaskRCNNPanopticDiffFrozenT0ScaleRoiBoth',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    use_neck_feat_for_decode_head=False,
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[320, 660, 1300, 1280],#[1280, 1280, 640, 320],# [64, 128, 320, 512],##,#[320, 790, 1430, 1280]
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg
                          ),
                        ),
        loss_decode=dict(type='CrossEntropyLossMmseg', use_sigmoid=False, loss_weight=0.0)
                    ),
    # below are maskrcnn stuff
    #the shape of x_b:  4 torch.Size([2, 64, 128, 128]) torch.Size([2, 128, 64, 64]) torch.Size([2, 320, 32, 32]) torch.Size([2, 512, 16, 16])
    #the shape of x_n:  5 torch.Size([2, 256, 128, 128]) torch.Size([2, 256, 64, 64]) torch.Size([2, 256, 32, 32]) torch.Size([2, 256, 16, 16]) torch.Size([2, 256, 8, 8])
    #Backbone
    #outs length:  4
    #outs in feat:  torch.Size([2, 320, 64, 64]) torch.Size([2, 790, 32, 32]) torch.Size([2, 1430, 16, 16]) torch.Size([2, 1280, 8, 8])
    #Diffusion
    #Without concat:
    #torch.Size([2, 1280, 8, 8]) torch.Size([2, 1280, 16, 16]) torch.Size([2, 640, 32, 32]) torch.Size([2, 320, 64, 64])
    #torch.Size([2, 320, 64, 64]) torch.Size([2, 790, 32, 32]) torch.Size([2, 1430, 16, 16]) torch.Size([2, 1280, 8, 8])
    #outs in feat:  torch.Size([2, 320, 64, 64]) torch.Size([2, 660, 32, 32]) torch.Size([2, 1300, 16, 16]) torch.Size([2, 1280, 8, 8])
    neck=dict(
        type='FPN',
        in_channels=[320, 660, 1300, 1280],#[1280, 1280, 640, 320],#[1280,1300, 660, 320],#[1280, 1300, 660, 320],#,#[1280,1300, 660, 320],#[320, 660, 1300, 1280],#[64, 128, 320, 512],#,#,[,#
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=maskrcnn_losses_weights),
        loss_bbox=dict(type='L1Loss', loss_weight=maskrcnn_losses_weights)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_instance_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=maskrcnn_losses_weights),#maskrcnn_losses_weights),
            loss_bbox=dict(type='L1Loss', loss_weight=maskrcnn_losses_weights)),#maskrcnn_losses_weights)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_instance_classes,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=maskrcnn_losses_weights))),#maskrcnn_losses_weights))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,#0.53,#0.7,#0.9, #0.7,
                neg_iou_thr=0.3,#0.38,#0.3,#0.1,
                min_pos_iou=0.3,#0.4,#0.3,#0.1,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,#0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        mode='whole',
        rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0
                ),
        rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100,
                    mask_thr_binary=0.5
                )
                )
        )
