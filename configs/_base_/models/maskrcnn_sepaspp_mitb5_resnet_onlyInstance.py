_base_ = [
    './mask_rcnn_r50_fpn_onlyInstance.py',
    #'../datasets/coco_instance_clip.py',
    # '../configs/_base_/schedules/schedule_1x.py',
    #'../default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DenseCLIP_MaskRCNN',
    pretrained='PATH_TO_FILL/pretrained/RN50.pt',
    context_length=5,
    seg_loss=True,
    clip_head=False,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=1344,#512,
        style='pytorch'),
    #decode_head=dict(
    #    type='FPN',
    #    num_classes=150,
    #    loss_decode=dict(
    #        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
    type='DAFormerHead',
    in_channels=[256, 512, 1024, 2067],#[256,256,256,256],#[64, 128, 320, 512], #[64, 128, 320, 512],#[320, 790, 1430, 1280]
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
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=1024,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    neck=dict(
        type='FPN',
        #num_classes=150,
        in_channels=[256, 512, 1024, 2067],#[256, 512, 1024, 2048 + 80],
        out_channels=256,
        num_outs=5),

    )
