_base_ = [
    './mask_rcnn_r50_fpn.py',
    #'../datasets/coco_instance_clip.py',
    # '../configs/_base_/schedules/schedule_1x.py',
    #'../default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
#breakpoint()
model = dict(
    type='DenseCLIP_MaskRCNN_VIT',
    pretrained='PATH_TO_FILL/pretrained/ViT-B-16.pt',
    context_length=5,
    text_head=False,
    text_dim=512,
    score_concat_index=2,
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=640,
        style='pytorch'),
    #decode_head=dict(
    #    type='FPN',
    #    num_classes=150,
    #    loss_decode=dict(
    #        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        type='FPNHead',
        num_classes=19,#150,
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        #in_channels=[768, 768, 768+150, 768],
        in_channels=[768, 768, 768+19, 768],
        out_channels=256,
        num_outs=4),

    )
