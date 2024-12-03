# optimizer
#optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0, paramwise_cfg=dict(
        custom_keys={
            'unet': dict(lr_mult=0.0, decay_mult=0),
            'text_adaptor': dict(lr_mult=0.0, decay_mult=0),
            'neck':dict(lr_mult=0.0, decay_mult=0),
            'encoder_vq': dict(lr_mult=0, decay_mult=0),
            'rpn_head': dict(lr_mult=0, decay_mult=0),
        }))
optimizer_config = dict()
