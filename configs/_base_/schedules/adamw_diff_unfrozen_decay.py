# optimizer
#optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = dict(type='AdamW', lr=0.00004, weight_decay=5e-3, paramwise_cfg=dict(
        custom_keys={
            'unet': dict(lr_mult=0., decay_mult=0.0),
            'encoder_vq': dict(lr_mult=0, decay_mult=0)
        }))
optimizer_config = dict()
