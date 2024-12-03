optimizer = dict(type='AdamW', lr=0.00008, weight_decay=5e-3, paramwise_cfg=dict(
        custom_keys={
            'unet': dict(lr_mult=0.1, decay_mult=0),
            'encoder_vq': dict(lr_mult=0, decay_mult=0)
        }))
optimizer_config = dict()
