# optimizer
#optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
#optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.01, paramwise_cfg=dict(
#        custom_keys={
#            'unet': dict(lr_mult=0.1, decay_mult=0),
#            #'text_adaptor': dict(lr_mult=0.1, decay_mult=0),
#            'encoder_vq': dict(lr_mult=0, decay_mult=0)
#        }))
#optimizer_config = dict()

optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001,
        paramwise_cfg=dict(custom_keys={'unet': dict(lr_mult=0.1),
                                        'encoder_vq': dict(lr_mult=0.1),
                                        'text_adaptor': dict(lr_mult=0.1),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
