# optimizer
#optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = dict(type='AdamW', lr=0.00008, weight_decay=5e-3, paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=0),
            'text_encoder': dict(lr_mult=0, decay_mult=0),
            'norm': dict(decay_mult=0.)}))
        #}))
optimizer_config = dict()
#optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
