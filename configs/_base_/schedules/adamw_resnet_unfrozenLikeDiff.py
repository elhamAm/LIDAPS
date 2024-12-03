# optimizer
#optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
#print('just adam')
#optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
#optimizer_config = dict()


optimizer = dict(type='AdamW', lr=0.00008, weight_decay=5e-3, paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'text_encoder': dict(lr_mult=0, decay_mult=0),
            'norm': dict(lr_mult=0, decay_mult=0)
        }))
optimizer_config = dict()