# learning policy
#lr_config = dict(policy='poly', warmup='linear',  warmup_iters=1500, warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
#lr_config = dict(policy='poly', warmup='linear',  warmup_iters=1500, warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])