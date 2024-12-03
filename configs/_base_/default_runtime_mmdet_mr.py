checkpoint_config = dict(interval=1)

#import wandb
#wandb.login(key='1cab7645befb1b548e079e7898cad3455fbe240d', relogin=True)
#breakpoint()
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    #    dict(type='MMDetWandbHook',
    #    init_kwargs={'project': 'MMDetection-tutorial','dir': 'PATH_TO_FILL/elham/wandb','config':{'pos':0.5}}, 
    #    interval=10,
    #    log_checkpoint_metadata=True,
    #    log_checkpoint=True,
    #    num_eval_images=10)
    ]
    )
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
