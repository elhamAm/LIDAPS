# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------------


import itertools


def get_model_base(architecture, backbone, semantic_decoder):
    return f'_base_/models/{architecture}_{semantic_decoder}_{backbone}.py'

def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    #cfg['data']['train']['rare_class_sampling'] = dict(
    #    min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id, machine_name):

    def config_from_vars():
        #breakpoint()
        cfg = {
            'debug': debug,
            '_base_': ['_base_/default_runtime_mmdet_mr.py'],
            'n_gpus': n_gpus,
            'gpu_mtotal': gpu_mtotal,
            'n_cpus': n_cpus,
            'mem_per_cpu': mem_per_cpu,
            'machine': machine,
            'exp_sub': exp_sub,
            'exp_root': exp_root,
            'total_train_time': total_train_time,
        }

        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        #breakpoint()
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone, semantic_decoder)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }

        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None

        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        #breakpoint()
        
        if activate_panoptic:
            if uda == 'target-only':
                cfg['_base_'].append(f'_base_/datasets/uda_{target}_bottomup_panoptic.py')
            elif uda == 'source-only':
                cfg['_base_'].append(f'_base_/datasets/uda_{source}_to_{target}_bottomup_panoptic_source_only.py')
            else:
                cfg['_base_'].append(f'_base_/datasets/uda_{source}_to_{target}_bottomup_panoptic.py')
                cfg['_base_'].append(f'_base_/uda/{uda}.py')

        if 'dacs' in uda and plcrop:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
            cfg['uda']['debug_img_interval'] = debug_img_interval
        if 'dacs' in uda and not plcrop:
            cfg.setdefault('uda', {})

        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})

        # Setup the ann_dir for validation
        cfg['data'].setdefault('val', {})
        cfg['data']['val']['ann_dir'] = ann_dir
        cfg['data']['val']['data_root'] = data_root


        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)
        #breakpoint()
        # Setup optimizer and schedule
        if 'dacs' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']

        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(by_epoch=False, interval=checkpoint_interval, max_keep_ckpts=1)
        # Set the log_interval
        cfg['log_config'] = dict(interval=log_interval)

        cfg['evaluation'] = dict(
            interval=eval_interval,
            metric=eval_metric_list,
            eval_type=eval_type,
            dataset_name=target,
            gt_dir=gt_dir_instance,
            debug=debug,
            num_samples_debug=num_samples_debug,
            gt_dir_panop=gt_dir_panoptic,
            post_proccess_params=post_proccess_params,
            evalScale=evalScale,
            visuals_pan_eval=visuals_pan_eval,
        )

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
        if 'dacs' in uda and plcrop:
            uda_mod += '_cpl'
        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture_mod}_' \
                      f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        #breakpoint()
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn')
        #breakpoint()
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    debug = False
    machine = machine_name
    iters = 320000
    interval = iters
    interval_debug = 3
    uda = 'dacs_a999_fdthings_bottomup'
    data_root = 'PATH_TO_FILL/cityscapes' #'data/cityscapes'
    # ----------------------------------------
    # --- Set the debug time configs ---
    # ----------------------------------------
    n_gpus = 2 if debug else 1
    batch_size = 1 if debug else 2  # samples_per_gpu
    workers_per_gpu = 0 if debug else 1  # if 'dacs' in uda else 2
    eval_interval = interval_debug if debug else interval
    checkpoint_interval = 2000#interval_debug if debug else interval
    ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
    log_interval = 1 if debug else 50
    debug_img_interval = 1 if debug else 5000
    # ----------------------------------------
    seed = 0
    center_threshold = 0.1
    evalScale = None
    architecture = 'bottomup'
    backbone = 'mitb5'
    semantic_decoder = 'sepaspp'
    source, target = 'synthia', 'cityscapes'
    datasets = [(source, target)]
    eval_type = 'panop_deeplab'
    gt_dir_instance = 'PATH_TO_FILL/data/cityscapes/gtFine/val' #'data/cityscapes/gtFine/val'
    gt_dir_panoptic = 'PATH_TO_FILL/data/cityscapes/gtFine_panoptic' #'data/cityscapes/gtFine_panoptic'
    num_samples_debug = 12
    post_proccess_params = dict(
        num_classes=19,
        ignore_label=255,
        train_id_to_eval_id=[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0],
        mapillary_dataloading_style='DADA',
        label_divisor=1000,
        cityscapes_thing_list = [11, 12, 13, 14, 15, 16, 17, 18],
        center_threshold=center_threshold,
        nms_kernel=7,
        top_k_instance=200,
    )
    opt = 'adamw'
    schedule = 'poly10warm'
    lr = 0.00006
    pmult = True
    rcs_T = 0.01
    plcrop = True
    activate_panoptic = True
    visuals_pan_eval = False
    eval_metric_list = ['mIoU', 'mPQ', 'mAP']
    exp_root = "PATH_TO_FILL/edaps_experiments"#
    exp_sub = f'exp-{id:05d}'

    # Euler Stuff
    n_cpus = 16
    mem_per_cpu = 16000
    gpu_mtotal = 23000
    total_train_time = '24:00:00'

    cfgs = []
    # --------------------------------------------------------------------------------
    # M-Dec-BU (Table 5)
    # --------------------------------------------------------------------------------
    if id == 100:
        seeds = [42]
        for seed in seeds:
            #breakpoint()
            uda='source-only'
            cfg = config_from_vars()
            cfg['activate_panoptic'] = activate_panoptic
            #breakpoint()
            #8000 iteration
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00100/work_dirs/local-exp00100/230923_1706_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_32645/iter_8000.pth'
            #40000 iterations
            cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00100/work_dirs/local-exp00100/230924_0022_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_502c7/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic.py', '_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #

    #source only diffusion backbone
    if id == 105:
        seeds = [42]
        for seed in seeds:
            #breakpoint()
            import os
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            os.environ['HF_CACHE'] = 'PATH_TO_FILL/cache'
            print(os.getenv('TRANSFORMERS_CACHE'))
            #breakpoint()
            opt = 'adamw_diff_bottomup'
            uda='source-only'
            cfg = config_from_vars()
            cfg['activate_panoptic'] = activate_panoptic
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only.py', '_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only_diff.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/bottomup_sepaspp_mitb5.py', '_base_/models/bottomup_sepaspp_mitb5_diff.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00105/work_dirs/local-exp00105/230919_2358_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_6ade3'
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #source only diffusion backbone
    if id == 107:
        seeds = [42]
        for seed in seeds:
            #breakpoint()
            import os
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            os.environ['HF_CACHE'] = 'PATH_TO_FILL/cache'
            print(os.getenv('TRANSFORMERS_CACHE'))
            #breakpoint()
            opt = 'adamw_diff_bottomup_frozen'
            uda='source-only'
            cfg = config_from_vars()
            cfg['activate_panoptic'] = activate_panoptic
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only.py', '_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only_diff.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/bottomup_sepaspp_mitb5.py', '_base_/models/bottomup_sepaspp_mitb5_diff_frozen.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00105/work_dirs/local-exp00105/230919_2358_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_6ade3'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00107/work_dirs/local-exp00107/231021_0105_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_544ff/iter_116000_.pth'
            #8000 iterations
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00107/work_dirs/local-exp00107/230921_1846_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_a5bed/iter_8000.pth'
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #source only diffusion backbone
    if id == 108:
        seeds = [42]
        for seed in seeds:
            #breakpoint()
            import os
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            os.environ['HF_CACHE'] = 'PATH_TO_FILL/cache'
            print(os.getenv('TRANSFORMERS_CACHE'))
            #breakpoint()
            opt = 'adamw_diff_bottomup_frozenEnc'
            uda='source-only'
            cfg = config_from_vars()
            cfg['activate_panoptic'] = activate_panoptic
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only.py', '_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only_diff.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/bottomup_sepaspp_mitb5.py', '_base_/models/bottomup_sepaspp_mitb5_diff_frozenEnc.py') for s in cfg['_base_']]
            #8000 iterations
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00108/work_dirs/local-exp00108/230922_1514_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_232ee/iter_8000.pth'
            #40000 iterations
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00108/work_dirs/local-exp00108/230923_1723_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_cb115/iter_40000.pth'

            #breakpoint()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #dacs diffusion backbone
    if id == 110:
        seeds = [42]
        for seed in seeds:
            #breakpoint()
           
            import os
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            os.environ['HF_CACHE'] = 'PATH_TO_FILL/cache'
            print(os.getenv('TRANSFORMERS_CACHE'))
            #breakpoint()
            opt = 'adamw_diff_bottomup_frozenEnc'
            #uda='dacs'
            uda = 'dacs_a999_fdthings_bottomup'
            batch_size = 1
            cfg = config_from_vars()
            cfg['activate_panoptic'] = activate_panoptic
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only.py', '_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only_diff.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/bottomup_sepaspp_mitb5.py', '_base_/models/bottomup_sepaspp_mitb5_diff_frozenEnc.py') for s in cfg['_base_']]
            #8000 iterations
            cfg['checkpoint_path'] = "PATH_TO_FILL/edaps_experiments/exp-00110/work_dirs/local-exp00110/231105_1728_syn2cs_dacs_a999_fdthings_bottomup_rcs001_cpl_bottomup_mitb5_poly10warm_s42_1a0e6/iter_40000.pth"
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00108/work_dirs/local-exp00108/230922_1514_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_232ee/iter_8000.pth'
            #40000 iterations
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00108/work_dirs/local-exp00108/230923_1723_syn2cs_source-only_bottomup_mitb5_poly10warm_s42_cb115/iter_40000.pth'

            #breakpoint()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
        #source only diffusion backbone
    if id == 106:
        seeds = [42]
        for seed in seeds:
            #uda='source-only'
            #breakpoint()
            
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['activate_panoptic'] = activate_panoptic
            #breakpoint()
            #cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic.py', '_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_source_only.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    if id == 109:
        seeds = [42]
        for seed in seeds:
            #breakpoint()
            import os
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            os.environ['HF_CACHE'] = 'PATH_TO_FILL/cache'
            print(os.getenv('TRANSFORMERS_CACHE'))
            #breakpoint()
            opt = 'adamw_diff_bottomup_frozenEnc'
            #uda='dacs'
            #uda='source-only'
            cfg = config_from_vars()
            cfg['activate_panoptic'] = activate_panoptic
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic.py', '_base_/datasets/uda_synthia_to_cityscapes_bottomup_panoptic_diff.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/bottomup_sepaspp_mitb5.py', '_base_/models/bottomup_sepaspp_mitb5_diff_frozenEnc.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    # ---------------------------------------------------------------------------------------------
    # S-Net (Table 5)
    # ---------------------------------------------------------------------------------------------
    elif id == 101:
        architecture = 'bottomup_snet'
        #breakpoint()
        seeds = [0,1,2]
        for seed in seeds:
            cfg = config_from_vars()
            cfg['activate_panoptic'] = activate_panoptic
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # ---------------------------------------------------------------------------------------------
    # M-Dec-BU (Table 5) : Evaluation and Prediction Visualization
    # ---------------------------------------------------------------------------------------------
    elif id == 102:
        seed = 0
        visuals_pan_eval = False # set it True for saving visualization
        batch_size = 1
        workers_per_gpu = 0
        cfg = config_from_vars()
        cfg['model']['activate_panoptic'] = activate_panoptic
        cfg['model']['decode_head']['activate_panoptic'] = activate_panoptic
        cfg['checkpoint_path'] = '/path/to/the/checkpoint'
        cfgs.append(cfg)
        for cfg_base in cfg['_base_']:
            print(cfg_base)

    return cfgs