# ---------------------------------------------------------------------------------------
# 2024 ETH Zurich, Elham Amin Mansour, on the basis of the work of Suman Saha, Lukas Hoyer 2022-2023.
# 
# ---------------------------------------------------------------------------------------

import itertools


def set_semantic_and_instance_loss_weights(cfg, loss_weight_semanitc, loss_weight_instance):
    cfg.setdefault('model', {})
    # daformer semantic head
    cfg['model'].setdefault('decode_head', {})
    cfg['model']['decode_head'].setdefault('loss_decode', {})
    cfg['model']['decode_head']['loss_decode']['loss_weight'] = loss_weight_semanitc
    # mask-rcnn instance head
    cfg['model'].setdefault('rpn_head', {})
    cfg['model']['rpn_head'].setdefault('loss_cls', {})
    cfg['model']['rpn_head'].setdefault('loss_bbox', {})
    cfg['model']['rpn_head']['loss_cls']['loss_weight'] = loss_weight_instance
    cfg['model']['rpn_head']['loss_bbox']['loss_weight'] = loss_weight_instance
    cfg['model'].setdefault('roi_head', {})
    cfg['model']['roi_head'].setdefault('bbox_head', {})
    cfg['model']['roi_head']['bbox_head'].setdefault('loss_cls', {})
    cfg['model']['roi_head']['bbox_head'].setdefault('loss_bbox', {})
    cfg['model']['roi_head']['bbox_head']['loss_cls']['loss_weight'] = loss_weight_instance
    cfg['model']['roi_head']['bbox_head']['loss_bbox']['loss_weight'] = loss_weight_instance
    cfg['model']['roi_head'].setdefault('mask_head', {})
    cfg['model']['roi_head']['mask_head'].setdefault('loss_mask', {})
    cfg['model']['roi_head']['mask_head']['loss_mask']['loss_weight'] = loss_weight_instance
    if loss_weight_semanitc == 0:
        cfg['evaluation']['metric'] = ['mAP']
    else:
        cfg['evaluation']['metric'] = ['mIoU']
    return cfg


def get_default_runtime_base():
    return '_base_/default_runtime_mmdet_mr.py'


def get_model_base_dacs(architecture, semantic_decoder, backbone, uda_model_type, ):
    if uda_model_type == 'dacs' or uda_model_type == 'advseg':
        dacs_model_base = f'_base_/models/{architecture}_{semantic_decoder}_{backbone}_diff.py'
    elif 'dacs_inst' in uda_model_type:
        dacs_model_base = f'_base_/models/{architecture}_{semantic_decoder}_{backbone}_dacsInst.py'
    else:
        raise NotImplementedError(f'No impl found for uda_model_type: {uda_model_type}')
    #breakpoint()
    return dacs_model_base


def get_model_base(architecture, backbone, uda, semantic_decoder='sepaspp', uda_model_type='dacs', ):
    dacs_model_base = None
    if uda == 'dacs' or uda == 'advseg':
        dacs_model_base = get_model_base_dacs(architecture, semantic_decoder, backbone, uda_model_type, )
    #breakpoint()
    return {
            'target-only': f'_base_/models/{architecture}_{semantic_decoder}_{backbone}_diff.py',
            'source-only': f'_base_/models/{architecture}_{semantic_decoder}_{backbone}_diff.py',
            'dacs':  dacs_model_base,
            'advseg': dacs_model_base, # does not matter if we are using adversarial training or dacs for UDA, the model is same
    }[uda]


def get_dataset_base_dacs(include_diffusion_data, source, target, evalScale):
    if not include_diffusion_data:
        if evalScale:
            dacs_dataset_base = f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic_evalScale_{evalScale}.py'
        else:
            dacs_dataset_base = f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic_diff.py'
        dacs_dataset_base = f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic_diff.py'#f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic_diffusion.py'
    #breakpoint()
    return dacs_dataset_base


def get_dataset_base(uda, source, target, include_diffusion_data=False, evalScale=None, fog_beta='0.02'):
    dacs_dataset_base=None
    if uda == 'dacs' or uda == 'advseg':
        if 'cityscapesFoggy' not in target:
            #breakpoint()
            dacs_dataset_base = get_dataset_base_dacs(include_diffusion_data, source, target, evalScale)
        else:
            dacs_dataset_base = f'_base_/datasets/uda_{source}_to_{target}_maskrcnn_panoptic_beta_{fog_beta}.py'
    #breakpoint()
    return {
            'target-only': f'_base_/datasets/{uda}_{target}_maskrcnn_panoptic_diff.py',
            'source-only': f'_base_/datasets/{uda}_{source}_to_{target}_maskrcnn_panoptic_diff.py',
            'dacs':        dacs_dataset_base,
            'advseg': dacs_dataset_base, # does not matter whther we use adversarial training or dacs for UDA, the dataset base would be the same
    }[uda]


def get_uda_base(uda_sub_type, uda_model_type='dacs'):
    if uda_model_type == 'dacs':
        uda_model = 'dacs'
    elif uda_model_type == 'dacs_inst':
        uda_model = 'dacs_inst'
    elif uda_model_type == 'dacs_inst_v2':
        uda_model = 'dacs_inst_v2'
    elif uda_model_type == 'advseg':
        uda_model = 'advseg'
    #breakpoint()
    if uda_model_type == 'advseg':
        return f'_base_/uda/{uda_model}.py'
    else:
        return f'_base_/uda/{uda_model}_{uda_sub_type}.py'


def get_optimizer_base(opt):
    return f'_base_/schedules/{opt}.py'


def get_schedule_base(schedule):
    return f'_base_/schedules/{schedule}.py'


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def get_eval_params(mask_score_threshold, debug, mapillary_dataloading_style,
                    semantic_pred_numpy_array_location=None,
                    dump_semantic_pred_as_numpy_array=False,
                    load_semantic_pred_as_numpy_array=False,
                    use_semantic_decoder_for_instance_labeling=False,
                    use_semantic_decoder_for_panoptic_labeling=False,
                    nms_th=None,
                    intersec_th=None,
                    upsnet_mask_pruning=False,
                    generate_thing_cls_panoptic_from_instance_pred=False,
                    ):

    train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0]
    thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
    panop_deeplab_eval_post_process_params = dict(num_classes=19,
                                                  ignore_label=255,
                                                  mapillary_dataloading_style=mapillary_dataloading_style,
                                                  label_divisor=1000,
                                                  train_id_to_eval_id=train_id_to_eval_id,
                                                  thing_list=thing_list,
                                                  mask_score_threshold=mask_score_threshold,
                                                  debug=debug,
                                                  dump_semantic_pred_as_numpy_array=dump_semantic_pred_as_numpy_array,
                                                  load_semantic_pred_as_numpy_array=load_semantic_pred_as_numpy_array,
                                                  semantic_pred_numpy_array_location=semantic_pred_numpy_array_location,
                                                  use_semantic_decoder_for_instance_labeling=use_semantic_decoder_for_instance_labeling,
                                                  use_semantic_decoder_for_panoptic_labeling=use_semantic_decoder_for_panoptic_labeling,
                                                  nms_th=nms_th,
                                                  intersec_th=intersec_th,
                                                  upsnet_mask_pruning=upsnet_mask_pruning,
                                                  generate_thing_cls_panoptic_from_instance_pred=generate_thing_cls_panoptic_from_instance_pred,
                                                  )
    return panop_deeplab_eval_post_process_params


def generate_experiment_cfgs(id, machine_name):
    def get_initial_cfg():
        return {
            'debug': debug,
            '_base_': [],
            'n_gpus': n_gpus,
            'gpu_mtotal': gpu_mtotal,
            'total_train_time': total_train_time,
            'n_cpus': n_cpus,
            'mem_per_cpu': mem_per_cpu,
            'machine': machine,
            'resume_from': resume_from,
            'load_from': load_from,
            'only_eval': only_eval,
            'only_train': only_train,
            'activate_auto_scale_lr': activate_auto_scale_lr,
            'auto_scale_lr': dict(enable=activate_auto_scale_lr, base_batch_size=16),
            'print_layer_wise_lr': print_layer_wise_lr,
            'file_sys': file_sys,
            'launcher': launcher,
            'generate_only_visuals_without_eval': generate_only_visuals_without_eval,
            'dump_predictions_to_disk': dump_predictions_to_disk,
            'evaluate_from_saved_png_predictions': evaluate_from_saved_png_predictions,
            'panop_eval_temp_folder_previous': panop_eval_temp_folder_previous,
            'exp_sub': exp_sub,
            'exp_root': exp_root,
        }

    def config_from_vars():
        #breakpoint()
        cfg = get_initial_cfg()
        # get default runtime base config
        cfg['_base_'].append(get_default_runtime_base())
        # set seed
        if seed is not None:
            cfg['seed'] = seed
        # get model base config
        cfg['_base_'].append(get_model_base(architecture, backbone, uda,
                                            semantic_decoder=semantic_decoder,
                                            uda_model_type=uda_model_type,
                                            )
                             )

        # get dataset base config
        cfg['_base_'].append(get_dataset_base(uda, source, target,
                                              include_diffusion_data=include_diffusion_data,
                                              evalScale=evalScale,
                                              fog_beta=fog_beta,
                                               )
                             )
        #print('----------------------base of cfg: ', cfg['_base_'])
        # get uda base config

        if ('dacs' in uda) or ('advseg' in uda):
            cfg['_base_'].append(get_uda_base(uda_sub_type, uda_model_type=uda_model_type))
        #
        if 'dacs' in uda and plcrop:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        elif 'dacs' in uda and not plcrop:
            cfg.setdefault('uda', {})

        if 'advseg' in uda:
            cfg.setdefault('uda', {})
            if plcrop:
                cfg['uda']['pseudo_weight_ignore_top'] = 15
                cfg['uda']['pseudo_weight_ignore_bottom'] = 120

        if 'dacs' in uda:
            cfg['uda']['share_src_backward'] = share_src_backward
            cfg['uda']['imnet_feature_dist_lambda'] = imnet_feature_dist_lambda
            cfg['uda']['alpha'] = mean_teacher_alpha
            cfg['uda']['pseudo_threshold'] = pseudo_threshold
            cfg['uda']['disable_mix_masks'] = disable_mix_masks
            cfg['uda']['debug_img_interval'] = debug_img_interval

        if 'advseg' in uda:
            cfg['uda']['imnet_feature_dist_lambda'] = imnet_feature_dist_lambda
            cfg['uda']['debug_img_interval'] = debug_img_interval

        if 'dacs_inst' in uda_model_type:
            cfg['uda']['activate_uda_inst_losses'] = activate_uda_inst_losses
            cfg['uda']['mix_masks_only_thing'] = mix_masks_only_thing
            cfg['uda']['inst_pseduo_weight'] = inst_pseduo_weight
            cfg['uda']['swtich_off_mix_sampling'] = swtich_off_mix_sampling  # NOT IN USE
            cfg['uda']['switch_off_self_training'] = switch_off_self_training # NOT IN USE
        #breakpoint()
        cfg['data'] = dict( samples_per_gpu=batch_size, workers_per_gpu=workers_per_gpu, train={})
        # setup config for RCS
        if  (('dacs' in uda) and rcs_T is not None) or (('advseg' in uda) and rcs_T is not None):
            cfg = setup_rcs(cfg, rcs_T)
        # Setup the ann_dir for validation
        cfg['data'].setdefault('val', {})
        cfg['data']['val']['ann_dir'] = ann_dir
        cfg['data']['val']['data_root'] = data_root_val
        if include_diffusion_data:
            # cfg.setdefault('data', {}).setdefault('train', {})
            cfg['data'].setdefault('train', {}).setdefault('target', {})
            cfg['data']['train']['target']['include_diffusion_data'] = include_diffusion_data
            cfg['data']['train']['target']['diffusion_set'] = diffusion_set

        # Setup optimizer
        # if 'dacs' in uda:
        cfg['optimizer_config'] = None  # Don't use outer optimizer
        # get optimizer base config
        cfg['_base_'].append(get_optimizer_base(opt))
        # get schedule base config
        cfg['_base_'].append(get_schedule_base(schedule))
        # set the learing rate of the backbone to lr
        # if pmult is True then set the learing rate of the neck and heads to lr*10.0
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            if set_diff_pmult_for_sem_and_inst_heads:
                assert pmult_inst_head, 'pmult_inst_head can not be None!'
                opt_param_cfg['decode_head'] = dict(lr_mult=10.) # semantic head
                opt_param_cfg['neck'] = dict(lr_mult=pmult_inst_head)  # this for the FPN
                opt_param_cfg['rpn_head'] = dict(lr_mult=pmult_inst_head)
                opt_param_cfg['roi_head'] = dict(lr_mult=pmult_inst_head)
            else:
                opt_param_cfg['neck'] = dict(lr_mult=10.)  # this for the FPN
                opt_param_cfg['head'] = dict(lr_mult=10.) # all heads: decode-head, fpn-head, roi-head

        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)
        # Set evaluation configs

        if use_class_specific_mask_score_th:
            mask_score_threshold_dynamic = mask_score_threshold_class_specific
        else:
            mask_score_threshold_dynamic = mask_score_threshold

        cfg['evaluation'] = dict(interval=eval_interval, metric=eval_metric_list,
                                 eval_type=eval_type, dataset_name=target,
                                 gt_dir=gt_dir_instance, gt_dir_panop=gt_dir_panoptic, num_samples_debug=num_samples_debug,
                                 post_proccess_params=get_eval_params(mask_score_threshold_dynamic, debug,
                                                                      mapillary_dataloading_style=mapillary_dataloading_style,
                                                                      semantic_pred_numpy_array_location=semantic_pred_numpy_array_location,
                                                                      dump_semantic_pred_as_numpy_array=dump_semantic_pred_as_numpy_array,
                                                                      load_semantic_pred_as_numpy_array=load_semantic_pred_as_numpy_array,
                                                                      use_semantic_decoder_for_instance_labeling=use_semantic_decoder_for_instance_labeling,
                                                                      use_semantic_decoder_for_panoptic_labeling=use_semantic_decoder_for_panoptic_labeling,
                                                                      nms_th=nms_th,
                                                                      intersec_th=intersec_th,
                                                                      upsnet_mask_pruning=upsnet_mask_pruning,
                                                                      generate_thing_cls_panoptic_from_instance_pred=generate_thing_cls_panoptic_from_instance_pred,  # TODO: False
                                                                      ),
                                 visuals_pan_eval=dump_visuals_during_eval,
                                 evalScale=evalScale,
                                 evaluate_from_saved_numpy_predictions=evaluate_from_saved_numpy_predictions,
                                 evaluate_from_saved_png_predictions=evaluate_from_saved_png_predictions,

                                 )
        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(by_epoch=False, interval=checkpoint_interval, max_keep_ckpts=2)
        # Set the log_interval
        cfg['log_config'] = dict(interval=log_interval)
        # Construct config name
        uda_mod = uda
        if  (('dacs' in uda) and (rcs_T is not None)) or (('advseg' in uda) and (rcs_T is not None)):
            uda_mod += f'_rcs{rcs_T}'
        if ('dacs' in uda) or ('advseg' in uda) and plcrop:
            uda_mod += '_cpl'
        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture}_' + f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' + f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') .replace('False', 'F').replace('cityscapes', 'cs').replace('synthia', 'syn')
        # returning the config for a single experiment
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    debug = False # TODO
    machine = machine_name
    iters = 8000
    interval = iters
    interval_debug = 3
    uda = 'dacs'
    data_root_val = './data/cityscapes'
    # ----------------------------------------
    # --- Set the debug time configs ---
    # ----------------------------------------
    n_gpus = 1 if debug else 1
    batch_size = 1 if debug else 2  # samples_per_gpu
    workers_per_gpu = 0 if debug else 4 # if 'dacs' in uda else 2
    eval_interval = interval_debug if debug else interval
    checkpoint_interval = 5000#40000#1800#interval_debug if debug else interval
    ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
    log_interval = 1 if debug else 50
    debug_img_interval = 1 if debug else 5000
    # ----------------------------------------
    architecture = 'maskrcnn'
    backbone = 'mitb5'
    models = [(architecture, backbone)]
    udas = [uda]
    uda_sub_type = 'a999_fdthings'
    source, target = 'synthia', 'cityscapes'
    datasets = [(source, target)]
    seed = 0
    plcrop = True
    rcs_T = 0.01
    imnet_feature_dist_lambda = 0.005
    opt = 'adamw_diff'
    schedule = 'poly10warm_diff'
    lr = 0.00008#0.00006
    pmult = True
    only_train = False
    only_eval = False
    eval_type = 'maskrcnn_panoptic'
    resume_from = None
    load_from = None
    activate_auto_scale_lr = False
    print_layer_wise_lr = False
    share_src_backward = True
    uda_model_type = 'dacs'
    activate_uda_inst_losses = False
    mix_masks_only_thing = False
    inst_pseduo_weight = None
    num_samples_debug = 12
    gt_dir_instance = './data/cityscapes/gtFine/val'
    gt_dir_panoptic = './data/cityscapes/gtFine_panoptic'
    eval_metric_list = ['mIoU', 'mPQ', 'mAP']
    mapillary_dataloading_style = 'OURS'
    set_diff_pmult_for_sem_and_inst_heads = False
    semantic_decoder = 'sepaspp'
    dump_semantic_pred_as_numpy_array = False
    load_semantic_pred_as_numpy_array = False
    semantic_pred_numpy_array_location = None
    mask_score_threshold = 0.95
    mask_score_threshold_class_specific = None
    use_class_specific_mask_score_th = False
    use_semantic_decoder_for_instance_labeling = False  # Not in use
    use_semantic_decoder_for_panoptic_labeling = False  # Not in use
    launcher = None
    upsnet_mask_pruning = False
    generate_thing_cls_panoptic_from_instance_pred = False
    nms_th = None
    intersec_th = None
    generate_only_visuals_without_eval = False
    dump_predictions_to_disk = False
    # diffusion data
    include_diffusion_data = False
    diffusion_set = None
    pmult_inst_head = None
    evalScale = None
    evaluate_from_saved_numpy_predictions = False
    evaluate_from_saved_png_predictions = False
    panop_eval_temp_folder_previous = None
    mean_teacher_alpha = 0.999
    pseudo_threshold = 0.968
    disable_mix_masks = False
    # The below params are not in use
    swtich_off_mix_sampling = False
    switch_off_self_training = False
    dump_visuals_during_eval = True #False  # if True, save the predictions to disk at evaluation
    exp_root = "PATH_TO_FILL/edaps_experiments"#"PATH_TO_FILL"
    exp_sub = f'exp-{id:05d}'
    # override experiment folders, if they are not none, these values will be used
    # override_exp_folders = False
    # str_unique_name = None
    str_panop_eval_temp_folder = None

    # Euler Stuff
    n_cpus = 16
    mem_per_cpu = 16000
    gpu_mtotal = 23000
    total_train_time = '48:00:00' # TODO: for public release set it to 24
    file_sys = 'Slurm'

    #
    fog_beta = 0.02 # 0.005, 0.01 or 0.02

    cfgs = []

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes (Table 1)
    # -------------------------------------------------------------------------
    if id == 1:
        seeds = [0,1,2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # ---------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes with mitb3 backbone for ICCV2023 Rebuttal
    # ---------------------------------------------------------------------------------------------------
    elif id == 11:
        backbone = 'mitb3'
        print_layer_wise_lr = False
        seeds = [0, 1, 2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    # ---------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes with mitb2 backbone for ICCV2023 Rebuttal
    # ---------------------------------------------------------------------------------------------------
    elif id == 12:
        total_train_time = '24:00:00'
        backbone = 'mitb2'
        print_layer_wise_lr = False
        seeds = [0, 1, 2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # --------------------------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes train with Adverserial loss (for ICCV2023 Rebuttal)
    # backbone = 'mitb5', rcs_T = None, imnet_feature_dist_lambda = 0
    # --------------------------------------------------------------------------------------------------------------------
    elif id == 13:
        backbone = 'mitb5' if debug else 'mitb5' # OR r101v1c
        total_train_time = '48:00:00'
        opt = 'adamw' # Or sgd
        architecture = 'maskrcnn'
        rcs_T = None # 0.01 OR None
        imnet_feature_dist_lambda = 0 # 0.005 Or 0
        pmult = True # if sgd, then set it to False
        print_layer_wise_lr = False
        uda = 'advseg'
        uda_sub_type = uda
        uda_model_type = uda
        plcrop = False
        seeds = [0, 1, 2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # --------------------------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes train with Adverserial loss (for ICCV2023 Rebuttal)
    # backbone = 'mitb5', rcs_T = 0.01, imnet_feature_dist_lambda = 0
    # --------------------------------------------------------------------------------------------------------------------
    elif id == 14:
        backbone = 'mitb5'  # OR r101v1c
        total_train_time = '48:00:00'
        opt = 'adamw'  # Or sgd
        architecture = 'maskrcnn'
        rcs_T = 0.01  # 0.01 OR None
        imnet_feature_dist_lambda = 0  # 0.005 Or 0
        pmult = True  # if sgd, then set it to False
        print_layer_wise_lr = False
        uda = 'advseg'
        uda_sub_type = uda
        uda_model_type = uda
        plcrop = False
        seeds = [0,1,2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # --------------------------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes train with Adverserial loss (for ICCV2023 Rebuttal)
    # backbone = 'mitb5', rcs_T =None, imnet_feature_dist_lambda = 0.005
    # --------------------------------------------------------------------------------------------------------------------
    elif id == 15:
        backbone = 'mitb5'  # OR r101v1c
        total_train_time = '48:00:00'
        opt = 'adamw'  # Or sgd
        architecture = 'maskrcnn'
        rcs_T = None  # 0.01 OR None
        imnet_feature_dist_lambda = 0.005  # 0.005 Or 0
        pmult = True  # if sgd, then set it to False
        print_layer_wise_lr = False
        uda = 'advseg'
        uda_sub_type = uda
        uda_model_type = uda
        plcrop = False
        seeds = [0,1,2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # --------------------------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes train with Adverserial loss (for ICCV2023 Rebuttal)
    # backbone = 'mitb5', rcs_T =0.01, imnet_feature_dist_lambda = 0.005
    # --------------------------------------------------------------------------------------------------------------------
    elif id == 16:
        backbone = 'mitb5'  # OR r101v1c
        total_train_time = '48:00:00'
        opt = 'adamw'  # Or sgd
        architecture = 'maskrcnn'
        rcs_T = 0.01  # 0.01 OR None
        imnet_feature_dist_lambda = 0.005  # 0.005 Or 0
        pmult = True  # if sgd, then set it to False
        print_layer_wise_lr = False
        uda = 'advseg'
        uda_sub_type = uda
        uda_model_type = uda
        plcrop = False
        seeds = [0,1,2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # --------------------------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes train with Adverserial loss (for ICCV2023 Rebuttal)
    # backbone = 'r101v1c', rcs_T = None, imnet_feature_dist_lambda = 0
    # --------------------------------------------------------------------------------------------------------------------
    elif id == 17:
        architecture = 'maskrcnn'
        semantic_decoder = 'dlv2red'
        backbone = 'r101v1c'  # OR r101v1c
        total_train_time = '48:00:00'
        opt = 'sgd'  # Or sgd
        rcs_T = None  # 0.01 OR None
        imnet_feature_dist_lambda = 0  # 0.005 Or 0
        pmult = False  # if sgd, then set it to False
        print_layer_wise_lr = False
        uda = 'advseg'
        uda_sub_type = uda
        uda_model_type = uda
        plcrop = False
        seeds = [0, 1, 2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # --------------------------------------------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes train with Adverserial loss (for ICCV2023 Rebuttal)
    # backbone = 'r101v1c', rcs_T =0.01, imnet_feature_dist_lambda = 0.005
    # --------------------------------------------------------------------------------------------------------------------
    elif id == 18:
        architecture = 'maskrcnn'
        semantic_decoder = 'dlv2red'
        backbone = 'r101v1c'  # OR r101v1c
        total_train_time = '48:00:00'
        opt = 'sgd'  # adamw Or sgd
        rcs_T = 0.01  # 0.01 OR None
        imnet_feature_dist_lambda = 0.005  # 0.005 Or 0
        pmult = False  # if sgd, then set it to False
        print_layer_wise_lr = False
        uda = 'advseg'
        uda_sub_type = uda
        uda_model_type = uda
        plcrop = False
        seeds = [0, 1, 2]
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)


    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : Cityscapes → CityscapesFoggy :
    # -------------------------------------------------------------------------
    elif id == 19:
        # gpu_mtotal = 10000 # for debug only
        # total_train_time = '24:00:00'  # for debug only
        backbone = 'mitb2' if debug else 'mitb5'
        rcs_T = None if debug else 0.01

        fog_beta = 0.02  # 0.005, 0.01 or 0.02
        source = 'cityscapes'
        target = 'cityscapesFoggy'
        seeds = [0, 1, 2]
        data_root_val = 'data/cityscapesFoggy'
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : Cityscapes → CityscapesFoggy :
    # -------------------------------------------------------------------------
    elif id == 20:
        fog_beta = 0.01  # 0.005, 0.01 or 0.02
        source = 'cityscapes'
        target = 'cityscapesFoggy'
        seeds = [0, 1, 2]
        data_root_val = 'data/cityscapesFoggy'
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : Cityscapes → CityscapesFoggy :
    # -------------------------------------------------------------------------
    elif id == 21:
        fog_beta = 0.005  # 0.005, 0.01 or 0.02
        source = 'cityscapes'
        target = 'cityscapesFoggy'
        seeds = [0, 1, 2]
        data_root_val = 'data/cityscapesFoggy'
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)


    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : Cityscapes → CityscapesFoggy:
    # Dump Visualization on the CityscapesFoggy valset
    # -------------------------------------------------------------------------
    elif id == 22:
        # gpu_mtotal = 10000 # for debug only
        total_train_time = '04:00:00'  # for debug only
        fog_beta = 0.02  # 0.005, 0.01 or 0.02
        source = 'cityscapes'
        target = 'cityscapesFoggy'
        data_root_val = 'data/cityscapesFoggy'
        batch_size = 1
        workers_per_gpu = 0
        #evaluate_from_saved_png_predictions=True
        generate_only_visuals_without_eval = False
        dump_visuals_during_eval = True
        checkpoint_path = 'PATH_TO_FILL/experiments/daformer_panoptic_experiments/euler-exp00019/230526_1015_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a7d96'
        # checkpoint_path = 'PATH_TO_FILL/DATADISK2/apps/experiments/edaps_experiments/euler-exp00019/230526_1015_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a7d96'
        cfg = config_from_vars()
        cfg['checkpoint_path'] = checkpoint_path
        #breakpoint()
        cfgs.append(cfg)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Mapillary (Table 2)
    # -------------------------------------------------------------------------
    elif id == 2:
        data_root_val = 'data/mapillary'
        ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls' #  json file contains panoptic ground truths
        target = 'mapillary'
        num_samples_debug = 13
        gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
        gt_dir_panoptic = 'data/mapillary'
        seeds = [1, 2] # [0, 1, 2]
        plcrop = False
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # --------------------------------------------------------------------------------
    # EDAPS (M-Dec-TD)
    # Ablation study of the UDA strategies on SYNTHIA → Cityscapes
    # (Table 7 - row 2nd to row 5th)
    # --------------------------------------------------------------------------------
    elif id == 3:
        seeds = [0, 1, 2]
        mean_teacher_alphas = [0, 0.999] # (MT)
        imnet_feature_dist_lambdas = [0.0, 0.005] # (FD)
        rcs_Ts = [None, 0.01] # (RCS)
        for seed, mean_teacher_alpha, imnet_feature_dist_lambda, rcs_T in itertools.product(seeds, mean_teacher_alphas, imnet_feature_dist_lambdas, rcs_Ts):
            cfg = config_from_vars()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes
    # source-only and target-only (oracle or supervised) models
    # (Table 3 bottom row; Table 7 top row : Source-only model)
    # -------------------------------------------------------------------------
    elif id == 4:
        udas = [
            'source-only',
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #checkpoint_path='PATH_TO_FILL/exp-00004/work_dirs/local-exp00004/230908_2313_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_e4eda'
            checkpoint_path='PATH_TO_FILL/exp-00004/work_dirs/local-exp00004/230912_1424_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_e5b84'#4000 iterations
            cfg['checkpoint_path'] = checkpoint_path
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
        #breakpoint()
    #diffusion backbone validation 
    elif id == 985:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            generate_only_visuals_without_eval = True
            #eval_metric_list = ['mIoU']
            batch_size = 1
            workers_per_gpu = 0
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen.py') for s in cfg['_base_']]
            #checkpoint_path='PATH_TO_FILL/exp-00990/work_dirs/local-exp00990/230822_1737_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_7a914/'
            checkpoint_path='PATH_TO_FILL/exp-00988/work_dirs/local-exp00988/230824_0259_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_12533/'
            cfg['checkpoint_path'] = checkpoint_path

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone validation 
    elif id == 989:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            generate_only_visuals_without_eval = True
            #eval_metric_list = ['mIoU']
            batch_size = 1
            workers_per_gpu = 0
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_unfrozen.py') for s in cfg['_base_']]
            checkpoint_path='PATH_TO_FILL/exp-00990/work_dirs/local-exp00990/230822_1737_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_7a914/'
            cfg['checkpoint_path'] = checkpoint_path

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training without freezing
    elif id == 990:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            opt = 'adamw_diff_unfrozen'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_unfrozen.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone only semantic validation from checkpoint
    elif id == 994:
        udas = [
            'target-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            eval_metric_list = ['mIoU']
            batch_size = 1
            workers_per_gpu = 0
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic.py') for s in cfg['_base_']]
            checkpoint_path='PATH_TO_FILL/exp-00996/work_dirs/local-exp00996/230816_0045_syn2cs_target-only_maskrcnn_mitb5_poly10warm_s0_e1365/'
            cfg['checkpoint_path'] = checkpoint_path

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    
    #diffusion backbone only semantic square validation from checkpoint
    elif id == 995:
        udas = [
            'target-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            eval_metric_list = ['mIoU']
            batch_size = 1
            workers_per_gpu = 0
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff_square.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic.py') for s in cfg['_base_']]
            checkpoint_path = 'PATH_TO_FILL/exp-00993/work_dirs/local-exp00993/230816_1410_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_8ade7/'
            cfg['checkpoint_path'] = checkpoint_path

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training only semantic square
    elif id == 996:
        udas = [
            'target-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff_square.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training only instance with no freezing
    elif id == 988:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'

            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training only instance with no freezing
    elif id == 984 or id == 983 or id == 982 or id ==981 or id==980 or id==979 or id==978 or id==977 or id == 976 or id == 975:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'

            cfg = config_from_vars()
            if(id==984):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi1.py') for s in cfg['_base_']]
            elif(id==983):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi2.py') for s in cfg['_base_']]
            elif(id==982):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi3.py') for s in cfg['_base_']]
            elif(id==981):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi4.py') for s in cfg['_base_']]
            elif(id==980):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi5.py') for s in cfg['_base_']]
            elif(id==979):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi6.py') for s in cfg['_base_']]
            elif(id==978):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet2_roi0.py') for s in cfg['_base_']]
            elif(id==977):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roiLossNothing.py') for s in cfg['_base_']]
            elif(id==976):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
            elif(id==975):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #frozen ones
    elif id == 971 or id == 972:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            opt = 'adamw_diff'
            schedule = 'poly10warm'

            cfg = config_from_vars()
            if(id==972):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
            elif(id==971):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unet_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 967:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_decay'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi1_t0_bumbleHyper_reverse.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 965:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 964:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_thres.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 963:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_frozen_roi1_t0_bumbleHyper_scaleRoi_thres.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 962:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_frozen_roi1_t0_bumbleHyper_scaleRoi_thresboth.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 961:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_frozen_roi1_t0_bumbleHyper_scaleRoi_thresboth_rpn.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 960:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_rpn.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 920:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_2'
            schedule = 'poly10warm_diff'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_rpn.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 918:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 908:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure_instanceOnly.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 907:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure_semanticOnly.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 906:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 897:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 895:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101_unfrozen.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    elif id == 873:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=40000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101.py') for s in cfg['_base_']] #
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00873/work_dirs/local-exp00873/231019_1403_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_resnet_s0_09370/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 874:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101_unfrozen.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    elif id == 858:
        import os
        udas = [
            'source-only'
            #'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=40000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101_unfrozen.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base) 
    elif id == 875:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=40000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101_unfrozen.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)    
    elif id == 870:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_unfrozenLikeDiff_play'
            schedule = 'poly10warm_play'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101_unfrozen_play.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)    
    elif id == 894:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 883:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 905:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)    
    elif id == 904:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly_simplefpn.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 900:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly_simplefpn_tuned.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 901:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly_simplefpnDec.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 898:#901
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly_simplefpnDec_tuned.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 893:#901
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPure_simplefpnDec_tuned.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 882:#901
        import os
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']

            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_resnet'
            #schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPure_simplefpnDec_tuned.py') for s in cfg['_base_']] #
            #cfg['checkpoint_path'] ="PATH_TO_FILL/edaps_experiments/exp-00882/work_dirs/local-exp00882/231013_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_resnet_s0_dfcc3/iter_200000.pth"
            #breakpoint()
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 843:#901
        import os
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']
            #eval_metric_list = ['mAP']

            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_resnet'
            #schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPure_simplefpnDec_tuned_justSem.py') for s in cfg['_base_']] #
            #cfg['checkpoint_path'] ="PATH_TO_FILL/edaps_experiments/exp-00882/work_dirs/local-exp00882/231013_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_resnet_s0_dfcc3/iter_200000.pth"
            #breakpoint()
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 846:#901
        import os
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']

            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_resnet'
            #schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPure_simplefpnDec_tuned2.py') for s in cfg['_base_']] #
            #cfg['checkpoint_path'] ="PATH_TO_FILL/edaps_experiments/exp-00882/work_dirs/local-exp00882/231013_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_resnet_s0_dfcc3/iter_200000.pth"
            #breakpoint()
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    elif id == 845:#901
        import os
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']

            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_resnet'
            #schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPure_simplefpnDec_tuned3.py') for s in cfg['_base_']] #
            #cfg['checkpoint_path'] ="PATH_TO_FILL/edaps_experiments/exp-00882/work_dirs/local-exp00882/231013_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_resnet_s0_dfcc3/iter_200000.pth"
            #breakpoint()
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 877:#901
        import os
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPure_simplefpnDec_tuned.py') for s in cfg['_base_']] #
            cfg['checkpoint_path'] ="PATH_TO_FILL/edaps_experiments/exp-00882/work_dirs/local-exp00882/231013_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_resnet_s0_dfcc3/iter_200000.pth"
            #breakpoint()
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    elif id == 885:#901
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_vit2'#'adamw_resnet'
            schedule = 'poly10warm'#'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPure_simplefpnDec_tuned.py') for s in cfg['_base_']] #
            checkpointPath="PATH_TO_FILL/edaps_experiments/exp-00885/work_dirs/local-exp00885/231012_0140_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_7169f/iter_280000.pth"
            cfg['checkpoint_path'] = checkpointPath
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 892:#901
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_noFPN.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 891:
        #breakpoint()
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'

            opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

                
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_simpleFPNOnly.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    
    elif id == 903:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly_noFPN.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 899: #903
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly_noFPN_tuned.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 902:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_frozen'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_viTPureInstanceOnly_artificialFPN.py') for s in cfg['_base_']] #
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 910:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet'
            schedule = 'poly10warm_resnet'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnet_fpn.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 909:
        import os
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_vit'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_vit.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 959:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_where'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            checkpointPath='PATH_TO_FILL/exp-00960/work_dirs/local-exp00960/230905_1540_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_083c9'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_whereScale.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = checkpointPath
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 9592:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_3'
            schedule = 'poly10warm_diff'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            #checkpointPath='PATH_TO_FILL/exp-00960/work_dirs/local-exp00960/230905_1540_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_083c9'
            #checkpointPath='PATH_TO_FILL/exp-00920/work_dirs/local-exp00920/230914_1348_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_04816' the frozen rpn part with the original nms
            checkpointPath='PATH_TO_FILL/exp-09592/work_dirs/local-exp09592/230914_1656_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_32d15'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_whereScale.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = checkpointPath
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 931:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_Both'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            #checkpointPath='PATH_TO_FILL/exp-00960/work_dirs/local-exp00960/230905_1540_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_083c9'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedOnlyMask.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = checkpointPath
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 957:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_Both'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = checkpointPath
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 956 or id == 955 or id == 954 or id == 953 or id == 952 or id == 951 or id == 950 or id == 949 or id == 948 or id == 947 or id == 946 or id == 945 or id == 942 or id == 930 or id == 929 or id == 928 or id == 927 or id == 926 or id == 925 or id == 924 or id == 923 or id==922 or id==921 or id==9232 or id==919 or id==917 or id==916 or id==915 or id==914 or id==912 or id==913:
        udas = [
            'dacs'#'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_Both'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm'#'poly10warm_diff'#changed from nothing to _diff for 925
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            #checkpointPath='PATH_TO_FILL/exp-00960/work_dirs/local-exp00960/230905_1540_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_083c9'
            if(id == 956):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached5.py') for s in cfg['_base_']]
            elif(id == 955):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached4.py') for s in cfg['_base_']]
            elif(id == 954):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached3.py') for s in cfg['_base_']]
            elif(id == 953):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached2.py') for s in cfg['_base_']]
            elif(id == 952):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached1.py') for s in cfg['_base_']]
            elif(id == 951):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached.py') for s in cfg['_base_']]
            elif(id == 929):
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00951/work_dirs/local-exp00951/230907_2126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_86390' #detached 951
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00952/work_dirs/local-exp00952/230907_2126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_975af' #detached1 952
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00953/work_dirs/local-exp00953/230907_2126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_4b459' #detached2 953
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00954/work_dirs/local-exp00954/230907_2126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b8913' #detached3 954
                cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00956/work_dirs/local-exp00956/230907_2126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_f1857' #detached5 956
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_detached5.py') for s in cfg['_base_']] 
            elif(id == 950):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_compDetached.py') for s in cfg['_base_']]
            elif(id == 949):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_compDetached1.py') for s in cfg['_base_']]
            elif(id == 948):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_compDetached2.py') for s in cfg['_base_']]
            elif(id == 924):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_compDetached3.py') for s in cfg['_base_']]
            elif(id == 923):
                cfg['checkpoint_path'] ='PATH_TO_FILL/exp-00923/work_dirs/local-exp00923/230912_1927_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_ee3e7'
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_compDetached4.py') for s in cfg['_base_']]
            elif(id == 9232):
                opt = 'adamw_diff_unfrozen_3'
                schedule = 'poly10warm_diff'
                #opt = 'adamw_diff_unfrozen_hyper'
                #schedule = 'poly10warm'
                cfg = config_from_vars()
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_compDetached4-2.py') for s in cfg['_base_']]
                cfg['checkpoint_path'] ='PATH_TO_FILL/exp-09232/work_dirs/local-exp09232/230914_1657_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_01ea4'
            elif(id == 947):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached.py') for s in cfg['_base_']]
            elif(id == 930):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedmaskAct.py') for s in cfg['_base_']]
            elif(id == 928):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedclsAct.py') for s in cfg['_base_']]
            elif(id == 927):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedrep.py') for s in cfg['_base_']]
            elif(id == 926):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedActAll.py') for s in cfg['_base_']]
            elif(id == 925):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedTuneLosses.py') for s in cfg['_base_']]
            elif(id == 922):
                opt = 'adamw_diff_unfrozen_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                cfg = config_from_vars()
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedTuneAdamWeight.py') for s in cfg['_base_']]
            elif(id == 919):
                import os
                os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
                opt = 'adamw_resnet_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                cfg = config_from_vars()
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedGradientClip.py') for s in cfg['_base_']]
            elif(id == 913):
                import os
                os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
                opt = 'adamw_resnet_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                eval_metric_list = ['mIoU', 'mPQ', 'mAP']
                cfg = config_from_vars()
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedGradientClip_withSemantic.py') for s in cfg['_base_']]
            elif(id == 917):
                import os
                os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
                opt = 'adamw_resnet_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                cfg = config_from_vars()
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedGradientClip_scale30.py') for s in cfg['_base_']]
            elif(id == 916):
                import os
                os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
                opt = 'adamw_resnet_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                cfg = config_from_vars()
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedGradientClip_scale60.py') for s in cfg['_base_']]
            elif(id == 915):
                os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
                opt = 'adamw_resnet_diff_frozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                cfg = config_from_vars()
                cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00915/work_dirs/local-exp00915/230918_1653_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_4982a'
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_frozen_t0_scaleRoi_thresboth_gradientClip.py') for s in cfg['_base_']]
            elif(id == 912):
                import os
                os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
                opt = 'adamw_resnet_diff_frozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                eval_metric_list = ['mIoU', 'mPQ', 'mAP']
                cfg = config_from_vars()
                cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00912/work_dirs/local-exp00912/230919_1802_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_19d17'
                #'PATH_TO_FILL/edaps_experiments/exp-00915/work_dirs/local-exp00915/230918_1653_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_4982a'
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_frozen_t0_scaleRoi_thresboth_gradientClip_withSemantic.py') for s in cfg['_base_']]
            elif(id == 914):
                import os
                os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
                opt = 'adamw_resnet_diff_frozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                cfg = config_from_vars()
                cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00914/work_dirs/local-exp00914/230918_1636_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_cf53c'
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_frozen_t0_scaleRoi_thresboth_gradientClip_scales.py') for s in cfg['_base_']]
            elif(id == 921):
                batch_size=2
                opt = 'adamw_diff_unfrozen_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
                schedule = 'poly10warm_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
                cfg = config_from_vars()
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedNoAttn.py') for s in cfg['_base_']]
            elif(id == 946):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached1.py') for s in cfg['_base_']]
            elif(id == 945):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached2.py') for s in cfg['_base_']]
            elif(id == 942):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetachedTuneAdamWeight.py') for s in cfg['_base_']]
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00945/work_dirs/local-exp00945/230908_0127_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_cece9' # the masks have very low scores look like boxes
                #cfg['checkpoint_path'] ='PATH_TO_FILL/exp-00946/work_dirs/local-exp00946/230908_0127_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_48eac'#946 the rpn is lower full of gtif
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00937/work_dirs/local-exp00937/230909_1834_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_f79d3' #937 the mask looks bad the bbox looks good 2-1
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00936/work_dirs/local-exp00936/230909_1837_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_bb21c' #936 2-2
                #cfg['checkpoint_path'] ='PATH_TO_FILL/exp-00947/work_dirs/local-exp00947/230908_0127_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_8c06a' #947
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00922/work_dirs/local-exp00922/230913_0140_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_90fb2' #rare sweep 922 adam tuning
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00922/work_dirs/local-exp00922/230913_0147_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_b026f' #smart sweep 922 adam tuning
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00922/work_dirs/local-exp00922/230913_0345_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_4b562' # ancient sweep 922 adam tuning crashed to grid
                cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00922/work_dirs/local-exp00922/230913_0508_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_d7e3e'#worldly sweep 922 adam tuning crashes to grid
                #cfg['checkpoint_path'] = 'PATH_TO_FILL/exp-00931/work_dirs/local-exp00931/230910_1655_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_29253'
                #cfg['checkpoint_path'] ='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230906_1849_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_5420c'
                #cfg['checkpoint_path'] ='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230906_1806_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_f1bdd'
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 941 or id == 940 or id == 939 or id == 938 or id == 937 or id == 936 or id == 935 or id == 934 or id == 933 or id == 932 :
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_Both'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            if(id == 941):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached1-1.py') for s in cfg['_base_']]
            elif(id == 940):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached1-2.py') for s in cfg['_base_']]
            elif(id == 939):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached1-3.py') for s in cfg['_base_']]
            elif(id == 938):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached1-4.py') for s in cfg['_base_']]
            elif(id == 937):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached2-1.py') for s in cfg['_base_']]
            elif(id == 936):
                #cfg['checkpoint_path'] 'PATH_TO_FILL/exp-00936/work_dirs/local-exp00936/230912_1441_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_90459'
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached2-2.py') for s in cfg['_base_']]
            elif(id == 935):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached2-3.py') for s in cfg['_base_']]
            elif(id == 934):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached2-4.py') for s in cfg['_base_']]
            elif(id == 933):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached2-5.py') for s in cfg['_base_']]
            elif(id == 932):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_bothWeighted_nondetached2-6.py') for s in cfg['_base_']]

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 944:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_where'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            #checkpointPath='PATH_TO_FILL/exp-00960/work_dirs/local-exp00960/230905_1540_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_083c9'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_whereScale_fixingReceptive.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = checkpointPath
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 943:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_where'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            checkpointPath='PATH_TO_FILL/exp-00944/work_dirs/local-exp00944/230908_1202_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_f76a9'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_whereScale_fixingReceptive.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = checkpointPath
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 958:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_where'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            checkpointPath='PATH_TO_FILL/exp-00960/work_dirs/local-exp00960/230905_1540_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_083c9' #testing
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230908_1403_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_334fe'
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230908_1457_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_c58a0'
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230908_1516_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_710dc' #decoder off
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_1245_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_59645' #decoder on
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_1503_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_67e27' #40 decoder on threshold changed
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_1503_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_f1895' #20 decoder on threshold changed
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_1502_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_18dd3' #100 decoder on thresold changed this sucks
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_1502_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_64ecd' #56 decoder on thresold changed  looks good
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_1842_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_46e35' #decoder off old threshold scale 56
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_2354_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_619a8'#0.5 nms
            #checkpointPath='PATH_TO_FILL/exp-00959/work_dirs/local-exp00959/230910_2323_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_7dd63'#0.9 nms
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_scaleRoi_whereScale.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = checkpointPath
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 966:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            batch_size = 1
            workers_per_gpu = 0

            checkpointPath='PATH_TO_FILL/exp-00967/work_dirs/local-exp00967/230831_1647_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_374d0/keep/'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi1_t0_bumbleHyper_reverse.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            cfg['checkpoint_path'] = checkpointPath
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 968:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            opt = 'adamw_diff_unfrozen_hyper'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper_reverse.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 970 or id == 969:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            batch_size = 1
            workers_per_gpu = 0

            cfg = config_from_vars()
            if(id==970):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
                checkpoint_path='PATH_TO_FILL/exp-00972/work_dirs/local-exp00972/230830_0237_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_4f94f/'
                cfg['checkpoint_path']= checkpoint_path
            elif(id==969):
                cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unet_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
                checkpoint_path='PATH_TO_FILL/exp-00971/work_dirs/local-exp00971/230830_0051_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_d368d/'
                cfg['checkpoint_path'] = checkpoint_path

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone validation for instance only
    elif id == 974:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            generate_only_visuals_without_eval = True
            #eval_metric_list = ['mIoU']
            batch_size = 1
            workers_per_gpu = 0
            cfg = config_from_vars()
            #cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #checkpoint_path='PATH_TO_FILL/exp-00990/work_dirs/local-exp00990/230822_1737_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_7a914/'
            checkpoint_path='PATH_TO_FILL/exp-00975/work_dirs/local-exp00975/230830_0255_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_2cc20/'
            cfg['checkpoint_path'] = checkpoint_path

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion unet validation for instance only
    elif id == 973:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            generate_only_visuals_without_eval = True
            #eval_metric_list = ['mIoU']
            batch_size = 1
            workers_per_gpu = 0
            cfg = config_from_vars()
            #cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlyInst_unfrozen_unet_roi1_t0_bumbleHyper.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff_square.py') for s in cfg['_base_']]
            #checkpoint_path='PATH_TO_FILL/exp-00990/work_dirs/local-exp00990/230822_1737_syn2cs_source-only_maskrcnn_mitb5_poly10warm_diff_s0_7a914/'
            checkpoint_path='PATH_TO_FILL/exp-00976/work_dirs/local-exp00976/230828_2037_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_9fef2/'
            cfg['checkpoint_path'] = checkpoint_path

            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    
    #diffusion backbone training only semantic
    elif id == 1000:
        udas = [
            'target-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            eval_metric_list = ['mIoU']
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training only semantic
    elif id == 993:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            eval_metric_list = ['mIoU']
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training only semantic fpn head
    elif id == 991:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #breakpoint()
            eval_metric_list = ['mIoU']
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_fpnhead.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training only semantic without freezing
    elif id == 992:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']
            opt = 'adamw_diff_unfrozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm'#'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=8000#8000#8000
            cfg['evaluation']['interval']=8000#8000#8000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 886:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt = 'adamw_diff_unfrozen'#'adamw_resnet_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm'#'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000#8000
            cfg['evaluation']['interval']=320000#8000#8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_ver2.py') for s in cfg['_base_']]
            #the checkpoint for 8000
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00886/work_dirs/local-exp00886/231009_0255_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_26374/iter_8000.pth' 
            #the checkpoint for 40000
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00886/work_dirs/local-exp00886/231009_1153_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_7da97/iter_40000.pth'
            #the checkpoint for 80000
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00886/work_dirs/local-exp00886/231012_0140_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_bea6f/iter_80000.pth'
            #the checkpoint for 120000
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00886/work_dirs/local-exp00886/231012_0140_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_bea6f/iter_120000.pth'

            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 881:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt = 'adamw_diff_unfrozen'#'adamw_resnet_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000#8000
            cfg['evaluation']['interval']=320000#8000#8000
            #breakpoint()
            #cfg['checkpoint_config']['interval']=8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_ver2.py') for s in cfg['_base_']]

            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00881/work_dirs/local-exp00881/231013_0355_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_193ce/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 884:
        udas = [
            'source-only'
        ]
        #breakpoint()
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt = 'adamw_mask2former'#'adamw_resnet_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly_mask2former'#'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            #breakpoint()
            cfg['runner']['max_iters']=40000#8000#8000
            cfg['evaluation']['interval']=40000#8000#8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/mask2former_mitb5_diff_OnlySemantic_unfrozen_freezing_withInstance.py') for s in cfg['_base_']]
            #the checkpoint for 8000
            
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_mask2former.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00886/work_dirs/local-exp00886/231009_0255_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_26374/iter_8000.pth' 
            
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 889:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']#, 'mPQ', 'mAP']
            opt = 'adamw_resnet_diff_frozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_freezing.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=20000#8000#8000
            cfg['evaluation']['interval']=20000#8000#8000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 880:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']#, 'mPQ', 'mAP']
            opt = 'adamw_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm'#'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
            #opt = 'adamw_resnet_diff_frozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_freezing.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00880/work_dirs/local-exp00880/231019_0130_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_4c557/iter_120000.pth'
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00880/work_dirs/local-exp00880/231021_2047_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_c02e0/iter_96000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000#8000
            cfg['evaluation']['interval']=320000#8000#8000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 888:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt = 'adamw_resnet_diff_frozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_freezing_withInstance.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000
            cfg['evaluation']['interval']=320000#8000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 887:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt = 'adamw_resnet_diff_frozen'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
            
            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_freezing_withInstanceSimpleNeck.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000
            cfg['evaluation']['interval']=320000#8000
            
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 879:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt = 'adamw_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_freezing_withInstance.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000
            cfg['evaluation']['interval']=320000#8000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 878:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt='adamw_diff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_freezing_withInstanceSimpleNeck.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000
            cfg['evaluation']['interval']=320000#8000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 872:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            opt='adamw_diff_play'
            schedule = 'poly10warm_play'
            #opt = 'adamw_diff'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_freezing_withInstanceSimpleNeck_play.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfg['runner']['max_iters']=320000#8000
            cfg['evaluation']['interval']=320000#8000
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 911:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'

            opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00911/work_dirs/local-exp00911/230921_1645_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_cc75d/iter_8000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 876:
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'

            #opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'

            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000
            cfg['evaluation']['interval']=320000
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00911/work_dirs/local-exp00911/230921_1645_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_cc75d/iter_8000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 896:
        #breakpoint()
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'

            opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

                
            cfg = config_from_vars()
            cfg['runner']['max_iters']=320000#8000
            cfg['evaluation']['interval']=320000#8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_simpleFPN.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00896/work_dirs/local-exp00896/231012_0141_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_3fc8e/iter_120000.pth'
            #breakpoint()
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 871:
        #breakpoint()
        udas = [
            'dacs'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'
            batch_size = 1
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'
            #opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

                
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000#8000
            cfg['evaluation']['interval']=8000#8000
            cfg['checkpoint_config']['interval']=8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_simpleFPN.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_acc.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00871/work_dirs/local-exp00871/231105_1747_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c94fd/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00896/work_dirs/local-exp00896/231012_0141_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_3fc8e/iter_120000.pth'
            #breakpoint()
            #breakpoint()
            
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 840:
        #breakpoint()
        udas = [
            'dacs'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'
            batch_size = 1
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'
            #opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

                
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000#8000
            cfg['evaluation']['interval']=40000#8000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_simpleFPN.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_acc.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00871/work_dirs/local-exp00871/231105_1747_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c94fd/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00896/work_dirs/local-exp00896/231012_0141_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_3fc8e/iter_120000.pth'
            #breakpoint()
            #breakpoint()
            
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 839:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_clipVitVisionSpatial.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 838:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_clipResnetVisionSpatial.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clip.py') for s in cfg['_base_']]
            cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00838/work_dirs/local-exp00838/231108_0901_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1ce88/iter_40000.pth'
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 837:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_textRefineContext.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clip.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 836:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw_text'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_ContextTextRefineContext.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clip.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 835:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devils_dacsJustThings.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedSim.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00835/work_dirs/local-exp00835/231114_1505_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_33c96/iter_40000.pth'
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 834:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devils_dacsJustThings.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 833:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 832:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text_200.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 831:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text_300.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 830:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text_800.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 829:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_200.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 828:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_800.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 827:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_inMixed.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 826:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_inMixed_checkSize.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 825:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_inMixed_checkSize_fewer.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 824:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_inMixed_checkSize_fewer.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 823:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_inMixed_checkSize_fewer_noOverlap.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 822:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil6.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #changing 852
    elif id == 821:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #changing 852
    elif id == 820:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 814:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 806:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 804:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_simil.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 803:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text_contexual.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 802:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text_crossEntropy.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 801:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_rpn_clip.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    elif id == 800:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_meanPool.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 799:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_simil_meanPool.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    elif id == 798:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_meanPool_contrastive.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()    
    elif id == 797:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_meanPool_mask.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 796:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_simil_meanPool_mask.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    elif id == 795:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_meanPool_contrastive_mask.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 

    elif id == 793:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_meanPool_contrastive_weight.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()  
    elif id == 794:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst4_noCheckSize_clip_meanPool_contrastive_mask_weight.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    elif id == 792:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    elif id == 791:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    elif id == 790:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th07.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()  
    elif id == 789:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()  
    #864 real1         
    elif id == 778:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ff23e/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ff23e/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    # 864 real 2 
    elif id == 777:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_235d3/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00777/work_dirs/local-exp00777/231221_2315_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7b2a5/iter_50000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_235d3/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #real 3 
    elif id == 776:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_aa00c/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_aa00c/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    #real 4
    elif id == 775:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
   #continuing 864 with instances
    elif id == 774:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base) 

   #continuing 864 with instances without aug
    elif id == 767:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base) 
    #continuing 1007 1
    elif id == 773:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ef056/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ef056/'
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #continuing 1007 2
    elif id == 772:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_037cc/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_037cc/'
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #continuing 1007 3
    elif id == 771:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_8c3ba/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_8c3ba/'
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #continuing 1007 4
    elif id == 770:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_67928/iter_40000.pth'
            #breakpoint()
            
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231212_2217_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_67928/'
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 805:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text4.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #853 changing
    elif id == 819:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00853/work_dirs/local-exp00853/231101_1341_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_33c92/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #changing 852
    elif id == 818:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst8.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #changing 852
    elif id == 817:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst7.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 816:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text2.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 815:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_text3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_textEmbedL2.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 841:
        #breakpoint()
        udas = [
            'dacs'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'
            batch_size = 1
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'
            #opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

                
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000#8000
            cfg['evaluation']['interval']=8000#8000
            cfg['checkpoint_config']['interval']=8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_simpleFPN_AP.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_acc.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00896/work_dirs/local-exp00896/231012_0141_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_3fc8e/iter_120000.pth'
            cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00841/work_dirs/local-exp00841/231105_1954_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_09a2b/iter_8000.pth'
            #breakpoint()
            #breakpoint()
            
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 842:
        #breakpoint()
        udas = [
            'dacs'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'
            batch_size = 1
            opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'
            #opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            #schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

                
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000#8000
            cfg['evaluation']['interval']=40000#8000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_simpleFPN.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_acc.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00992/work_dirs/local-exp00992/230921_1128_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_927e3/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00896/work_dirs/local-exp00896/231012_0141_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_3fc8e/iter_120000.pth'
            #breakpoint()
            #breakpoint()
            
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 890:
        #breakpoint()
        udas = [
            'source-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            #eval_metric_list = ['mIoU']
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #opt = 'adamw_diff_unfrozen'
            #schedule = 'poly10warm'

            opt = 'adamw_resnet_diff_2'#'adamw_diff_unfrozen_2'#changed from Both to _2 for 925
            schedule = 'poly10warm_resnet_diff'#'poly10warm_diff'#changed from nothing to _diff for 925

                
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000#8000
            cfg['evaluation']['interval']=8000#8000
            cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00890/work_dirs/local-exp00890/231005_1646_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_diff_s0_8a34e/iter_8000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen_withInstance_simpleFPN_noProp.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #diffusion backbone training only semantic without freezing:target only
    elif id == 986:
        udas = [
            'target-only'
        ]
        
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']
            opt = 'adamw_diff_unfrozen'
            schedule = 'poly10warm'

            cfg = config_from_vars()

            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_diff_OnlySemantic_unfrozen.py') for s in cfg['_base_']]
            #cfg[paspp_mitb5_diff_OnlySemantic_unfrozen'_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training only instance
    elif id == 987:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mAP']
            opt = 'adamw'
            schedule = 'poly10warm'

            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_onlyInst.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training only semantic
    elif id == 999:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']
            opt = 'adamw'
            schedule = 'poly10warm'

            cfg = config_from_vars()
            
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_OnlySemantic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_nophoto.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training only semantic
    elif id == 998:
        udas = [
            'target-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_OnlySemantic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training 
    elif id == 997:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #we dont need this
    elif id == 859:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training 
    elif id == 1007:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training 
    elif id == 769:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 768:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    #setting all negatives to weight 0 in the losses
    elif id == 766:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_posOnly.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    #using confidence scores
    elif id == 765:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_confScores.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target_scores.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    elif id == 764:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target_noFilter.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    #real 4
    elif id == 763:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_confScores.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_withScore.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #real 4
    elif id == 762:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noFilter.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #real 4
    elif id == 761:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_confScores.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_withScore_noFilter.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned aug 1
    elif id == 760:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_83bdc/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_83bdc/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned aug 2
    elif id == 759:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2312_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7945c/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2312_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7945c/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned aug 3
    elif id == 758:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2312_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d8fe6/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2312_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d8fe6/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned aug 4
    elif id == 757:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2312_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d23fb/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/231221_2312_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d23fb/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned aug 5
    elif id == 756:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/240104_1506_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7d1ba/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/240104_1506_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7d1ba/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned aug 6
    elif id == 755:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/240104_1506_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_40651/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00774/work_dirs/local-exp00774/240104_1506_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_40651/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #source to target pasting
    elif id == 754:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target_with_source.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)

    #both aligned  1
    elif id == 753:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00767/work_dirs/local-exp00767/231228_1448_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_3dd2a/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00767/work_dirs/local-exp00767/231228_1448_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_3dd2a/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned  2
    elif id == 752:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00767/work_dirs/local-exp00767/231228_1448_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1ff55/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00767/work_dirs/local-exp00767/231228_1448_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1ff55/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #both aligned  3
    elif id == 751:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00767/work_dirs/local-exp00767/231228_1448_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_32529/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00767/work_dirs/local-exp00767/231228_1448_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_32529/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT_norm_noCond.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #real 4
    elif id == 750:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00750/work_dirs/local-exp00750/240117_1543_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_4f128/iter_50000.pth'
            #breakpoint()
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #like 864 but for synthia to mapillary
    elif id == 749:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00749/work_dirs/local-exp00749/240121_0031_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_4eab3/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to mapillary
    elif id == 748:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00748/work_dirs/local-exp00748/240121_0024_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_0efd6/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy
    elif id == 747:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'


            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00747/work_dirs/local-exp00747/240121_0048_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ea053/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for synthia to mapillary continuing for 10k with instance mixing
    elif id == 733:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'


            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00749/work_dirs/local-exp00749/240121_0031_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_4eab3/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00749/work_dirs/local-exp00749/240121_0031_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_4eab3/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to mapillary continuing for 10k with instance mixing
    elif id == 732:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00748/work_dirs/local-exp00748/240121_0024_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_0efd6/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00748/work_dirs/local-exp00748/240121_0024_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_0efd6/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy continuing for 10k with instance mixing
    elif id == 731:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'


            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00747/work_dirs/local-exp00747/240121_0048_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ea053/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00747/work_dirs/local-exp00747/240121_0048_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ea053/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #filter 0 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 730:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #40k
            #cfg['checkpoint_path']="/PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth'
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00730/work_dirs/local-exp00730/240210_2230_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a25cf/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
        #filter 0 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 729:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00729/work_dirs/local-exp00729/240210_2229_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_7b7ad/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
        #filter 0 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 728:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00728/work_dirs/local-exp00728/240210_2229_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_3e633/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.25 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 727:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00727/work_dirs/local-exp00727/240210_2229_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_5515f/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.25 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 726:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00726/work_dirs/local-exp00726/240211_0034_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_c298f/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.25 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 725:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00725/work_dirs/local-exp00725/240211_0046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_2f6cc/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.5 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 724:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00724/work_dirs/local-exp00724/240211_0053_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_74f2a/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.5 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 723:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00723/work_dirs/local-exp00723/240211_0101_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_7567f/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.5 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 722:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth'
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00722/work_dirs/local-exp00722/240211_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_7217f/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.75 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 721:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth'
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00721/work_dirs/local-exp00721/240211_0200_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_2953b/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.75 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 720:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00720/work_dirs/local-exp00720/240211_0241_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_178a9/iter_50000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.75 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 719:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            #breakpoint()
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth'
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00719/work_dirs/local-exp00719/240211_0417_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_8b55e/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 1.0 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 718:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter1.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 1.0 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 717:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter1.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 1.0 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 716:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter1.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.8 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 715:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.8 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 714:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.8 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 713:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.9 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 712:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.9 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 711:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.9 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 710:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #filter 0.8 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 709:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a1e01/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a1e01/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
    #filter 0.8 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 708:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
    #filter 0.8 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 707:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
    #filter 0.9 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 706:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a1e01/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a1e01/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
    #filter 0.9 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 705:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
    #filter 0.9 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 704:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
    #continuing with semantic going from target to source
    elif id == 703:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete_semanticMixing.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()

    #mixing for 741 semantic+edaps(no FD)
    elif id == 702:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD)
    elif id == 701:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 743 semantic+edaps(no FD)
    elif id == 700:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="/PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 738 semantic+edaps(FD included)
    elif id == 699:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 739 semantic+edaps(FD included)
    elif id == 698:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 740 semantic+edaps(FD included)
    elif id == 697:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 744 vanilla edaps
    elif id == 696:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #mixing for 745 vanilla edaps
    elif id == 695:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)
    #mixing for 746 vanilla edaps
    elif id == 694:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfgs.append(cfg)

    #mixing for 741 semantic+edaps(no FD) 0.75
    elif id == 693:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #just removed for submission
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD) 0.75
    elif id == 692:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 743 semantic+edaps(no FD) 0.75
    elif id == 691:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 741 semantic+edaps(no FD) 0.5
    elif id == 690:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD) 0.5
    elif id == 689:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 743 semantic+edaps(no FD) 0.5
    elif id == 688:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 741 semantic+edaps(no FD) 0.25
    elif id == 687:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD) 0.25
    elif id == 686:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 743 semantic+edaps(no FD) 0.25
    elif id == 685:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 741 semantic+edaps(no FD) 0
    elif id == 684:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD) 0
    elif id == 683:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 743 semantic+edaps(no FD) 0
    elif id == 682:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)


    #mixing for 738 semantic+edaps(FD included) 0.75
    elif id == 681:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 739 semantic+edaps(FD included) 0.75
    elif id == 680:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 740 semantic+edaps(FD included) 0.75
    elif id == 679:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)

    #mixing for 738 semantic+edaps(FD included) 0.5
    elif id == 678:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 739 semantic+edaps(FD included) 0.5
    elif id == 677:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 740 semantic+edaps(FD included) 0.5
    elif id == 676:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)


    #mixing for 738 semantic+edaps(FD included) 0.25
    elif id == 675:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 739 semantic+edaps(FD included) 0.25
    elif id == 674:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 740 semantic+edaps(FD included) 0.25
    elif id == 673:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)


    #mixing for 738 semantic+edaps(FD included) 0.0
    elif id == 672:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 739 semantic+edaps(FD included) 0.0
    elif id == 671:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 740 semantic+edaps(FD included) 0.0
    elif id == 670:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)


    #mixing for 742 semantic+edaps(no FD) 0
    elif id == 669:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD) 025
    elif id == 668:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD) 05
    elif id == 667:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 742 semantic+edaps(no FD) 075
    elif id == 666:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 741 semantic+edaps(no FD) 0
    elif id == 665:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter0.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 741 semantic+edaps(no FD) 025
    elif id == 664:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter025.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 741 semantic+edaps(no FD) 05
    elif id == 663:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter05.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #mixing for 741 semantic+edaps(no FD) 075
    elif id == 662:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)


    #source to target pasting
    elif id == 661:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            #breakpoint()
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target_with_source.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #source to target pasting
    elif id == 660:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            #breakpoint()
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target_with_source.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #source to target pasting
    elif id == 659:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            #breakpoint()
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudo_target_with_source.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)

    #like 864 but for synthia to mapillary
    elif id == 658:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for synthia to mapillary
    elif id == 657:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for synthia to mapillary
    elif id == 656:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    #like 864 but for cityscape to mapillary
    elif id == 655:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to mapillary
    elif id == 654:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to mapillary
    elif id == 653:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00653/work_dirs/local-exp00653/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_80d90/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy
    elif id == 652:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy
    elif id == 651:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy
    elif id == 650:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    #like 864 but for synthia to mapillary
    elif id == 649:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00658/work_dirs/local-exp00658/240216_1615_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ba670/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00658/work_dirs/local-exp00658/240216_1615_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ba670/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for synthia to mapillary
    elif id == 648:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00657/work_dirs/local-exp00657/240216_1612_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_751a6/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00657/work_dirs/local-exp00657/240216_1612_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_751a6/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for synthia to mapillary
    elif id == 647:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00656/work_dirs/local-exp00656/240216_1611_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c8fee/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00656/work_dirs/local-exp00656/240216_1611_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c8fee/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    #like 864 but for cityscape to mapillary
    elif id == 646:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00655/work_dirs/local-exp00655/240216_1558_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_345e0/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00655/work_dirs/local-exp00655/240216_1558_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_345e0/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to mapillary
    elif id == 645:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00654/work_dirs/local-exp00654/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_5326c/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00654/work_dirs/local-exp00654/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_5326c/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to mapillary
    elif id == 644:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00653/work_dirs/local-exp00653/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_80d90/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00653/work_dirs/local-exp00653/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_80d90/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter09.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy
    elif id == 643:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00652/work_dirs/local-exp00652/240216_1555_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f2fca/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00652/work_dirs/local-exp00652/240216_1555_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f2fca/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy
    elif id == 642:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00651/work_dirs/local-exp00651/240216_1554_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_5c845/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00651/work_dirs/local-exp00651/240216_1554_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_5c845/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 864 but for cityscape to cityscape_foggy
    elif id == 641:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00650/work_dirs/local-exp00650/240216_1553_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_f67ca/iter_40000.pth'
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00650/work_dirs/local-exp00650/240216_1553_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_f67ca/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter08.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)


    #like 653 but for cityscape to mapillary
    elif id == 640:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path'] = 'PATH_TO_FILL/edaps_experiments/exp-00653/work_dirs/local-exp00653/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_80d90/iter_40000.pth'
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_saveAcc.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    #like 864 but for cityscape to mapillary
    elif id == 639:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00653/work_dirs/local-exp00653/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_80d90/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_saveAcc.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #mixing for 743 semantic+edaps(no FD) 0.75
    elif id == 638:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50400
            #breakpoint()
            cfg['evaluation']['interval']=50400
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete_new2.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)

    #continuing with semantic going from target to source 741
    elif id == 637:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_noDelete_semanticMixing.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #continuing with semantic going from target to source 742
    elif id == 636:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_noDelete_semanticMixing.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #continuing with semantic going from target to source 743
    elif id == 635:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_noDelete_semanticMixing.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #vanilla for synthia to mapillary
    elif id == 634:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']= "PATH_TO_FILL/edaps_experiments/exp-00634/work_dirs/local-exp00634/240310_1227_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_19a75/iter_50000.pth"
            cfg['resumeElham']= "PATH_TO_FILL/edaps_experiments/exp-00634/work_dirs/local-exp00634/240310_1227_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_19a75/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #vanilla for synthia to mapillary
    elif id == 633:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00633/work_dirs/local-exp00633/240310_1227_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_178ac/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00633/work_dirs/local-exp00633/240310_1227_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_178ac/iter_50000.pth"
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #vanilla for synthia to mapillary
    elif id == 632:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00632/work_dirs/local-exp00632/240310_1227_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_cdbf6/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00632/work_dirs/local-exp00632/240310_1227_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_cdbf6/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    #vanilla for cityscape to mapillary
    elif id == 631:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00631/work_dirs/local-exp00631/240310_1227_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e3ac0/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00631/work_dirs/local-exp00631/240310_1227_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e3ac0/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #vanilla for cityscape to mapillary
    elif id == 630:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00630/work_dirs/local-exp00630/240310_1227_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_43c7d/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00630/work_dirs/local-exp00630/240310_1227_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_43c7d/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #vanilla for cityscape to mapillary
    elif id == 629:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):

            data_root_val = 'data/mapillary'
            ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            target = 'mapillary'
            source='cityscapes'
            #num_samples_debug = 13
            gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
            gt_dir_panoptic = 'data/mapillary'

            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00629/work_dirs/local-exp00629/240310_1227_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_0748f/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00629/work_dirs/local-exp00629/240310_1227_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_0748f/iter_50000.pth"
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00653/work_dirs/local-exp00653/240216_1557_cs2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_80d90/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_mapillary_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #vanilla for cityscape to cityscape_foggy
    elif id == 628:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00628/work_dirs/local-exp00628/240310_1227_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_029de/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00628/work_dirs/local-exp00628/240310_1227_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_029de/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #vanilla for cityscape to cityscape_foggy
    elif id == 627:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00627/work_dirs/local-exp00627/240310_1227_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_6119c/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00627/work_dirs/local-exp00627/240310_1227_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_6119c/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #vanilla for cityscape to cityscape_foggy
    elif id == 626:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            source = 'cityscapes'
            target = 'cityscapesFoggy'
            #seeds = [0, 1, 2]
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            #ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val' if debug else 'gtFine_panoptic/cityscapes_panoptic_val'
            
            data_root_val = 'data/cityscapesFoggy'
            #ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
            #num_samples_debug = 13
            gt_dir_instance = './data/cityscapesFoggy/gtFine/val'
            gt_dir_panoptic = './data/cityscapesFoggy/gtFine_panoptic'
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00626/work_dirs/local-exp00626/240310_1226_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_fe7b4/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00626/work_dirs/local-exp00626/240310_1226_cs2csFoggy_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_fe7b4/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_cityscapes_to_cityscapesFoggy_maskrcnn_panoptic_beta_0.02.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    #original backbone training with no FD
    elif id == 625:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training with no FD
    elif id == 624:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training with no FD
    elif id == 623:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 622:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_50000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #original backbone training 
    elif id == 621:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth"
            #744 2
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0516_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_f1024/iter_40000.pth"
            #744 1
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #filter 0.75 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 620:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=1000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00625/work_dirs/local-exp00625/240301_1122_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_33df9/iter_40000.pth"
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00625/work_dirs/local-exp00625/240301_1122_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_33df9/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #filter 0.75 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 619:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=1000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00624/work_dirs/local-exp00624/240301_1257_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_4005c/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00624/work_dirs/local-exp00624/240301_1257_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_4005c/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    #filter 0.75 source to target 0: to do: Fill out the checkpoint that it needs to start from
    elif id == 618:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=1000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)

    # 625 original backbone training with no FD
    elif id == 617:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00625/work_dirs/local-exp00625/240301_1122_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_33df9/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #743
    elif id == 616:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a1e01/iter_40000.pth"
            #743 2
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0210_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_9d196/iter_40000.pth"
            #743 1
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #mixing for 743 semantic+edaps(no FD) 0.75
    elif id == 615:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_40000.pth"
            cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    elif id == 614:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    elif id == 613:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    elif id == 612:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            #cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00623/work_dirs/local-exp00623/240301_1401_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e259f/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
    # 691
    elif id == 611:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_50000.pth"
            #cfg['resumeElham']="PATH_TO_FILL/edaps_experiments/exp-00691/work_dirs/local-exp00691/240213_1311_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_c2ab2/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo_filter075.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_pseudoInstance_occlusion_thParam_noDelete.py') for s in cfg['_base_']]
            #cfgs.append(cfg)
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)


    #864
    elif id == 741:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            #741 2
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            #741 1
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0109_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_44d97/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)



    #original backbone training 
    elif id == 746:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240122_1445_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_39108/iter_40000.pth"
            #746 2
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240128_0856_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_0804d/iter_40000.pth"
            #746 1
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00746/work_dirs/local-exp00746/240128_0729_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_3e39f/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training 
    elif id == 745:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_40000.pth"
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00745/work_dirs/local-exp00745/240122_1443_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_312f1/iter_50000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone training 
    elif id == 744:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=60000
            cfg['evaluation']['interval']=60000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240122_1442_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_afe8f/iter_40000.pth"
            #744 2
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0516_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_f1024/iter_40000.pth"
            #744 1
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00744/work_dirs/local-exp00744/240128_0328_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_92f04/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #864
    elif id == 743:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a1e01/iter_40000.pth"
            #743 2
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0210_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_9d196/iter_40000.pth"
            #743 1
            #removed for submission
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00743/work_dirs/local-exp00743/240128_0145_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_d6ee1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #864
    elif id == 742:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_f6211/iter_40000.pth"
            #742 2
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0138_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_48a98/iter_40000.pth"
            #742 1
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00742/work_dirs/local-exp00742/240128_0133_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_89df1/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #864
    elif id == 741:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240122_2046_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_4c846/iter_40000.pth"
            #741 2
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0111_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_e3625/iter_40000.pth"
            #741 1
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00741/work_dirs/local-exp00741/240128_0109_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_44d97/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #864 with imagenet align
    elif id == 740:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_990de/iter_40000.pth"
            #740 2
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0108_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_50b5e/iter_40000.pth"
            #740 1
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00740/work_dirs/local-exp00740/240128_0107_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1cdcb/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #864 with imagenet align
    elif id == 739:
        udas = [
            'dacs'
        ]
        seeds = [1]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth"
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00739/work_dirs/local-exp00739/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_fa229/iter_40000.pth'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #864 with imagenet align
    elif id == 738:
        udas = [
            'dacs'
        ]
        seeds = [2]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240122_2059_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_70ba2/iter_40000.pth"
            #738 2
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0106_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_c300e/iter_40000.pth"
            #738 1
            #cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00738/work_dirs/local-exp00738/240128_0101_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_dee7c/iter_40000.pth"
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #semantic align and instance mixing from the beginning
    elif id == 737:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_nocaDelete.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #semantic align and instance mixing from the beginning with also the image align
    elif id == 736:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #instance mixing from the beginning with also the image align
    elif id == 735:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #semantic align and instance mixing from the beginning with also the image align
    elif id == 734:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=10000
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            #cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_noDelete.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
  
    #real 4
    elif id == 775:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()

    #864 real1         
    elif id == 778:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ff23e/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ff23e/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    # 864 real 2 
    elif id == 777:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_235d3/iter_40000.pth'
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00777/work_dirs/local-exp00777/231221_2315_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7b2a5/iter_50000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_235d3/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #real 3 
    elif id == 776:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_aa00c/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_aa00c/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    #real 4
    elif id == 775:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231219_1247_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f93ac/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #like 1007 but fd is off
    elif id == 860:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 869:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00869/work_dirs/local-exp00869/231027_1126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_25e79/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 868:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil2.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 867:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 866:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]

            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00866/work_dirs/local-exp00866/231027_1627_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_401c4/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 865:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_udaClip.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_clip.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clip.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 855:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_udaClipText.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 854:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00854/work_dirs/local-exp00854/231101_1642_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7730d/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 853:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst2.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00853/work_dirs/local-exp00853/231101_1341_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_33c92/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 852:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 788:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_noCheck_avg.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 787:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst_noCheck_max.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 786:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    elif id == 785:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_old.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
    #continuing 864 with instances
    elif id == 784:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #continuing 864 with instances
    elif id == 781:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_withInstances_sameVIT.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 780:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_differentSizesPlaces.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint() 
    elif id == 779:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=50000
            #breakpoint()
            cfg['evaluation']['interval']=50000
            cfg['checkpoint_config']['interval']=50000
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/iter_40000.pth'
            #breakpoint()
            cfg['resumeElham']='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f0866/'
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3_pseudo.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_pseudoInstance_occlusion_th08_differentSizesPlaces_rolling.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00852/work_dirs/local-exp00852/231101_0620_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_01eb8/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()   
    elif id == 851:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst2.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 848:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_inst2.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 847:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU', 'mPQ', 'mAP']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=8000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101_unfrozen.py') for s in cfg['_base_']] #
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 844:
        import os
        udas = [
            #'source-only'
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            eval_metric_list = ['mIoU']
            #eval_metric_list = ['mAP']
            opt = 'adamw_resnet_unfrozenLikeDiff'
            schedule = 'poly10warm'
            #opt = 'adamw_diff_unfrozen_hyper'
            #schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            os.environ['TRANSFORMERS_CACHE'] = 'PATH_TO_FILL/cache'
            #checkpoint_path='PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230921_1648_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_71069/iter_8000.pth'#'PATH_TO_FILL/edaps_experiments/exp-00918/work_dirs/local-exp00918/230919_0723_syn2cs_source-only_maskrcnn_mitb5_poly10warm_resnet_s0_c3c73'#iteration 10000
            #cfg['checkpoint_path'] = checkpoint_path
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_resnetPure101_unfrozen_justSem.py') for s in cfg['_base_']] #
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/default_runtime_mmdet_mr.py', '_base_/default_runtime.py') for s in cfg['_base_']]
            #breakpoint()
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 850:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_poda.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_resnet.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_clipText.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 867 but for dacs
    elif id == 864:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 813:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=40000
            cfg['checkpoint_path']="PATH_TO_FILL/edaps_experiments/exp-00813/work_dirs/local-exp00813/240117_0119_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_af1f9/iter_40000.pth"
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_aug.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 812:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_sample.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_afee9/iter_40000.pth'#'PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 811:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_afee9/iter_40000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 810:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_teacher.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00864/work_dirs/local-exp00864/231121_2343_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_afee9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 809:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pos50percent.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 808:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pos75percent.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    elif id == 807:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_pos90percent.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231116_0124_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_733b9/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231101_1605_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_1df77/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 869: not running for some nan problem
    elif id == 863:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #breakpoint()
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00869/work_dirs/local-exp00869/231027_1126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_25e79/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 867 but for dacs
    elif id == 862:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil3.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 869 dacs
    elif id == 861:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #breakpoint()
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00869/work_dirs/local-exp00869/231027_1126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_25e79/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 866 but for dacs
    elif id == 857:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=8000
            #breakpoint()
            cfg['evaluation']['interval']=8000
            cfg['checkpoint_config']['interval']=8000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00867/work_dirs/local-exp00867/231027_1154_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b7aad/iter_8000.pth'
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 866: not running for some nan problem
    elif id == 856:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #breakpoint()
            cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00856/work_dirs/local-exp00856/231102_0758_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_55742/iterSave/iter_20000_.pth'
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00869/work_dirs/local-exp00869/231027_1126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_25e79/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #like 866: not running for some nan problem
    elif id == 849:
        udas = [
            'dacs'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            opt = 'adamw'
            schedule = 'poly10warm'
            cfg = config_from_vars()
            cfg['runner']['max_iters']=40000
            #breakpoint()
            cfg['evaluation']['interval']=40000
            cfg['checkpoint_config']['interval']=10000
            #breakpoint()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5_devil4_copy.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/target-only_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/target-only_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/uda_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            cfg['uda']['imnet_feature_dist_lambda']=0.0
            #cfg['_base_'] = [s.replace('_base_/uda/dacs_a999_fdthings.py', '_base_/uda/dacs_a999_fdthings_fdoff.py') for s in cfg['_base_']]
            #cfg['uda']['imnet_feature_dist_lambda']=0.0
            #breakpoint()
            #cfg['checkpoint_path']='PATH_TO_FILL/edaps_experiments/exp-00869/work_dirs/local-exp00869/231027_1126_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_25e79/iter_8000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-00997/work_dirs/local-exp00997/230923_1825_syn2cs_source-only_maskrcnn_mitb5_poly10warm_s0_b76ea/iter_40000.pth'
            #cfg['checkpoint_path'] ='PATH_TO_FILL/edaps_experiments/exp-01007/work_dirs/local-exp01007/231013_0644_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_e6156/iter_80000.pth'
            cfgs.append(cfg)
            #breakpoint()
            #breakpoint()
            for cfg_base in cfg['_base_']:
                print(cfg_base)
    #original backbone for training on the backbone local
    elif id == 1001:
        udas = [
            'source-only'
        ]
        seeds = [0]
        for seed, uda in  itertools.product(seeds, udas):
            cfg = config_from_vars()
            cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
            cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
            #cfg['evaluation']['gt_dir']=cfg['evaluation']['gt_dir'].replace('..','.')
            #cfg['evaluation']['gt_dir_panop']=cfg['evaluation']['gt_dir_panop'].replace('PATH_TO_FILL','.')
            #cfg['data']['val']['ann_dir']=cfg['data']['val']['ann_dir'].replace('PATH_TO_FILL','.')
            #cfg['data']['val']['data_root']=cfg['data']['val']['data_root'].replace('PATH_TO_FILL','.')
            cfg['exp_root']=cfg['exp_root'].replace('..','.')
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)
        
    #original backbone validating suman's checkpoint and validating on cityscape
    elif id == 1002:
        batch_size = 1
        workers_per_gpu = 0
        #generate_only_visuals_without_eval = False
        #evaluate_from_saved_png_predictions=True
        #dump_visuals_during_eval = True
        #workers_per_gpu = 0
        #dump_semantic_pred_as_numpy_array = True
        #generate_only_visuals_without_eval = True
        #dump_visuals_during_eval = True
        dump_predictions_to_disk = False#True
        evaluate_from_saved_numpy_predictions = True
        #panop_eval_temp_folder_previous = 'edaps_experiments/exp-00008/work_dirs/local-exp00008/230416_2234_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_63b2d/panoptic_eval/panop_eval_16-04-2023_22-34-34-937426'
        panop_eval_temp_folder_previous = 'edaps_experiments/exp-01002/work_dirs/local-exp01002/230808_1221_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_ce05b'#/panoptic_eval/panop_eval_08-08-2023_09-44-25-807959'
        panop_eval_temp_folder_previous = './edaps_experiments/exp-01003/work_dirs/local-exp01003/230813_2004_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_cf8a2'
        checkpoint_path = './pretrained_edaps/pretrained_edaps_weights/'
        cfg = config_from_vars()
        cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
        cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
        #cfg['evaluation']['gt_dir']=cfg['evaluation']['gt_dir'].replace('PATH_TO_FILL','.')
        #cfg['evaluation']['gt_dir_panop']=cfg['evaluation']['gt_dir_panop'].replace('PATH_TO_FILL','.')
        #cfg['data']['val']['ann_dir']=cfg['data']['val']['ann_dir'].replace('PATH_TO_FILL','.')
        #cfg['data']['val']['data_root']=cfg['data']['val']['data_root'].replace('PATH_TO_FILL','.')
        #cfg['exp_root']=cfg['exp_root'].replace('PATH_TO_FILL','.')
        cfg['checkpoint_path'] = checkpoint_path
        #breakpoint()
        cfgs.append(cfg)
    #validating by saving numpy and no evaluating
    elif id == 1003:
        batch_size = 1
        workers_per_gpu = 0
        #generate_only_visuals_without_eval = True
        #evaluate_from_saved_png_predictions=True
        #dump_visuals_during_eval = True
        #workers_per_gpu = 0
        #dump_semantic_pred_as_numpy_array = True
        #generate_only_visuals_without_eval = True
        #dump_visuals_during_eval = True
        dump_predictions_to_disk = True#True
        evaluate_from_saved_numpy_predictions = False
        #panop_eval_temp_folder_previous = 'edaps_experiments/exp-00008/work_dirs/local-exp00008/230416_2234_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_63b2d/panoptic_eval/panop_eval_16-04-2023_22-34-34-937426'
        #panop_eval_temp_folder_previous = 'edaps_experiments/exp-01002/work_dirs/local-exp01002/230808_0944_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_008a8/panoptic_eval/panop_eval_08-08-2023_09-44-25-807959'
        checkpoint_path = './pretrained_edaps/pretrained_edaps_weights/'
        cfg = config_from_vars()
        cfg['_base_'] = [s.replace('_base_/models/maskrcnn_sepaspp_mitb5_diff.py', '_base_/models/maskrcnn_sepaspp_mitb5.py') for s in cfg['_base_']]
        cfg['_base_'] = [s.replace('_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic_diff.py', '_base_/datasets/source-only_synthia_to_cityscapes_maskrcnn_panoptic.py') for s in cfg['_base_']]
        #cfg['evaluation']['gt_dir']=cfg['evaluation']['gt_dir'].replace('PATH_TO_FILL','.')
        #cfg['evaluation']['gt_dir_panop']=cfg['evaluation']['gt_dir_panop'].replace('PATH_TO_FILL','.')
        #cfg['data']['val']['ann_dir']=cfg['data']['val']['ann_dir'].replace('PATH_TO_FILL','.')
        #cfg['data']['val']['data_root']=cfg['data']['val']['data_root'].replace('PATH_TO_FILL','.')
        #cfg['exp_root']=cfg['exp_root'].replace('PATH_TO_FILL','.')
        cfg['checkpoint_path'] = checkpoint_path
        #breakpoint()
        cfgs.append(cfg)
    elif id == 1004:
        batch_size = 1
        workers_per_gpu = 0

    # -------------------------------------------------------------------------
    # M-Net: SYNTHIA → Cityscapes (Table 5)
    # M-Net training and evaluation are done in 4 stages:
    # Stage-1: Train the semantic segmentation network (id=50)
    # Stage-2: Train the instance segmentation network (id=51)
    # Stage-3: Extract the semantic segmentation predictions (id=52)
    # Stage-4: Extract the instance segmentation predictions and
    # evaluate the M-Net (id=53)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Stage-1: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Train the semantic segmentation network
    # -------------------------------------------------------------------------
    elif id == 50:
        seeds = [0, 1, 2]
        loss_weight_semanitc, loss_weight_instance = 1.0, 0.0
        for seed in seeds:
            cfg = config_from_vars()
            cfg = set_semantic_and_instance_loss_weights(cfg, loss_weight_semanitc, loss_weight_instance)
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # Stage-2: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Train the instance segmentation network
    # -------------------------------------------------------------------------
    elif id == 51:
        seeds = [0, 1, 2]
        loss_weight_semanitc, loss_weight_instance = 0.0, 1.0
        for seed in seeds:
            cfg = config_from_vars()
            cfg = set_semantic_and_instance_loss_weights(cfg, loss_weight_semanitc, loss_weight_instance)
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # Stage-3: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Utilize the semantic segmentation network that has been trained in expid=50
    # to generate predictions for semantic segmentation,
    # and save the predictions as a numpy array to the disk.
    # -------------------------------------------------------------------------
    elif id == 52:
        batch_size = 1
        workers_per_gpu = 0
        dump_semantic_pred_as_numpy_array = True
        eval_metric_list = ['mIoU']
        # Put here the checkpoint locations of the semantic segmentation network that has been trained in expid=50
        # An example is given below:
        semantic_model_checkpoint_locations = [
            'path/to/the/trained/semantic/segmentation/network/model1',
            'path/to/the/trained/semantic/segmentation/network/model2',
            'path/to/the/trained/semantic/segmentation/network/model3'
        ]
        # An example:
        # semantic_model_checkpoint_locations = [
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_322b3',
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s1_6eb04',
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s2_22080',
        #     ]
        for cl in semantic_model_checkpoint_locations:
            cfg = config_from_vars()
            cfg['checkpoint_path'] = cl
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # Stage-4: M-Net: SYNTHIA → Cityscapes (Table 5)
    # Evaluate M-Net model
    # -------------------------------------------------------------------------
    elif id == 53:
        batch_size = 1
        workers_per_gpu = 0
        load_semantic_pred_as_numpy_array = True
        # Set the paths for the instance segmentation model which has been trained in expid=51
        instance_model_checkpoint_locations = [
            'path/to/the/trained/instance/segmentation/network/model1',
            'path/to/the/trained/instance/segmentation/network/model2',
            'path/to/the/trained/instance/segmentation/network/model3'
        ]
        # Set the paths for the saved smenaitc segmentation predictions
        semantic_pred_numpy_array_location_list = [
            'path/to/the/semanitc/segmentation/predictions/numpy/files/model1',
            'path/to/the/semanitc/segmentation/predictions/numpy/files/model2',
            'path/to/the/semanitc/segmentation/predictions/numpy/files/model3'
        ]
        # Examples are given below:
        # instance_model_checkpoint_locations = [
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_a5398',
        # ]
        # semantic_pred_numpy_array_location_list = [
        #     '/<experiment-root-folder>/'
        #     'local-exp4022/221104_2333_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_322b3/'
        #     'panoptic_eval/panop_eval_09-11-2022_15-52-53-341753/semantic'
        # ]
        for imcl, semantic_pred_numpy_array_location in zip(instance_model_checkpoint_locations, semantic_pred_numpy_array_location_list):
            cfg = config_from_vars()
            cfg['checkpoint_path'] = imcl
            cfgs.append(cfg)
            for cfg_base in cfg['_base_']:
                print(cfg_base)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes:
    # Evaluate EDAPS Model
    # -------------------------------------------------------------------------
    elif id == 6:
        seed = 0 # set the seed value accordingly to the seed value used to the train the model
        batch_size = 1
        workers_per_gpu = 0
        checkpoint_path = 'path/to/the/latest/checkpoint'
        # e.g., checkpoint_path = 'edaps_experiments/exp-00001/work_dirs/euler-exp00001/230412_1715_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_7f2ba'
        cfg = config_from_vars()
        cfg['checkpoint_path'] = checkpoint_path
        cfgs.append(cfg)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Cityscapes :
    # generate visualization without evaluation
    # This for the demo, just download the pretrained EDAPS model
    # save it to pretrained_edaps/
    # and run inference on the Cityscapes validation set
    # The predictions will be saved to disk
    # -------------------------------------------------------------------------
    elif id == 7:
        #breakpoint()
        batch_size = 1
        workers_per_gpu = 0
        generate_only_visuals_without_eval = True
        dump_visuals_during_eval = True
        checkpoint_path = 'path/to/the/latest/checkpoint'
        cfg = config_from_vars()
        cfg['checkpoint_path'] = checkpoint_path
        cfgs.append(cfg)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Mapillary:
    # Evaluate EDAPS Model
    # -------------------------------------------------------------------------
    elif id == 8:
        seed = 0  # set the seed value accordingly to the seed value used to the train the model
        total_train_time = '4:00:00' # TODO: remove this before final release
        batch_size = 1
        workers_per_gpu = 0
        data_root_val = 'data/mapillary'
        ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
        target = 'mapillary'
        num_samples_debug = 13
        gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
        gt_dir_panoptic = 'data/mapillary'
        # checkpoint_path = 'path/to/the/latest/checkpoint'
        checkpoint_path = 'PATH_TO_FILL/experiments/daformer_panoptic_experiments/euler-exp00002/230413_0956_syn2mapillary_dacs_rcs001_maskrcnn_mitb5_poly10warm_s0_0bf2e' # TODO: remove before final relase
        # checkpoint_path = 'PATH_TO_FILL/DATADISK2/apps/experiments/edaps_experiments/euler-exp00002/230413_0956_syn2mapillary_dacs_rcs001_maskrcnn_mitb5_poly10warm_s0_0bf2e' # TODO: remove before final relase
        cfg = config_from_vars()
        cfg['checkpoint_path'] = checkpoint_path
        cfgs.append(cfg)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Mapillary:
    # Evaluate EDAPS Model
    # -------------------------------------------------------------------------
    elif id == 9:
        dump_predictions_to_disk = True
        evaluate_from_saved_numpy_predictions = True  # this loads the model predictions saved as numpy files and then run evaluation,
        # since there are 2000 mapillary val images, loading all the 2000 predictions on RAM requires a large RAM memory
        # so I first save the predictions to disk as numpy files and then load them for evaluation

        seed = 0  # set the seed value accordingly to the seed value used to the train the model
        total_train_time = '4:00:00'  # TODO: remove this before final release
        batch_size = 1
        workers_per_gpu = 0
        data_root_val = 'data/mapillary'
        ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
        target = 'mapillary'
        num_samples_debug = 13
        gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
        gt_dir_panoptic = 'data/mapillary'
        # checkpoint_path = 'path/to/the/latest/checkpoint'
        # checkpoint_path = 'PATH_TO_FILL/experiments/daformer_panoptic_experiments/euler-exp00002/230413_0956_syn2mapillary_dacs_rcs001_maskrcnn_mitb5_poly10warm_s0_0bf2e' # TODO: remove before final relase
        checkpoint_path = 'PATH_TO_FILL/DATADISK2/apps/experiments/edaps_experiments/euler-exp00002/230413_0956_syn2mapillary_dacs_rcs001_maskrcnn_mitb5_poly10warm_s0_0bf2e'  # TODO: remove before final relase

        # 'PATH_TO_FILL/DATADISK2/apps/experiments/edaps_experiments/euler-exp00008/230415_1541_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_2e6ba/panoptic_eval/panop_eval_15-04-2023_15-41-14-693044'
        cfg = config_from_vars()
        cfg['checkpoint_path'] = checkpoint_path
        cfgs.append(cfg)

    # -------------------------------------------------------------------------
    # EDAPS (M-Dec-TD) : SYNTHIA → Mapillary:
    # Evaluate EDAPS Model
    # -------------------------------------------------------------------------
    elif id == 10:
        evaluate_from_saved_png_predictions = True
        panop_eval_temp_folder_previous = 'edaps_experiments/exp-00008/work_dirs/local-exp00008/230416_2234_syn2mapillary_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_63b2d/panoptic_eval/panop_eval_16-04-2023_22-34-34-937426'
        seed = 0  # set the seed value accordingly to the seed value used to the train the model
        total_train_time = '4:00:00'  # TODO: remove this before final release
        batch_size = 1
        workers_per_gpu = 0
        data_root_val = 'data/mapillary'
        ann_dir = 'val_panoptic_19cls_debug' if debug else 'val_panoptic_19cls'
        target = 'mapillary'
        num_samples_debug = 13
        gt_dir_instance = 'data/mapillary/val_panoptic_19cls'
        gt_dir_panoptic = 'data/mapillary'
        cfg = config_from_vars()
        cfg['checkpoint_path'] = None # since all the predictions are saved as PNG files we dont need to run inference again
        cfgs.append(cfg)


    # --- RETURNING CFGS ---
    return cfgs
