# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import mmcv
import torch
import wandb
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmseg.datasets import build_dataset
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from tools.parser_argument_help_str import paht1, paht2, wm1, wm2
import sys
from mmdet.models.builder import build_train_model
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='(Deprecated, please use --gpu-id) number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='(Deprecated, please use --gpu-id) ids of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff-seed', action='store_true', help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction, help=paht1)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help=paht2)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto-scale-lr', action='store_true', help='enable automatically scaling LR.')
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot be both specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    return args


def main(args):
    args = parse_args(args)
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    #breakpoint()
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.train_cfg.work_dir = cfg.work_dir
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support single GPU mode in non-distributed training. Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(wm2)
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    cfg.device = get_device()
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: ' f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]


    #import wandb
    #wandb.login(key='', relogin=True)
    #breakpoint()



    #breakpoint()
    datasets = [build_dataset(cfg.data.train)]
    #breakpoint()
    #model.class_names = list(datasets[0].CLASSES)
    if 'DenseCLIP' in cfg.model.type:
        cfg.model.class_names = list(datasets[0].CLASSES)
    model = build_train_model(cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    #breakpoint()
    if 'checkpoint_path' in cfg:
        checkpoint_path = cfg.checkpoint_path
        logger.info('The following checkpoints will be evaluated ...')
        logger.info(checkpoint_path)
        # generating checkpoint file path

        checkpoint_file_path = checkpoint_path#os.path.join(checkpoint_path, 'iter_10000.pth')#'latest.pth') #'iter_5400.pth')
        #breakpoint()
        logger.info(f'Evaluation will be done for the model {checkpoint_file_path}')
        #breakpoint()

        checkpoint = load_checkpoint(model, checkpoint_file_path, map_location='cpu')
        
        #breakpoint()
    
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
    model.CLASSES = datasets[0].CLASSES
    #breakpoint()
    #cfg.model.class_names = list(datasets[0].CLASSES)
    #model.class_names = list(datasets[0].CLASSES)

    #breakpoint()
    #cfg['model']['train_cfg']['rpn']['assigner']['neg_iou_thr']
    
    #cfg['log_config']['hooks'][1]['init_kwargs']['project']='try1'


    #wandb.agent(sweep_id, function=main, count=10)
    #breakpoint()


    def main1():
        default_config={
            "pos_rpn": 0.5,
            "neg_rpn":0.5,
            "min_rpn":0.5,
            "pos_rcnn":0.5,
            "neg_rcnn":0.5,
            "min_rcnn":0.5,
            "dir":'PATH_TO_FILL', 
            #'project':'elhamam/sweep-984/'
            
      }
        import wandb
        wandb.login(key='', relogin=True)
        wandb.init(config=default_config)
        config = wandb.config
        #breakpoint()
        #breakpoint()
        cfg['model']['train_cfg']['rpn']['assigner']['neg_iou_thr']=config.neg_rpn
        cfg['model']['train_cfg']['rpn']['assigner']['pos_iou_thr']=config.pos_rpn
        cfg['model']['train_cfg']['rpn']['assigner']['min_pos_iou']=config.min_rpn

        cfg['model']['train_cfg']['rpn']['assigner']['neg_iou_thr']=config.neg_rcnn
        cfg['model']['train_cfg']['rpn']['assigner']['pos_iou_thr']=config.pos_rcnn
        cfg['model']['train_cfg']['rpn']['assigner']['min_pos_iou']=config.min_rcnn
        train_detector(model, datasets, cfg, distributed=distributed, validate=(not cfg.only_train), timestamp=timestamp,  meta=meta,)
    #'
    def main2():
        default_config={
            "lr": 0.00008,
            "lr_mult":0.1,
            "decay_mult":0,
            'weight_decay':0,
            "dir":'PATH_TO_FILL', 
            #'project':'elhamam/sweep-984/'
            
      }
        import wandb
        wandb.login(key='', relogin=True)
        wandb.init(config=default_config)
        config = wandb.config
        #breakpoint()
        #breakpoint()
        cfg['optimizer']['lr']=config.lr
        cfg['optimizer']['paramwise_cfg']['custom_keys']['unet']['decay_mult']=config.decay_mult
        cfg['optimizer']['paramwise_cfg']['custom_keys']['unet']['lr_mult']=config.lr_mult
        cfg.optimizer.weight_decay=config.weight_decay
        #breakpoint()
        train_detector(model, datasets, cfg, distributed=distributed, validate=(not cfg.only_train), timestamp=timestamp,  meta=meta,)
    def main3():
        default_config={
            "loss_rpn_bbox": 1.0,
            "loss_rpn_cls": 1.0,
            "loss_bbox": 1.0,
            "loss_cls": 1.0,
            "loss_mask": 1.0,
            "dir":'PATH_TO_FILL', 
            #'project':'elhamam/sweep-984/'
            
      }
        import wandb
        wandb.login(key='', relogin=True)
        wandb.init(config=default_config)
        config = wandb.config
        #breakpoint()
        #breakpoint()
 
        cfg['model']['rpn_head']['loss_bbox']['loss_weight']=config.loss_rpn_bbox
        cfg['model']['rpn_head']['loss_cls']['loss_weight']=config.loss_rpn_cls

        cfg['model']['roi_head']['bbox_head']['loss_bbox']['loss_weight']=config.loss_bbox
        cfg['model']['roi_head']['bbox_head']['loss_cls']['loss_weight']=config.loss_cls

        cfg['model']['roi_head']['mask_head']['loss_mask']['loss_weight']=config.loss_mask


        #breakpoint()
        train_detector(model, datasets, cfg, distributed=distributed, validate=(not cfg.only_train), timestamp=timestamp,  meta=meta,)
    #u1eqnrdh'
    #breakpoint()
    sweepSamplingIsOn=False
    sweepLrOptIsOn=False
    wandbIsOn=False
    oldSweep=False
    sweepLossesOn=False
    if(str(cfg.exp)=='925'):
        sweepSamplingIsOn=False
        sweepLrOptIsOn=False
        wandbIsOn=True
        oldSweep=True
        sweepLossesOn=True
    if(str(cfg.exp)=='4' or str(cfg.exp)=='936' or str(cfg.exp)=='937' or str(cfg.exp)=='924' or str(cfg.exp)=='923' or str(cfg.exp)=='912' or str(cfg.exp)=='913' or str(cfg.exp)=='908'  or str(cfg.exp)=='907'  or str(cfg.exp)=='906' or  str(cfg.exp)=='898' or  str(cfg.exp)=='897' or  str(cfg.exp)=='896' or  str(cfg.exp)=='895' or  str(cfg.exp)=='894' or  str(cfg.exp)=='883'  or  str(cfg.exp)=='886' or  str(cfg.exp)=='893') :    
        sweepSamplingIsOn=False
        sweepLrOptIsOn=False
        wandbIsOn=True
        oldSweep=False
        sweepLossesOn=False
    if(str(cfg.exp)=='975' or str(cfg.exp)=='976'):
        sweepSamplingIsOn=False
        sweepLrOptIsOn=False
        wandbIsOn=False
        oldSweep=False
    if(str(cfg.exp)=='922'):
        sweepSamplingIsOn=False
        sweepLrOptIsOn=True
        wandbIsOn=True
        oldSweep=True
        sweepLossesOn=False

    #elif(str(cfg.exp)=='971' or str(cfg.exp)=='972'): #or str(cfg.exp)=='968' or str(cfg.exp)=='967'):
    #    sweepSamplingIsOn=False
    #    sweepLrOptIsOn=False
    #    wandbIsOn=True
    #    oldSweep=False

    if(sweepSamplingIsOn):


        sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'loss'},
        'parameters': 
        {
            "pos_rpn": {'max': 1., 'min': 0.5},
            "neg_rpn": {'max': 0.5, 'min': 0.},
            "min_rpn": {'max': 0.5, 'min': 0.},
            "pos_rcnn": {'max': 1., 'min': 0.5},
            "neg_rcnn": {'max': 0.5, 'min': 0.},
            "min_rcnn": {'max': 0.5, 'min': 0.},
        }
        }
        
        #sweep_id = wandb.sweep(
        #sweep=sweep_configuration, 
        #project='sweep-984-ver3',
        #)
        #breakpoint()
        sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='sweep-'+str(cfg.exp),
        )
        if(oldSweep):
            sweep_id='27k4xnco' 
        print('###############################the sweep id is: ', sweep_id,'#############################')
        #wandb.agent(sweep_id, main, count=1,project="sweep-984-ver3")
        wandb.agent(sweep_id, main1, count=1,project="sweep-"+str(cfg.exp))
    elif(sweepLrOptIsOn):
        #breakpoint()
        sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'loss'},
        'parameters': 
        {
            "lr": {'max': 0.00009, 'min': 0.00001},
            "lr_mult": {'max': 0.01, 'min': 0.00000001},
            "decay_mult": {'max': 0.01, 'min': 0.00000001},
            "weight_decay": {'max': 0.01, 'min': 5e-3},

        }
        }
        #sweep_id = wandb.sweep(
        #sweep=sweep_configuration, 
        #project='sweep-984-ver3',
        #)
        #breakpoint()
        sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='-sweep-'+str(cfg.exp),
        )
        if(oldSweep):
            if(str(cfg.exp)=='975'):
                sweep_id='1ehyph4f' 
            elif(str(cfg.exp)=='976'):
                sweep_id='7so9e26t'
            elif(str(cfg.exp)=='922'):
                sweep_id='t7svf64q'
        print('###############################the sweep id is: ', sweep_id,'#############################')
        #wandb.agent(sweep_id, main, count=1,project="sweep-984-ver3")
        #breakpoint()
        wandb.agent(sweep_id, function=main2, count=1, project="-sweep-"+str(cfg.exp))
    elif(sweepLossesOn):
        #breakpoint()
        sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'loss'},
        'parameters': 
        {
            "loss_rpn_bbox": {'max': 1.0, 'min': 0.000},
            "loss_rpn_cls": {'max': 1.0, 'min': 0.000},
            "loss_bbox": {'max': 1.0, 'min': 0.000},
            "loss_cls": {'max': 1.0, 'min': 0.000},
            "loss_mask": {'max': 1.0, 'min': 0.000},

        }
        }
        #sweep_id = wandb.sweep(
        #sweep=sweep_configuration, 
        #project='sweep-984-ver3',
        #)
        #breakpoint()
        sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='-sweep-'+str(cfg.exp),
        )
        if(oldSweep):
            if(str(cfg.exp)=='925'):
                sweep_id='paxl6ylp' 

        print('###############################the sweep id is: ', sweep_id,'#############################')
        #wandb.agent(sweep_id, main, count=1,project="sweep-984-ver3")
        #breakpoint()
        wandb.agent(sweep_id, function=main3, count=1, project="-sweep-"+str(cfg.exp))
    elif(wandbIsOn):
        wandb.login(key='', relogin=True)
        wandb.init(project='sweep-'+str(cfg.exp))
        train_detector(model, datasets, cfg, distributed=distributed, validate=(not cfg.only_train), timestamp=timestamp,  meta=meta,)
    else:
        train_detector(model, datasets, cfg, distributed=distributed, validate=(not cfg.only_train), timestamp=timestamp,  meta=meta,)




    
    #wandb.init(project='my-first-sweep')
    #breakpoint()
    

if __name__ == '__main__':
    main(sys.argv[1:])
