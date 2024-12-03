# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Provide args as argument to main()
# - Snapshot source code
# - Build UDA model instead of regular one

import argparse
import copy
import os
import os.path as osp
import sys
import time
from argparse import Namespace

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models.builder import build_train_model
from mmseg.utils import collect_env, get_root_logger
from mmseg.utils.collect_env import gen_code_archive
from typing import Tuple

import torch
import torch.distributed as dist

import os

from torch import Tensor

import torch.multiprocessing as mp

def set_sharing_strategy(new_strategy=None):
    """
    https://pytorch.org/docs/stable/multiprocessing.html
    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    https://stackoverflow.com/questions/66426199/how-does-one-setup-the-set-sharing-strategy-strategy-for-multiprocessing-in-pyto
    """
    from sys import platform

    if new_strategy is not None:
        mp.set_sharing_strategy(new_strategy=new_strategy)
    else:
        if platform == 'darwin':  # OS X
            # only sharing strategy available at OS X
            mp.set_sharing_strategy('file_system')
        else:
            # ulimit -n 32767 or ulimit -n unlimited (perhaps later do try catch to execute this increase fd limit)
            mp.set_sharing_strategy('file_descriptor')

def use_file_system_sharing_strategy():
    """
    when to many file descriptor error happens

    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    """
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup_process(rank, world_size, port, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.

    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_DISABLE=1

    https://stackoverflow.com/questions/61075390/about-pytorch-nccl-error-unhandled-system-error-nccl-version-2-4-8

    https://pytorch.org/docs/stable/distributed.html#common-environment-variables
    """
    import torch.distributed as dist
    import os
    import torch

    if rank != -1:  # -1 rank indicates serial code
        print(f'setting up rank={rank} (with world_size={world_size})')
        # MASTER_ADDR = 'localhost'
        MASTER_ADDR = '127.0.0.1'
        # set up the master's ip address so this child process can coordinate
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        print(f"{MASTER_ADDR=}")
        os.environ['MASTER_PORT'] = port
        print(f"{port=}")

        # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
        if torch.cuda.is_available():
            # unsure if this is really needed
            # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
            # os.environ['NCCL_IB_DISABLE'] = '1'
            backend = 'nccl'
        print(f'{backend=}')
        # Initializes the default distributed process group, and this will also initialize the distributed package.
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        print(f'--> done setting up rank={rank}')

def cleanup(rank):
    """ Destroy a given process group, and deinitialize the distributed package """
    # only destroy the process distributed group if the code is not running serially
    if rank != -1:  # -1 rank indicates serial code
        dist.destroy_process_group()
def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    #breakpoint()
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)#"0"#str(0) #str(args.local_rank)
    #breakpoint()
    os.environ['RANK'] = "0"#[0,1]"#"1"#str(0)
    os.environ['WORLD_SIZE'] = "2"#str(2)
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '21061'
    #os.environ['NCCL_P2P_DISABLE']='1'
    return args


def main(args):
#def main():
    
    args = parse_args(args)

    #mydict={'config':'PATH_TO_FILL/my_edaps/configs/generated/local-exp00109/230927_1522_syn2cs_dacs_a999_fdthings_bottomup_rcs001_cpl_bottomup_mitb5_poly10warm_s42_f57b6.json', 'work_dir':None, 'load_from':None, 'resume_from':None, 'no_validate':False, 'gpus':None,'gpu_ids':None, 'seed':None, 'deterministic':False, 'options':None, 'launcher':'none', 'local_rank':0}
    #args = Namespace(**mydict)
    #args.launcher ='pytorch'

    cfg = Config.fromfile(args.config)

    #cfg['data']['samples_per_gpu']=1
    print(cfg)
    #breakpoint()
    #
    # setting the crop size and val set for cfg.debug mode
    cfg['crop_size'] = (512, 512) if cfg.debug else (512, 512)

    # if 'evaluation' in cfg:
    if 'dataset_name' in cfg['evaluation']:
        if cfg.debug and cfg['evaluation']['dataset_name'] == 'cityscapes':
            cfg['data']['val']['ann_dir'] = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
        elif cfg.debug and cfg['evaluation']['dataset_name'] == 'Mapillary':
            raise NotImplementedError('cfg.debug mode val set is not present for {} dataset!'.format(cfg['evaluation']['dataset_name']))
    #breakpoint()
    if 'activate_panoptic' in cfg:
        cfg['model']['decode_head']['debug'] = cfg.debug
        cfg['model']['activate_panoptic'] = cfg['activate_panoptic']
        cfg['model']['decode_head']['activate_panoptic'] = cfg['activate_panoptic']
    #
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.model.train_cfg.work_dir = cfg.work_dir
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    #add back
    #cfg.gpu_ids=[0,1]
    #args.gpus=2
    #args.gpu_ids=[0,1]


    #breakpoint()
    torch.cuda.device_count() 
    #breakpoint()
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    os.environ['OMPI_COMM_WORLD_RANK']='1'
    os.environ['OMPI_COMM_WORLD_LOCAL_RANK']='1'
    os.environ['OMPI_COMM_WORLD_SIZE']='2'
    #args.launcher='pytorch'
    #args.gpus=2
    #breakpoint()
    #args.gpu_ids=[0,1]
    #print(args)
    #print(args.gpu_ids)
    #print(args.launcher)
    #cfg.dist_params['devices']="0,1"
    #print('cfg.dist_params: ', cfg.dist_params)
    #cfg.dist_params['backend']='gloo'
    #dist_params = dict(backend='gloo')

    #from mmdet.utils import (build_ddp,setup_multi_processes)
    #setup_multi_processes(cfg)
    #distributed =True
    print(os.environ['WORLD_SIZE'])
    print(os.environ['RANK'])
    #breakpoint()
    
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True

        print('test_setup')
        #port = 29507#find_free_port()
        dist_params = dict(backend='gloo')
        #world_size = 2
        #rank=0
        #backend='gloo'
        #os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
        #os.environ['NCCL_IB_DISABLE'] = '1'
        #dist.init_process_group(backend, rank=rank, world_size=world_size)
        #mp.spawn(setup_process, args=(world_size, port), nprocs=1)
        #breakpoint()
        init_dist(args.launcher, **dist_params)
    #breakpoint()
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # snapshot source code
    gen_code_archive(cfg.work_dir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: 'f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]
    # build train model
    #breakpoint()
    model = build_train_model(cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    print('###############################')
    # build dataset
    print(cfg.data.train)
    print('###############################')
    print(cfg.data)
    print('################################')
    #breakpoint()
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)



    #world_size = 2
    #mp.spawn(example,
    #    args=(world_size,),
    #    nprocs=world_size,
    #    join=True)
    #breakpoint()
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main(sys.argv[1:])
    #main()
