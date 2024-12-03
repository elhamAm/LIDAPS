# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------------

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
import torch
from mmcv import Config, get_git_hash
from tools import train_mmdet
from tools import train
from tools import test_mmdet
from tools import test
from tools.panoptic_deeplab.utils import create_panop_eval_folders
from experiments import generate_experiment_cfgs
from experiments_bottomup import generate_experiment_cfgs as generate_experiment_cfgs_bottomup
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.autograd.set_detect_anomaly(True)


def run_command(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)

if __name__ == '__main__':

    config_file = None
    expId = None
    machine_name = None
    JOB_DIR = 'jobs'
    WORKDIR = 'work_dirs'
    GEN_CONFIG_DIR = 'configs/generated'
    #GEN_CONFIG_DIR_evals = ''

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp', type=int, default=expId, help='Experiment id as defined in experiment.py')
    group.add_argument('--config', default=config_file, help='Path to config file', )
    parser.add_argument('--machine', type=str, default=machine_name, help='Name of the machine')
    parser.add_argument('--local_rank', type=str, default=machine_name, help='Name of the machine')
    parser.add_argument('--exp_root', type=str, default=None, help='Root folder to save all EDAPS experimental results')
    parser.add_argument('--exp_sub', type=str, default=None, help='sub folder to save experimental results benong to a spefic experiment Id')
    args = parser.parse_args()
    assert (args.config is None) != (args.exp is None), 'Either config or exp has to be defined.'
    print('here1')
    cfgs, config_files = [], []
    # Training with Predefined Config
    if args.config is not None:
        print('here2')
        print(f'training with predefined config : {args.config}')
        cfg = Config.fromfile(args.config)
        # Specify Name and Work Directory
        exp_name = f'{args.machine}-exp{cfg["exp"]:05d}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        # setting paths for panoptic evaluation
        panop_eval_folder = os.path.join(cfg['exp_root'], cfg['exp_sub'], WORKDIR, exp_name, unique_name, 'panoptic_eval')
        panop_eval_temp_folder = create_panop_eval_folders(panop_eval_folder)
        panop_eval_outdir = os.path.join(panop_eval_temp_folder, 'visuals')
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join(cfg['exp_root'], cfg['exp_sub'], WORKDIR, exp_name, unique_name),
            'git_rev': get_git_hash(),
            'evaluation': {
                            'panop_eval_folder': panop_eval_folder,
                            'panop_eval_temp_folder': panop_eval_temp_folder,
                            'debug': cfg['debug'],
                            'out_dir': panop_eval_outdir,
                            },
        }
        
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        #breakpoint()
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        #print('here4')
        if args.exp in [100, 101, 102, 105, 106,107,108,109,110]:
            # Experiments belongs to M-Dec-BU or S-Net models (Table 5 in the main paper).
            cfgs = generate_experiment_cfgs_bottomup(args.exp, args.machine)
        elif args.exp in [1, 2, 3, 4, 50, 51, 52, 53, 6, 7, 8, 9, 10,1000, 1001, 1002, 1003, 999, 998, 997,996,995, 994, 993, 992, 991, 990, 989, 988, 987, 986, 985, 984,983,982,981,980,979, 978, 977, 976, 975, 974, 973, 972, 971, 970, 969, 968, 967, 966, 965, 964, 963, 962, 961, 960, 959, 958, 957, 956, 955, 954, 953, 952, 951, 950, 949, 948, 947, 946,945,944, 943, 942, 932,933,934,935,936,937,938,939,940,941, 931, 930, 929, 928, 927, 926, 925, 924, 923, 922, 921,920,9592,9232,919, 918, 917, 916, 915, 914, 913, 912, 911,910,909,908,907,906,905, 904, 903, 902, 901,899,900,898,897,896,895,894,893,892,891,890,889,888,887,886,885,884,883,882,881,880,879,878,877,876,875,874,873,872,871,870,1007,869,868,867,866,865,864,863,858,860,859,861,862,857,856,855,854,853,852,851,850,849,848,847,846,845,844,843,841,842,840,839,838, 837, 836, 835, 834, 833, 832,830,831,829,828,827,826, 825, 824, 823, 822, 821, 820, 819, 818, 817, 816, 815, 814, 813,812, 811, 810, 809, 808, 807, 806, 805, 804, 803, 802, 801, 800, 799,798,797,796, 795, 794, 793, 792, 791, 790,789, 788, 787, 786, 785, 784, 783, 782, 781, 780, 779, 778,777,776,775, 774, 773, 772,771,770, 769, 768, 767, 766, 765, 764, 763, 762, 761, 760, 759, 758, 757, 756, 755, 754, 753, 752, 751, 750, 749, 748, 747, 746, 745, 744, 743, 742, 741, 740, 739, 738, 737, 736, 735, 734, 733, 732, 731, 730, 729, 728, 727,726,725,724,723,722,721,720,719,718,717,716, 715,714,713,712,711,710,709,708,707,706,705,704, 703, 702, 701, 700, 699, 698, 697, 696, 695, 694, 693, 692, 691, 690, 689, 688, 687, 686, 685, 684, 683, 682, 681, 680, 679, 678, 677, 676, 675, 674, 673, 672, 671, 670, 669, 668, 667, 666, 665, 664, 663, 662, 661, 660, 659, 658, 657, 656, 655, 654, 653, 652, 651, 650, 649, 648, 647, 646, 645, 644, 643, 642, 641, 640, 639, 638, 637, 636, 635, 634, 633, 632, 631, 630, 629, 628, 627, 626, 625, 624, 623, 622, 621, 620, 619, 618, 617, 616, 615, 614, 613, 612, 611]:
            # Experiments belongs to EDAPS (M-Dec-TD) mdoels.
            cfgs = generate_experiment_cfgs(args.exp, args.machine)
        else:
            raise NotImplementedError(f"Do not find implementation for experiment id : {args.exp} !!")
        for i, cfg in enumerate(cfgs):
            machine = cfg['machine']
            exp_name = '{}-exp{:05d}'.format(cfg['machine'], args.exp)
            cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_{cfg["name"]}_{str(uuid.uuid4())[:5]}'
            cfg['work_dir'] = os.path.join(cfg['exp_root'], cfg['exp_sub'], WORKDIR, exp_name, cfg['name'])
            print(cfg['work_dir'])
            cfg['git_rev'] = get_git_hash()
            cfg['_base_'] = ['../../' + e for e in cfg['_base_']]
            # set configs for panoptic evaluation
            panop_eval_folder = os.path.join(cfg['work_dir'], 'panoptic_eval')
            panop_eval_temp_folder = create_panop_eval_folders(panop_eval_folder)
            cfg['evaluation']['panop_eval_folder'] = panop_eval_folder
            cfg['evaluation']['panop_eval_temp_folder'] = panop_eval_temp_folder
            cfg['evaluation']['debug']=cfg['debug']
            cfg['evaluation']['out_dir'] = os.path.join(panop_eval_temp_folder, 'visuals')
            # write cfg to json file
            cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
            os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
            assert not os.path.isfile(cfg_out_file)
            #breakpoint()
            with open(cfg_out_file, 'w') as of:
                json.dump(cfg, of, indent=4)
            config_files.append(cfg_out_file)

    if args.machine == 'local':
        #breakpoint()
        for i, cfg in enumerate(cfgs):
            #breakpoint()
            print('Run job {}'.format(cfg['name']))
            if cfg['exp'] in [ 101, 106,105,109,108]:
                train.main([config_files[i]])
                #breakpoint()
            elif cfg['exp'] in [1, 2, 3,  50, 51, 1000, 1001, 999, 998, 996, 993,  991, 990, 988, 987, 986,984,983,982,981,980,979, 978, 977, 976, 975, 972, 971, 968, 967, 965, 964, 963, 962, 960, 961, 959, 957, 956, 955, 954, 953, 952, 951, 950, 949, 948, 947,946,945,944, 941,940,939,938,937,936,935,934,933,992,932,931,930, 928, 927, 926, 925, 924, 922,4, 921, 920, 919, 917, 916, 913, 910,909,908,907,906,905,904,903,902,901,899,900,898,911,895,894,897,893,892,891,889,888,887,890,884, 883,879,878,876,896,886,875,874,872,870,868,997,865,864,863,873,858,860,859,861,862,857,867,869,866,855,852,853,851,850,849,848,847,882,846,845,844,843,842,840,839,837,836,838,834,833,832,831,830,829,828,827,826, 825, 824, 823,822,821,820,819, 818, 817, 816, 815, 814,  812, 810, 809, 808, 807, 806, 805,804, 803, 802, 801, 800, 799, 798, 797,796,795,794,793, 792, 1007, 791, 790, 789, 788, 787, 786, 785, 784, 781, 780, 779, 778,776,775, 774, 773, 772, 771,770, 769, 768, 767, 766, 765, 764, 763, 762, 761, 760, 759, 758, 757, 756, 755, 754, 753, 752, 751, 737, 736, 735, 734, 733, 732, 731,718,717,716, 715,714,713,712,711,710,709,708,707,706,705,704, 739, 703, 748, 702, 701, 700, 699, 698, 697, 696, 695, 694, 693, 692, 691, 690, 689, 688, 687, 686, 685, 684, 683, 682, 681, 680, 679, 678, 677, 676, 675, 674, 673, 672, 671, 670, 669, 668, 667, 666, 665, 664, 663, 662, 660, 659, 658, 657, 656, 655, 654, 652, 651, 650, 649, 648, 647, 646, 645, 644, 643, 642, 641, 653, 638, 637, 636, 635, 634, 633, 632, 631, 630, 629, 628, 627, 626, 625, 624, 623, 620, 619, 618, 614, 613, 612, 743, 741]:
                #breakpoint()
                train_mmdet.main([config_files[i]])
            elif cfg['exp'] in [52, 53, 6, 7, 8, 9, 10, 1002, 1003,995, 994, 989, 985, 974, 973, 970, 969, 966, 958, 956, 955, 954, 953, 952, 951, 950, 949, 948, 947, 943, 942,  929, 923, 9232,9592, 914, 915, 912, 918,  885,881,877,880,856,841,871,835, 811, 810, 777, 750, 813, 749, 747, 744, 745, 746, 738, 740, 742, 730, 729, 728, 727,726,725,724,723,722,721,720,719,661, 640, 639, 622, 621, 617, 616, 615, 611]:
                test_mmdet.main([config_files[i]])
            elif cfg['exp'] in [102,100,107,110]:
                test.main([config_files[i]])
            else:
                raise NotImplementedError(f"Do not find implementation for experiment id : {args.exp} !!")
            torch.cuda.empty_cache()
