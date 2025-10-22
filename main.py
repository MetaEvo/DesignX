import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['RAY_num_server_call_thread'] = '1'
import json
import torch
import pprint
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
from config import get_options
import numpy as np
from REINFORCE import REINFORCE
from utils.utils import set_seed
# DummyVectorEnv for Windows or Linux, SubprocVectorEnv for Linux
from env import DummyVectorEnv,SubprocVectorEnv
import platform, ray
from components.operators import *
from components.Population import *
from nets.actor_network import ConfigX
from configx_cfg import get_cfg_options


def load_agent(name):
    agent = {
        'REINFORCE': REINFORCE,
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent

def run(opts):
    # only one mode can be specified in one time, test or train
    assert opts.train==None or opts.test==None, 'Between train&test, only one mode can be given in one time'
    
    sys=platform.system()
    opts.is_linux=True if sys == 'Linux' else False
    torch.multiprocessing.set_sharing_strategy('file_system')
    # figure out the max_fes(max function evaluation times), in our experiment, we use 20w for 10D problem and 100w for 30D problem

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed to initialize the network
    set_seed(opts.seed)
    torch.autograd.set_detect_anomaly(True)

    # Set the device, you can change it according to your actual situation
    # opts.device = torch.device("cuda:1")
    # opts.device = torch.device("cpu")
    if opts.feature_extractor == 'ELA':
        opts.state_dim = 9
           
    # Figure out the RL algorithm
    # if opts.is_linux:
    agent = REINFORCE(opts,SubprocVectorEnv)
    # else:
    #     agent = PPO(opts,DummyVectorEnv)

    # Load data from load_path(if provided)
    if opts.load_name is not None:
        # opts.run_name = opts.load_name
        load_path = os.path.join(opts.load_path, opts.load_name)
        if opts.load_epoch is None:
            epoch_list = os.listdir(load_path)
            id_list = []
            for eid in epoch_list:
                id_list.append(int(eid[6:-3]))
            opts.load_epoch = np.max(id_list)
        
        load_path = os.path.join(load_path, f'epoch-{opts.load_epoch}.pt')
        agent.load(load_path)

    # Do validation only
    if opts.test:
        # Testing
        from utils.make_dataset import make_synthetic_dataset
        from rollout import rollout
        if opts.load_name is not None:
            opts.run_name = opts.load_name
        opts.log_dir = os.path.join('rollout_outputs', opts.run_name)
        if not os.path.exists(opts.log_dir):
            os.makedirs(opts.log_dir)
            
        cfg_opt = get_cfg_options()
        # cfg_opt.device = 'cpu'
        cfg_opt.save_dir = os.path.join(opts.save_dir, opts.run_name, 'ConfigX_models')
        cfg_opt.syn_train = opts.syn_train
        cfgX = ConfigX(cfg_opt)
        cfg_opt.load_path = "" + cfg_opt.load_path
        if cfg_opt.load_name is not None:
            load_path = os.path.join(cfg_opt.load_path, cfg_opt.load_name)
            load_path = os.path.join(load_path, f'epoch-{cfg_opt.load_epoch}.pt')
            cfgX.load(load_path)

        # Load the validation datasets
        _, test_dataloader=make_synthetic_dataset()

        set_seed(opts.testseed)
        # test_problem,opts,agent,cfgX, tb_logger,epoch_id=epoch
        rew, avg_best, curve, srg_info=rollout(test_dataloader,opts, agent, cfgX, None, opts.problem, 101)
        np.save(f"curve-{opts.load_name}_{opts.load_epoch}-{cfg_opt.load_name}_{cfg_opt.load_epoch}-{opts.problem}-{opts.repeat}-{opts.skip_FEs}.npy", curve, allow_pickle=True)
        np.save(f"gbest-{opts.load_name}_{opts.load_epoch}-{cfg_opt.load_name}_{cfg_opt.load_epoch}-{opts.problem}-{opts.repeat}-{opts.skip_FEs}.npy", avg_best, allow_pickle=True)
        
    else:  
        # configure tensorboard
        path = os.path.join(opts.log_dir, opts.run_name)
        if not os.path.exists(path):
            os.makedirs(path)
        tb_logger = SummaryWriter(path)
        
        set_seed(opts.seed)
        # Start the actual training loop
        agent.start_training(tb_logger)

import logging
if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ray.init(_temp_dir="/public3/home/gong_group/guohongshu/ray_tmp/", num_cpus=64, num_gpus=2)#logging_level=100, 
    warnings.filterwarnings("ignore")
    # main process
    run(get_options())

    