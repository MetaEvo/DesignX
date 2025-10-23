import os
import time
import argparse


def get_cfg_options(args=None):

    parser = argparse.ArgumentParser(description="Configuration options for Agent-2 (ConfigX)")

    parser.add_argument('--train',default=None,action='store_true',help='switch to train mode')
    parser.add_argument('--test',default=None,action='store_true', help='switch to inference mode')
    parser.add_argument('--single_task', type=int, default=None, help='switch to inference mode')
    parser.add_argument('--problem', default ='bbob')  # for test only
    parser.add_argument('--task', default ='usual')  # for test only
    parser.add_argument('--task_class', default ='ALL')
    parser.add_argument('--alg_abla_mod', default=False, type=bool,)
    parser.add_argument('--alg_abla_op', default=False, type=bool,)
    parser.add_argument('--rew_train', default=0, type=int,)
    parser.add_argument('--alg_sample', default='random')
    parser.add_argument('--clip_tail', default=True, type=bool,)

    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--epoch_end', type=int, default=1, help='maximum training epoch')
    parser.add_argument('--weighted_start', type=int, default=None, help='start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--encoder', default='attn', choices=['attn', 'mlp'])
    parser.add_argument('--positional', default='sin', help="normalization type, 'learnt' (default) or 'sin' or None")
    parser.add_argument('--morphological', default=True, type=bool,)
    parser.add_argument('--aligned_pe', default=False, type=bool,)
    parser.add_argument('--sep_state', default=True, type=bool,)
    parser.add_argument('--shuffle', default=False, type=bool,)
    parser.add_argument('--dim', default=None, type=int,)
    parser.add_argument('--skip_step', default=1, type=int,)
    parser.add_argument('--softmax', default=False, type=bool,)
    parser.add_argument('--train_cfx', type=bool, default=False, help="")

    # Overall settingss
    parser.add_argument('--seed', type=int, default=1, help='random seed to use')
    parser.add_argument('--dataseed', type=int, default=14, help='random seed for dataset generation 28 for trainsize32, 2024 for 64')
    parser.add_argument('--traindata_seed', type=int, default=1, help='random seed for dataset generation 28 for trainsize32, 2024 for 64')
    parser.add_argument('--testdata_seed', type=int, default=14, help='random seed for dataset generation 28 for trainsize32, 2024 for 64')
    parser.add_argument('--testseed', type=int, default=2025, help='random seed for dataset generation')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--store_path', default='outputs/')
    parser.add_argument('--trainsize', type=int, default=1024)
    parser.add_argument('--testsize', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64,help='number of instances per batch during training')
    parser.add_argument('--test_batch_size', type=int, default=64,help='number of instances per batch during training')
    parser.add_argument('--update_best_model_epochs',type=int,default=1,help='update the best model every n epoch')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--pool_path', default = None, help='run name %Y%m%dT%H%M%S')
    parser.add_argument('--syn_train', default=False, type=bool,)
    
    # Testing settings
    parser.add_argument('--load_path', default = 'ConfigX_model/', help='path to load model parameters and optimizer state from')
    parser.add_argument('--load_name', default = None, help='run name %Y%m%dT%H%M%S')
    parser.add_argument('--load_epoch', default = None, help='epoch id')
    
    parser.add_argument('--smac_load', default = None, help='smac load')
    parser.add_argument('--train_task', default = None,action='store_true', )
    parser.add_argument('--test_task', default = None,action='store_true', )

    # Task settings
    parser.add_argument('--Npop', default=[1, 2, 3, 4])
    parser.add_argument('--Npop_prob', default=[0.4, 0.2, 0.2, 0.2])
    parser.add_argument('--opm', default=[1, 2, 3])
    parser.add_argument('--NPmax', default=[50, 100, 200])
    parser.add_argument('--NPmin', default=[4, 10, 20])
    parser.add_argument('--NA', default=[1, 2, 3])
    parser.add_argument('--Xmax', default=5)
    parser.add_argument('--Vmax', default=[0.2, 0.3, 0.5])
    parser.add_argument('--regroup', default=[0,  5, 10])
    parser.add_argument('--maxAct', default=12)
    parser.add_argument('--maxCom', default=32)
    parser.add_argument('--actSpace', default=10)
    parser.add_argument('--MaxGen', default=500)

    # agent settings
    parser.add_argument('--encoder_head_num', type=int, default=4, help='head number of encoder')
    parser.add_argument('--critic_head_num', type=int, default=4, help='head number of critic encoder')
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of input embeddings')
    parser.add_argument('--decoder_hidden_dim', type=int, default=32, help='head number of decoder')
    parser.add_argument('--hidden_dim', type=int, default=16, help='dimension of hidden layers in Enc/Dec16')
    parser.add_argument('--hidden_dim1_critic',default=32,help='the first hidden layer dimension for critic32')
    parser.add_argument('--hidden_dim2_critic',default=16,help='the second hidden layer dimension for critic16')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='number of stacked layers in the encoder')
    parser.add_argument('--node_dim',default=9,type=int,help='feature dimension for backbone algorithm')
    parser.add_argument('--op_dim',default=16,type=int,help='feature dimension for backbone algorithm')
    parser.add_argument('--op_embed_dim',default=16,type=int,help='feature dimension for backbone algorithm')
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")

    parser.add_argument('--lr_model', type=float, default=1e-4, help="learning rate for the actor network")  # Todo: -> 1e-5
    parser.add_argument('--lr_decay', type=float, default=1., help='learning rate decay per epoch',)  # 0.912
    parser.add_argument('--max_learning_step',default=4000000,help='the maximum learning step for training')

    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor for future rewards0.999')
    parser.add_argument('--T_train', type=int, default=1800, help='number of itrations for training')
    parser.add_argument('--n_step', type=int, default=10, help='n_step for return estimation')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='maximum L2 norm for gradient clipping')
    parser.add_argument('--max_sigma',default=0.7,type=float,help='upper bound for actor output sigma')
    parser.add_argument('--min_sigma',default=0.1,type=float,help='lowwer bound for actor output sigma0.01')
    parser.add_argument('--show_figs', action='store_true', help='enable figure logging')

    parser.add_argument('--log_dir', default='cfgx_logs', help='directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=50, help='log info every log_step gradient steps')
    parser.add_argument('--save_dir', default='cfgx_outputs', help='directory to write output models to')
    
    opts = parser.parse_args(args)
    opts.run_name = time.strftime("%Y%m%dT%H%M%S")
    # opts.run_name = "{}_{}".format(opts.run_name, opts.run_time)

    return opts