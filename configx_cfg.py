import os
import time
import argparse


def get_cfg_options():

    parser = argparse.Namespace()

    parser.device='cpu'
    parser.positional='sin'

    # Overall settingss
    parser.seed=1
    parser.dataseed=14
    parser.traindata_seed=1
    parser.testdata_seed=14
    parser.testseed=2025
    parser.repeat=1
    parser.store_path='outputs/'
    parser.batch_size=64
    parser.test_batch_size=64
    
    # agent settings
    parser.encoder_head_num=4
    parser.critic_head_num=4
    parser.embedding_dim=64
    parser.decoder_hidden_dim=32
    parser.hidden_dim=16
    parser.hidden_dim1_critic=32
    parser.hidden_dim2_critic=16
    parser.n_encode_layers=3
    parser.node_dim=9
    parser.op_dim=16
    parser.op_embed_dim=16
    parser.normalization='layer'
    parser.maxAct=12
    parser.maxCom=32

    parser.lr_model=1e-4
    parser.lr_decay=1.
    parser.max_learning_step=1e9

    parser.gamma=0.99
    parser.T_train=1800
    parser.n_step=10
    parser.K_epochs=3
    parser.eps_clip=0.1
    parser.max_grad_norm=5.0
    parser.max_sigma=0.7
    parser.min_sigma=0.1
    parser.show_figs=False

    parser.log_dir='cfgx_logs'
    parser.log_step=50
    parser.save_dir='cfgx_outputs'
    
    parser.run_name = time.strftime("%Y%m%dT%H%M%S")

    return parser