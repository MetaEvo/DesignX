import os
import time
import argparse


def get_options():

    parser = argparse.Namespace()

    parser.ela_sample=100
    parser.device='cpu'

    # Overall settingss
    parser.seed=1
    parser.dataseed=14
    parser.traindata_seed=1
    parser.testdata_seed=14
    parser.testseed=2025
    parser.repeat=1
    parser.store_path='outputs/'
    parser.batch_size=128
    parser.test_batch_size=128

    # agent settings
    parser.feature_extractor ='ELA'
    parser.maxAct=12
    parser.maxCom=48
    parser.state_dim=9
    parser.dimfes_dim=8
    parser.n_head=8
    parser.embedding_dim=64
    parser.decoder_hidden_dim=32
    parser.hidden_dim=256
    parser.n_encode_layers=1
    parser.op_dim=16
    parser.op_embed_dim=16
    parser.normalization='layer'

    parser.lr_model=1e-4
    parser.lr_decay=1.

    parser.gamma=0.99
    parser.max_grad_norm=5.0
    parser.show_figs = False

    parser.log_dir='logs'
    parser.log_step=1
    parser.save_dir='outputs'
    
    parser.run_name = time.strftime("%Y%m%dT%H%M%S")

    return parser