import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
import numpy as np
import torch
import torch.nn as nn
from nets.graph_layers import PositionalEncodingSin
import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer_actions(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1_1,action_1_2, ...,action_1_n, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            hidden_size,
            max_length=None,
            n_layer=12,
            **kwargs
    ):
        super().__init__(state_dim, max_length=max_length)
        self.hidden_size = hidden_size
        
        # print("**kwargs:",kwargs)
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=n_layer,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        # print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}

    def forward(self, states):

        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # actions =  torch.nn.functional.one_hot(actions.to(torch.int64), num_classes=self.act_dim).to(torch.float32)

        transformer_outputs = self.transformer(
            inputs_embeds=states,
        )
        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_length, self.hidden_size)
        return x[:,-1, :]
    
