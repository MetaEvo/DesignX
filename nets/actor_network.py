import os
from torch import nn
import torch, time
from nets.graph_layers import MultiHeadEncoder, MLP_for_actor, EmbeddingNet, PositionalEncoding, PositionalEncodingSin
from torch.distributions import Normal, Gamma, Categorical
import numpy as np
from tqdm import tqdm
import copy, os
from utils import clip_grad_norms
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
        
class mySequential(nn.Sequential):
    def forward(self, inputs, q_length):
        for module in self._modules.values():
            # if type(inputs) == tuple:
            #     inputs = module(*inputs)
            # else:
            #     inputs = module(inputs)
            inputs = module(inputs, q_length)
        return inputs


class MLP(torch.nn.Module):
    def __init__(self,
                 opts,
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(opts.embedding_dim, opts.hidden_dim)
        self.fc2 = torch.nn.Linear(opts.hidden_dim, opts.embedding_dim)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, in_, holder):
        result = self.fc1(in_)
        result = self.ReLU(self.fc2(result).squeeze(-1))
        return result



class Actor(nn.Module):

    def __init__(self,
                 opts, 
                 ):
        super(Actor, self).__init__()

        self.embedding_dim = opts.embedding_dim
        self.hidden_dim = opts.hidden_dim
        self.n_heads_actor = opts.encoder_head_num
        self.decoder_hidden_dim = opts.decoder_hidden_dim        
        self.n_layers = opts.n_encode_layers
        self.normalization = opts.normalization
        self.node_dim = opts.node_dim
        self.op_dim = opts.op_dim
        self.op_embed = opts.op_embed_dim
        self.max_action = opts.maxAct
        self.max_sigma=opts.max_sigma
        self.min_sigma=opts.min_sigma
        self.opts = opts
        self.op_embedder = EmbeddingNet(self.op_dim, self.op_embed)
        self.fla_embedder = EmbeddingNet(self.node_dim, self.op_embed)

        self.embedder = EmbeddingNet(self.op_embed + self.op_embed,
                                        self.embedding_dim)
        
        if opts.positional is not None and opts.positional != "None":
            self.pos_embedding = PositionalEncoding(self.embedding_dim, opts.maxCom) if opts.positional == 'learnt' else PositionalEncodingSin(self.embedding_dim, opts.maxCom)
        
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization)
                for _ in range(self.n_layers)))  # stack L layers

        self.decoder = MLP_for_actor(self.embedding_dim, self.decoder_hidden_dim, self.max_action)

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in, q_length=None, detach_state=False, to_critic=False, only_critic=False):
        """
        x_in: shape=[bs, ps, feature_dim]
        """
        if detach_state:
            x_in = x_in.detach()
        x_in = x_in.to(self.opts.device)
        pe_id, x_ind, x_fla = x_in[:, :, 0].long(), x_in[:, :, 1:self.op_dim+1], x_in[:, :, 1+self.op_dim:]
            
        ind_em = self.op_embedder(x_ind)
        fla_em = self.fla_embedder(x_fla)
        h_em = self.embedder(torch.concatenate((ind_em, fla_em), -1))

        # pass through embedder
        if self.opts.positional is not None and self.opts.positional != "None":
            h_em = self.pos_embedding(h_em, pe_id)  # [bs, n_comp, dim_em]
        # pass through encoder
        logits = self.encoder(h_em, q_length)  # [bs, n_comp, dim_em]
        # share embeddings to critic net
        if only_critic:
            return logits
        # pass through decoder
        decoded = (torch.tanh(self.decoder(logits)) + 1.) / 2.

        decoded[:, :, torch.arange(decoded.shape[-1]//2)*2+1] = decoded[:, :, torch.arange(decoded.shape[-1]//2)*2+1] * (self.max_sigma-self.min_sigma)+self.min_sigma

        return (decoded, logits) if to_critic else decoded
    

