import os
from torch import nn
import torch, copy
from nets.graph_layers import MultiHeadEncoder, MLP_for_actor, EmbeddingNet, PositionalEncoding, PositionalEncodingSin
from nets.graph_layers import MLP3
from torch.distributions import Normal, Gamma, Categorical
from decision_transformer.models.decision_transformer import DecisionTransformer_actions
from utils.make_dataset import Module_pool
import numpy as np
from env.feature_extractor.feature_extractor import Feature_Extractor
from tqdm import tqdm
from components.operators import *
import ray

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
                 module_pool: Module_pool
                 ):
        super(Actor, self).__init__()
        self.module_pool = module_pool
        self.state_dim = opts.state_dim
        self.act_dim = module_pool.N + 1
        self.max_length = 1024
        self.max_ep_len = 1024   
        self.embed_dim = opts.embedding_dim
        self.fes_dim = opts.dimfes_dim
        self.n_layer = opts.n_encode_layers
        self.n_head = opts.n_head
        self.op_dim = opts.op_dim
        self.max_action = opts.maxAct
        self.maxCom=opts.maxCom
        self.opts = opts
        self.device = opts.device
        self.device_fe = self.device
        self.train_fe = False
        self.state_embed = torch.nn.Linear(self.state_dim+4, self.embed_dim).to(self.device)
            
        self.action_embed = torch.nn.Linear(self.act_dim, self.embed_dim).to(self.device)  # + 1 additional start token
        self.pe = PositionalEncodingSin(self.embed_dim, self.max_length).to(self.device)

        self.model = DecisionTransformer_actions(
                state_dim=self.state_dim,
                
                action_tanh=False,
                
                hidden_size=self.embed_dim,
                n_layer=self.n_layer,
                n_head=self.n_head,
                device=opts.device
            ).to(opts.device)
        self.action_predict = nn.Linear(self.embed_dim, module_pool.N).to(self.device)
        
    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}
    
    def forward(self, prob_info, rollout=True):
        """
        prob_info: shape=[bs, state_dim]
        """
        bs = len(prob_info)
        prob_info = prob_info.float().to(self.device)
            
        module_pool = self.module_pool
        actions = [[] for _ in range(bs)]
        # logp = torch.zeros(bs, requires_grad=True).to(self.device) # [[] for _ in range(bs)]
        logp = [0 for _ in range(bs)]
        entropy = [[] for _ in range(bs)]
        active = torch.ones(bs, dtype=torch.bool)

        modules = [[] for _ in range(bs)]  # the generated algorithms

        topo_rules = [['Initialization'] for _ in range(bs)]
        niching = torch.ones(bs)  # the number of subpopultions of each task
        subpop_count = torch.zeros(bs)  # the currently processed subpopulation, also an indicator of whether all subpopulations are completed
        alg_len = torch.zeros(bs)  # the length of the generated algorithms

        masks = torch.zeros(bs, module_pool.N, dtype=torch.bool)  # action sampling mask
        for i in range(bs):
            masks[i] = module_pool.get_mask(topo_rules[i])  # only initialization modules available in the mask
        init_mask = copy.deepcopy(masks)  # mask backup for multi-population generation
        pre_mods = [None for _ in range(bs)]  # the previous modules for topology analysis
        action_features = torch.zeros(bs, self.maxCom,self.act_dim)  # for using one-hot coding as action features, use 0 as start token
        allow_reduction = torch.ones(bs, dtype=bool)   # disable reduction in ES sub population
        index = 0
            
        act_pb = tqdm(total=self.opts.maxCom, desc='Get Action', leave=False, position=1)
        while active.any():
            state_embedding = self.state_embed(prob_info[active])
            if index > 1:
                state_embedding = state_embedding.clone().detach().to(self.device)
                # state_embeddings = self.state_embed(sa)
            state_embeddings_expanded = state_embedding.unsqueeze(1)     # [64, 1, 16]
            
            concatenated_embeddings = state_embeddings_expanded
                
            concatenated_embeddings = self.pe(concatenated_embeddings)

            logits = self.model(concatenated_embeddings)

            action_logits = self.action_predict(logits)
            action_logits[~masks[active]] = -torch.inf
            
            policy = Categorical(torch.softmax(action_logits, -1))
                
            if not rollout:
                action = policy.sample()
                lp = policy.log_prob(action)
                ent = policy.entropy()  # for logging only
            else:
                action = torch.argmax(torch.softmax(action_logits, -1), -1)
                lp = torch.zeros(bs)
                ent = policy.entropy()  # for logging only


            alg_len[active] += 1

            # get the selected modules
            mods = module_pool.actions_to_mods(action)
            # update masks
            j = 0
            for i in range(bs):
                if not active[i]:
                    continue
                # record results
                actions[i].append(action[j])
                logp[i] += lp[j]
                entropy[i].append(ent[j])
                
                action_features[i, index, :] = torch.nn.functional.one_hot((action[j] + 1).to(torch.int64), num_classes=self.act_dim).to(torch.float32)

                if mods[j].topo_type == 'Termination':  # current subpopulation is terminated
                    subpop_count[i] += 1
                    if subpop_count[i] >= niching[i]:  # all subpopulations are terminated
                        active[i] = False
                    else:
                        # back to the mask for the first subpopulation module and start the generation for the next subpopulation module list
                        masks[i] = copy.deepcopy(init_mask[i])
                        modules[i].append([])
                        allow_reduction[i] = True
                else:
                    # get the mask for the next module from the currently selected module
                    former_type = None
                    if pre_mods[i] is not None:
                        former_type=pre_mods[i].topo_type
                    topo_rule = []
                    if mods[j].topo_type == 'ES_sample':
                        allow_reduction[i] = False
                    ban_range = copy.deepcopy(module_pool.ban_range)
                    if niching[i] < 2:  # no niching, then no Sharing
                        ban_range.append('Sharing')
                    if not allow_reduction[i]:
                        ban_range.append('Reduction')
                        ban_range.append('Sharing')
                    for rule in mods[j].get_topo_rule(former_type):
                        if rule not in ban_range:
                            topo_rule.append(rule)

                    masks[i] = module_pool.get_mask(topo_rule)
                    if mods[j].topo_type == 'Initialization':
                        modules[i].append(copy.deepcopy(mods[j]))
                        modules[i].append([])  # start the (only) subpopulation in which modules are contained in a list, assume there is no niching
                        allow_reduction[i] = True
                    elif mods[j].topo_type == 'Niching': 
                        niching[i] = mods[j].Npop
                        init_mask[i] = copy.deepcopy(masks[i])
                        modules[i][-1] = copy.deepcopy(mods[j])  # assumption not hold, insert the niching between initialization and the subpopulation module list
                        modules[i].append([])  # then start the first subpopulation
                        allow_reduction[i] = True
                    else:
                        modules[i][-1].append(copy.deepcopy(mods[j]))  # other modules join the module list for the last subpopulation
                        
                pre_mods[i] = copy.deepcopy(mods[j])
                j += 1
            index += 1
            act_pb.update()
        act_pb.close()
        for i in range(len(entropy)):
            entropy[i] = torch.mean(torch.stack(entropy[i]))
        out = (action_features,#actions,
                torch.stack(logp),  # bs
                modules,
                alg_len,
                torch.tensor(entropy),)
        return out
