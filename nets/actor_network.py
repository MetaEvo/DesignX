import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
from torch import nn
import torch, time
from nets.graph_layers import MultiHeadEncoder, MLP_for_actor, EmbeddingNet, PositionalEncoding, PositionalEncodingSin
from nets.graph_layers import MLP3
from torch.distributions import Normal, Gamma, Categorical
import numpy as np
from tqdm import tqdm
import copy, os
from utils import clip_grad_norms
from utils.make_dataset import Taskset
from env import DummyVectorEnv,SubprocVectorEnv
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model
from utils.logger import log_to_tb_train

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
        # self.llm_hidden = llm_hidden
        self.op_embed = opts.op_embed_dim
        self.max_action = opts.maxAct
        self.max_sigma=opts.max_sigma
        self.min_sigma=opts.min_sigma
        self.opts = opts
        # config = [{'in': self.embedding_dim, 'out': self.hidden_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
        #           {'in': self.hidden_dim, 'out': self.hidden_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
        #           {'in': self.hidden_dim, 'out': self.embedding_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
        #          ]
        # self.log_sigma_min = -20.
        # self.log_sigma_max = -1.5
        if self.opts.sep_state:
            self.op_embedder = EmbeddingNet(self.op_dim, self.op_embed)
            self.fla_embedder = EmbeddingNet(self.node_dim, self.op_embed)

            self.embedder = EmbeddingNet(self.op_embed + self.op_embed,
                                         self.embedding_dim)
        else:
            self.embedder = EmbeddingNet(self.node_dim + self.op_dim if self.opts.morphological else self.node_dim,
                                         self.embedding_dim)
        
        if opts.positional is not None and opts.positional != "None":
            self.pos_embedding = PositionalEncoding(self.embedding_dim, opts.maxCom) if opts.positional == 'learnt' else PositionalEncodingSin(self.embedding_dim, opts.maxCom)
        
        if opts.encoder == 'attn':
            self.encoder = mySequential(*(
                    MultiHeadEncoder(self.n_heads_actor,
                                    self.embedding_dim,
                                    self.hidden_dim,
                                    self.normalization)
                    for _ in range(self.n_layers)))  # stack L layers
        else:
            self.encoder = MLP(opts)

        self.decoder = MLP_for_actor(self.embedding_dim, self.decoder_hidden_dim, self.max_action)
        # print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in, q_length=None, detach_state=False, to_critic=False, only_critic=False):
        """
        x_in: shape=[bs, ps, feature_dim]
        """
        # print(x_in)
        if detach_state:
            x_in = x_in.detach()
        if self.opts.sep_state:
            x_in = x_in.to(self.opts.device)
            pe_id, x_ind, x_fla = x_in[:, :, 0].long(), x_in[:, :, 1:self.op_dim+1], x_in[:, :, 1+self.op_dim:]
                
            ind_em = self.op_embedder(x_ind)
            fla_em = self.fla_embedder(x_fla)
            h_em = self.embedder(torch.concatenate((ind_em, fla_em), -1))
        else:
            pe_id = None
            h_em = self.embedder(x_in)

        # pass through embedder
        if self.opts.positional is not None and self.opts.positional != "None":
            # h_em = self.pos_embedding(self.embedder(torch.concatenate((llm_em, state), -1)), pe_id)  # [bs, n_comp, dim_em]
            h_em = self.pos_embedding(h_em, pe_id)  # [bs, n_comp, dim_em]
        # else:
        #     # h_em = llm_em
        #     h_em = self.embedder(x_in)
        # pass through encoder
        logits = self.encoder(h_em, q_length)  # [bs, n_comp, dim_em]
        # share embeddings to critic net
        if only_critic:
            return logits
        # pass through decoder
        decoded = (torch.tanh(self.decoder(logits)) + 1.) / 2.
        # print(decoded)

        decoded[:, :, torch.arange(decoded.shape[-1]//2)*2+1] = decoded[:, :, torch.arange(decoded.shape[-1]//2)*2+1] * (self.max_sigma-self.min_sigma)+self.min_sigma

        return (decoded, logits) if to_critic else decoded
    
    def get_logp(self, logits, actions):
        bs = logits.shape[0]
        logp = torch.zeros(bs)
        entor_p = []
        for i in range(bs):  # for each task
            logit = logits[i].cpu()
            action = actions[i]
            for j in range(len(action)):  # for each component
                act = action[j]
                lgt = logit[j]
                for k in range(len(act)):  # for each action
                    policy = Normal(lgt[k*2], lgt[k*2+1])
                    lp = policy.log_prob(act[k])
                    if torch.isinf(lp):
                        lp = 1e-32
                    logp[i] += lp
                    entor_p.append(policy.entropy())
        return logp, entor_p


class ConfigX:
    def __init__(self, opts):

        # figure out the options
        self.opts = opts
        # figure out the actor network
        self.actor = Actor(opts)
        self.run_name = self.opts.run_name
        self.log_step = 0
        
        if 1 > 0:
            # for the sake of ablation study, figure out the input_dim for critic according to setting
            input_critic=opts.embedding_dim
            # figure out the critic network
            self.critic = Critic(
                input_dim = input_critic,
                hidden_dim1 = opts.hidden_dim1_critic,
                hidden_dim2 = opts.hidden_dim2_critic,
            )

            # figure out the optimizer
            self.optimizer = torch.optim.AdamW(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] +
                [{'params': self.critic.parameters(), 'lr': opts.lr_model}])
            # figure out the lr schedule
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        # move to cuda
        self.actor.to(opts.device)
        if 1 > 0:
            self.critic.to(opts.device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.opts.device=device
        self.actor.opts.device=device
        optimizer_to(self.optimizer, device)

    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            # if self.opts.use_cuda:
            #     torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        # print('Saving model and state...')
        run_name = self.run_name
        # path = os.path.join(self.opts.save_dir, run_name)
        path = self.opts.save_dir
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(path, 'epoch-{}.pt'.format(epoch))
        )
        
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.actor(*args, **kwds)

    # change working mode to evaling
    def eval(self):
        torch.set_grad_enabled(False)  ##
        self.actor.eval()
        if not self.opts.test: self.critic.eval()

    # change working mode to training
    def train(self):
        torch.set_grad_enabled(True)  ##
        self.actor.train()
        if not self.opts.test: self.critic.train()

    def update(self, algorithms, rewards, traj_len, tb_logger, upper_epoch, device):
        opts = self.opts
        # dataset prepare
        if self.opts.trainsize <= 0 or opts.syn_train:
            trainsize = len(algorithms)
        else:
            trainsize = min(self.opts.trainsize, len(algorithms))
        
        selected = np.arange(trainsize)
        if trainsize < len(algorithms):
            if self.opts.alg_sample == 'worst':
                selected = np.argsort(rewards)[:trainsize]
            elif self.opts.alg_sample == 'both_ends':
                half_size = trainsize // 2
                sorted = np.argsort(rewards)
                selected = np.concatenate([sorted[:half_size], sorted[-half_size:]])
            else:  # random
                selected = np.random.choice(np.arange(len(algorithms)), size=trainsize, replace=False)
            
        algorithms = np.array(algorithms, dtype=object)[selected]
        rewards = np.array(rewards)[selected]
        traj_len = np.array(traj_len)[selected]
        
        order = np.argsort(traj_len)
        taskset = Taskset(algorithms.tolist(), opts.batch_size)
        taskset.index = order
        rewards = np.array(rewards)[order]
        # print(f'max trajectory length: {np.max(traj_len)}, min length: {np.min(traj_len)}')
        traj_lens = np.array(traj_len)[order]
        # data record
        task_weights = torch.ones(trainsize)
        cfgx_pb = tqdm(total=opts.epoch_end, desc='ConfigX Train', position=0)
        self.to(device)
        for epoch in range(opts.epoch_end):
            self.lr_scheduler.step()
            self.train()
            task_return = np.zeros(trainsize)
            task_cost = np.zeros(trainsize)
            step = np.ceil(trainsize / opts.batch_size)
            pbar = tqdm(total = step ,
                        desc = f'ConfigX Epoch {epoch}',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                        leave=False,
                        position=1)
            
            for batch_id, batch in enumerate(taskset):
                # if batch_id < 22:
                #     continue
                env_list=[lambda e=p: e for p in batch]
                envs=SubprocVectorEnv(env_list, )
                q_lengths = torch.zeros(len(batch))
                for i in range(len(batch)):
                    q_lengths[i] = batch[i].n_component
                # train procedule for a batch
                tw_l = batch_id*opts.batch_size
                tw_r = min((batch_id+1)*opts.batch_size, trainsize)
                # if tw_r < 1:
                #     tw_r = trainsize
                batch_step, batch_return, batch_cost=self.ppo_batch(
                                    envs,
                                    epoch,
                                    task_weights[tw_l:tw_r],
                                    tb_logger,
                                    opts,
                                    q_lengths,
                                    np.max(traj_lens[int(opts.batch_size*batch_id):int(opts.batch_size*(batch_id+1))]),
                                    batch_id, 
                                    )
                envs.close()
                pbar.update()
                task_return[tw_l:tw_r] += batch_return
                task_cost[tw_l:tw_r] += batch_cost
                # see if the learning step reach the max_learning_step, if so, stop training
                # if pre_step>=opts.max_learning_step:
                #     stop_training=True
                #     break
            pbar.close()
            
            task_weights = torch.softmax(1 - torch.from_numpy(task_return - rewards), 0) * trainsize
            self.save(f'{upper_epoch}-{epoch+1}')
            cfgx_pb.set_postfix({'changed avg reward': np.mean(task_return - rewards),
                                 'ratio': np.mean((task_return - rewards)/rewards),
                                 'from': np.mean(rewards),
                                 'to': np.mean(task_return)})
            cfgx_pb.update()
        cfgx_pb.close()
        self.to('cpu')
        return task_return, task_cost

    def ppo_batch(
            self,
            problem,
            epoch,
            task_weights,
            tb_logger,
            opts,
            q_lengths,
            traj_length,
            batch_id,
            ):
        
        # setup
        memory = Memory()
        # initial instances and solutions
        batch_return = np.zeros(opts.batch_size)
        batch_cost = np.zeros(opts.batch_size)
        problem.seed(opts.seed + np.arange(opts.batch_size) + epoch*1024 + batch_id*opts.batch_size)

        trng = torch.random.get_rng_state()

        state=problem.reset()
        state=torch.FloatTensor(state).to(opts.device)
        
        torch.random.set_rng_state(trng)

        # params for training
        gamma = opts.gamma
        n_step = opts.n_step
        
        K_epochs = opts.K_epochs
        eps_clip = opts.eps_clip
        info = None
        t = 0
        # initial_cost = obj
        done=np.zeros(opts.batch_size, dtype=np.bool_)
        step_pb = tqdm(total=traj_length, desc = 'ConfigX Step', leave=False, position=2)
        # sample trajectory
        end = False
        while not end:
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached_list = []
            bl_val_list = []

            # accumulate transition
            while t - t_s < n_step and not end:  
                memory.states.append(state.clone())
                # memory.states.append(state)
                logits, _to_critic = self.actor(state, 
                                                q_length=q_lengths,
                                                to_critic=True
                                                )
                if torch.sum(torch.isnan(logits)) > 0:
                    exit()
                trng = torch.random.get_rng_state()

                next_state,rewards,is_end,info = problem.step(logits.detach().cpu())

                torch.random.set_rng_state(trng)
                batch_return += rewards
                
                action, entro_p = [], []
                log_lh = []
                for ifo in info:
                    action.append(ifo['action_values'])
                    log_lh.append(ifo['logp'])
                    entro_p += ifo['entropy']
                    
                # log_lh, _ = self.actor.get_logp(logits, action)
                memory.actions.append(copy.deepcopy(action))
                memory.logprobs.append(torch.tensor(log_lh))
                memory.dones.append(torch.tensor(done))
                # action=action.cpu().numpy()
                entropy += entro_p

                baseline_val_detached, baseline_val = self.critic(_to_critic)
                bl_val_detached_list.append(baseline_val_detached)
                bl_val_list.append(baseline_val)

                # state transient
                memory.rewards.append(torch.FloatTensor(rewards).to(opts.device))
                # next
                t = t + 1
                state=torch.FloatTensor(next_state).to(opts.device)
                # state = copy.deepcopy(next_state)
                done = is_end
                step_pb.update()
                end = done.all()
                if hasattr(opts, 'clip_tail') and opts.clip_tail:
                    end = done.any()
            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time
            # begin update
            # old_actions = torch.stack(memory.actions)
            old_actions = memory.actions
            old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
            # old_states = memory.states
            # old_actions = all_actions.view(t_time, bs, ps, -1)
            old_logprobs = torch.tensor([])
            bl_val_detached = torch.tensor([]).to(opts.device)
            bl_val = torch.tensor([]).to(opts.device)
            for tt in range(t_time):
                old_logprobs = torch.concatenate([old_logprobs, memory.logprobs[tt][~memory.dones[tt]]])
                bl_val_detached = torch.concatenate([bl_val_detached, bl_val_detached_list[tt][~memory.dones[tt]]])
                bl_val = torch.concatenate([bl_val, bl_val_detached_list[tt][~memory.dones[tt]]])
            old_logprobs = old_logprobs.to(opts.device)
            # torch.stack(memory.logprobs).detach().view(-1).to(opts.device)
            
            # Optimize PPO policy for K mini-epochs:
            old_value = None
            
            for _k in tqdm(range(K_epochs), desc = 'ConfigX Learn', leave=False, position=3):
                if _k == 0:
                    logprobs = torch.tensor([], requires_grad=True)
                    for tt in range(t_time):
                        logprobs = torch.concatenate([logprobs, memory.logprobs[tt][~memory.dones[tt]]])
                    logprobs = logprobs.to(opts.device)
                else:
                    # Evaluating old actions and values :
                    logprobs = torch.tensor([]).to(opts.device)
                    entropy = []
                    bl_val_detached = torch.tensor([]).to(opts.device)
                    bl_val = torch.tensor([]).to(opts.device)
                    for tt in range(t_time):
                        # get new action_prob
                        logits, _to_critic = self.actor(old_states[tt],
                                                        q_length=q_lengths,
                                                        detach_state = True,
                                                        to_critic = True
                                                        )
                        log_p, entro_p = problem.action_interpret(logits.detach().cpu(), old_actions[tt], memory.dones[tt])
                        logprobs = torch.concatenate([logprobs, torch.tensor(log_p)[~memory.dones[tt]].to(opts.device)])
                        entropy += entro_p

                        baseline_val_detached, baseline_val = self.critic(_to_critic[~memory.dones[tt]])

                        bl_val_detached = torch.concatenate([bl_val_detached, baseline_val_detached.view(-1)])
                        bl_val = torch.concatenate([bl_val, baseline_val.view(-1)])
                entropy = torch.stack(entropy, 0).view(-1)
                    
                    
                Reward = torch.tensor([]).to(opts.device)
                reward_reversed = memory.rewards[::-1]
                dones_reversed = memory.dones[::-1]
                R = self.critic(self.actor(state,q_length=q_lengths,only_critic = True))[0]
                critic_output=R.clone()
                for r in range(len(reward_reversed)):
                    R[~dones_reversed[r]] = R[~dones_reversed[r]] * gamma + reward_reversed[r][~dones_reversed[r]]
                    Reward = torch.concat((Reward, R[~dones_reversed[r]].flip( (-1,))))
                Reward = Reward.flip( (-1,))
                    
                t6 = time.time()
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(torch.clamp(logprobs - old_logprobs.detach(), -torch.inf, 10)).to(Reward.device)
                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached
                weights = 1.


                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2) * weights
                reinforce_loss = reinforce_loss.mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2) * weights
                    baseline_loss = baseline_loss.mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2)) * weights
                    baseline_loss = v_max.mean()

                # check K-L divergence (for logging only)
                approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
                # calculate loss
                loss = baseline_loss + reinforce_loss
                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradient norm and get (clipped) gradient norms for logging
                grad_norms = clip_grad_norms(self.optimizer.param_groups, opts.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                
                # Logging to tensorboard
                log_to_tb_train(tb_logger, self, Reward,R,critic_output, ratios, bl_val_detached, total_cost, grad_norms, memory.rewards, entropy, approx_kl_divergence,
                                reinforce_loss, baseline_loss, logprobs, opts.show_figs, self.log_step)
                self.log_step += 1
                # end update
            

            memory.clear_memory()
        step_pb.close()
        for i, ifo in enumerate(info):
            batch_cost[i] = ifo['gbest_val']
        # return learning steps
        return ( t // n_step + 1) * K_epochs, batch_return, batch_cost

