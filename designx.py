import os, torch, ray
import numpy as np
from nets.design_network import Actor as Agent_1
from nets.actor_network import Actor as Agent_2
from nets.actor_network import Critic
from utils import torch_load_cpu, get_inner_model
from utils.make_dataset import Module_pool
from config import get_options
from configx_cfg import get_cfg_options
from tqdm import tqdm
from env.ela_feature import get_ela_feature
import warnings


# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.algorithms = []
        self.rewards = []
        self.traj_len = []

    def clear_memory(self):
        self.algorithms = []
        self.rewards = []
        self.traj_len = []
        
    def __len__(self):
        return len(self.algorithms)


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch


@ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
def ray_ela(sample, sample_y, rng):
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLS_NUM_THREADS'] = '1'
    os.environ['GOTO_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TORCH_NUM_THREADS'] = '1'
    os.environ['RAY_num_server_call_thread'] = '1'
    return get_ela_feature(sample, sample_y, rng)


@ray.remote(num_cpus=1, num_gpus=0)
def ray_all(agent, env, seed):
    warnings.filterwarnings('ignore')
    q_lengths = env.n_control
    
    traj_len = 0
    R = 0
    collect_gbest=0
    collect_curve = np.zeros(50000//500 + 1)  # maximal possible length
    
    is_end=False

    trng = torch.random.get_rng_state()
    env.seed(seed)
    state = env.reset()
    
    state=state.float().unsqueeze(0)

    torch.random.set_rng_state(trng)

    info = None
    # visualize the rollout process
    while not is_end:
        with torch.no_grad():
            logits = agent(state, 
                            q_length=torch.tensor([q_lengths]),
                            to_critic=False,
                            detach_state=True,
                            )
        trng = torch.random.get_rng_state()

        next_state,rewards,is_end,info = env.step(logits[0].detach())
        torch.random.set_rng_state(trng)
        traj_len += 1
        R += rewards
        # put action into environment(backbone algorithm to be specific)
        state=next_state.float().unsqueeze(0)
    collect_gbest=info['gbest_val']
    collect_curve[-len(info['curve']):] = np.array(info['curve'])
    collect_curve[:-len(info['curve'])] = info['curve'][0]
    return R, collect_gbest, collect_curve, traj_len
    

class DesignX:
    def __init__(self):

        # figure out the options
        opts = get_options()
        self.opts = opts

        # figure out the actor network
        self.actor = Agent_1(opts, Module_pool())
        
    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        print('Saving model and state...')
        run_name = self.opts.run_name
        path = os.path.join(self.opts.save_dir, run_name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(path, 'epoch-{}.pt'.format(epoch))
        )

    # change working mode to evaling
    def eval(self):
        torch.set_grad_enabled(False)  ##
        self.actor.eval()        
        
    def forward(self, ela_feature, rollout=True):
        return self.actor(ela_feature, rollout)
        
    def get_ELA_features(self, problem):
        opts = self.opts
        elas = []
        ela_fes = []
        ref_cost = []
        ref_x = []
        Xs, Ys, seds = [], [], []
        problem_info = []
        for p in tqdm(range(len(problem)), desc = 'ELA', leave=False, position=1):
            sed = opts.testseed
            rng = np.random.RandomState(sed)
            problem[p].reset()
            sample = rng.rand(opts.ela_sample*problem[p].dim, problem[p].dim) * (problem[p].ub - problem[p].lb) + problem[p].lb
            sample_y = problem[p].eval(sample)
            # ela, fes, tim = get_ela_feature(problem[p], sample, sample_y, sed)
            # elas.append(ela)
            # ela_fes.append(fes + sample.shape[0])
            # if np.max(sample_y) - np.min(sample_y) < 1e-5:
            #     print(problem[p], problem[p].dim, problem[p].ub, problem[p].lb)
            #     # print(problem[p].shift, problem[p].rotate, problem[p].shrink)
            #     print(sample.shape[0], np.std(sample, 0))
            #     print(sample_y)
            #     exit()
            Xs.append(sample)
            Ys.append(sample_y)
            seds.append(sed)
            ela_fes.append(sample.shape[0])
            problem_info.append([np.log10(problem[p].dim)/5, np.log10(problem[p].MaxFEs)/10, problem[p].ub / 100., problem[p].lb / 100.])
            ref_cost.append(np.min(sample_y))
            ref_x.append(sample[np.argmin(sample_y)])
        # Get ELA in RAY
        for p in tqdm(range(1), desc = 'ELA RAY', leave=False, position=1):
            object_refs = [ray_ela.remote(Xs[j], Ys[j], seds[j]) for j in range(len(problem))]
            results = ray.get(object_refs)
        
        # process results
        for p in tqdm(range(len(problem)), desc = 'ELA post process', leave=False, position=1):
            elas.append(results[p][0])
            # ela_fes[p] += results[p][1]
        ela_features = torch.concat([torch.tensor(elas).to(opts.device), torch.tensor(problem_info).float().to(opts.device)], -1).to(opts.device)
        return ela_features
        
        
class ConfigX:
    def __init__(self, ):

        # figure out the options
        opts = get_cfg_options()
        self.opts = opts
        # figure out the actor network
        self.actor = Agent_2(opts)
        self.run_name = self.opts.run_name
        self.log_step = 0
        
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
        self.critic.to(opts.device)

    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

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
        self.critic.eval()

    def rollout(self, batch, repeat=1):
        batch_size = len(batch)
        q_lengths = torch.zeros(batch_size)
        for i in range(batch_size):
            q_lengths[i] = batch[i].n_control
            
        # list to store the final optimization result
        R = np.zeros((batch_size, repeat))
        collect_gbest=np.zeros((batch_size,repeat))
        collect_curve = np.zeros((batch_size,repeat, 50000//500 + 1))  # maximal possible length
        traj_len = np.zeros(batch_size)

        for i in tqdm(range(repeat), desc = 'ConfigX Repeat', leave=False, position=1):
            object_refs = [ray_all.remote(self, batch[j]) for j in range(batch_size)]
            results = ray.get(object_refs)
            for j in range(batch_size):
                Ri, collect_gbesti, collect_curvei, traj_leni = results[j]
                R[j,i] = Ri
                collect_gbest[j,i] = collect_gbesti
                collect_curve[j,i,:] = collect_curvei
                traj_len[j] += traj_leni
            
        return R, collect_gbest, collect_curve



