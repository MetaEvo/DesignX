import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
import time
import warnings
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.utils import set_seed
from utils import clip_grad_norms
from nets.actor_network import ConfigX
from nets.design_network import Actor
from utils import torch_load_cpu, get_inner_model
from utils.logger import log_to_tb_train, log_to_val
import copy, pickle
from utils.make_dataset import *
from rollout import rollout, configx_rollout_ray_v2
from env.optimizer_env import Optimizer
from configx_cfg import get_cfg_options
from utils.make_dataset import Module_pool
from env.ela_feature import get_ela_feature
from env.feature_extractor.feature_extractor import Feature_Extractor
from env import DummyVectorEnv,SubprocVectorEnv, RayVectorEnv
import ray
from env.basic_problem import *

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


class REINFORCE:
    def __init__(self, opts,vector_env):

        # figure out the options
        self.opts = opts
        # the parallel environment
        self.vector_env=vector_env
        # figure out the actor network
        ban_range = []
        if opts.task_class == 'DE':
            ban_range = ['Initialization', 'Niching', 'BC', 'DE_Selection', 'Restart', 'Reduction', 'DE_Mutation', 'DE_Crossover', 'Sharing', 'Termination']
        self.actor = Actor(opts, Module_pool(ban_range=ban_range))
        
        if not opts.test:
            # figure out the optimizer
            self.optimizer = torch.optim.AdamW(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}])
            # figure out the lr schedule
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        # move to cuda
        # self.actor.to(opts.device)
        # self.actor.fe_model.to('cpu')


    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
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
        print('Saving model and state...')
        run_name = self.opts.run_name
        if self.opts.single_task is not None:
            run_name += f"_single-{self.opts.single_task}"
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

    # change working mode to training
    def train(self):
        torch.set_grad_enabled(True)  ##
        self.actor.train()

    def start_training(self, tb_logger):
        if self.opts.train_cfx:
            train_configx(0, self, tb_logger)
        else:
            train(0, self, tb_logger)
        

# inference for training
def train(rank, agent, tb_logger):  
    print("begin training")
    opts = agent.opts
    warnings.filterwarnings("ignore")

    training_problem, test_problem = make_synthetic_dataset()    
        
    # load ConfigX model
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # move optimizer's data onto chosen device
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)

    # generatate the train_dataset and test_dataset
    
    best_epoch=None
    best_avg_best_cost=None
    best_avg_best_rew=None
    best_epoch_list=[]
    mean_per_list=[]
    sigma_per_list=[]
    rew_per_list=[]
    pre_step=0

    epoch = 0
    agent.save(epoch)
    avg_best,sigma,rew = 0, 0, 0

    rew, avg_best, curve, _=rollout(test_problem,opts,agent,cfgX, tb_logger,epoch_id=epoch)
    sigma = np.mean(np.std(avg_best, -1))
    rew = np.mean(rew)
    mean_per_list.append(np.mean(avg_best))
    sigma_per_list.append(sigma)
    rew_per_list.append(rew)
    if epoch==opts.epoch_start:
        best_avg_best_cost=avg_best
        best_avg_best_rew=rew
        best_epoch=epoch
    elif avg_best > best_avg_best_cost:
        best_avg_best_cost=avg_best
        best_epoch=epoch
    best_epoch_list.append(best_epoch)
    # Start the actual training loop
    
    for epoch in range(opts.epoch_start, opts.epoch_end):
        # Training mode
            
        agent.train()
        agent.lr_scheduler.step(epoch)
        task_return = np.zeros(len(training_problem))
        task_cost = np.zeros(len(training_problem))


        # logging
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'],
                                                                                     opts.run_name) , flush=True)

        # start training
        step = np.ceil(len(training_problem) / opts.batch_size) if not opts.syn_train else np.ceil(opts.sub_train_size / opts.batch_size)
        pbar = tqdm(total = step,
                    desc = f'training Epoch {epoch}',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                    position=0)
        
        for batch_id, batch in enumerate(training_problem):
            # train procedule for a batch
            elas, action_features, alg_len, rewards, costs, optimizers, traj_len = train_batch(rank,
                        batch,
                        agent,
                        cfgX,
                        epoch,
                        pre_step,
                        tb_logger,
                        opts,
                        batch_id, 
                        pbar,
                        )
            pre_step += 1
            task_return[batch_id*opts.batch_size:min((batch_id+1)*opts.batch_size, len(training_problem))] = rewards
            task_cost[batch_id*opts.batch_size:min((batch_id+1)*opts.batch_size, len(training_problem))] = costs
                        
        pbar.close()

        if (epoch-opts.epoch_start) %  opts.checkpoint_epochs == 0:
            agent.save(epoch+1)
                    
        if ((epoch + 1)-opts.epoch_start) % opts.update_best_model_epochs==0 or epoch == opts.epoch_end-1:
            # validate the new model
            rew, avg_best,curve, _=rollout(test_problem,opts,agent,cfgX, tb_logger,epoch_id=epoch)
            sigma = np.mean(np.std(avg_best, -1))
            mean_per_list.append(np.mean(avg_best))
            rew = np.mean(rew)
            sigma_per_list.append(sigma)
            rew_per_list.append(rew)
            if epoch+1==opts.epoch_start:
                best_avg_best_cost=avg_best
                best_avg_best_rew=rew
                best_epoch=epoch+1
            elif avg_best > best_avg_best_cost:
                best_avg_best_cost=avg_best
                best_epoch=epoch+1
            best_epoch_list.append(best_epoch)
    

@ray.remote(num_cpus=1, num_gpus=0)
def ray_ela(sample, sample_y, rng):
    return get_ela_feature(sample, sample_y, rng)


@ray.remote(num_cpus=1, num_gpus=0)
def ray_problem_embed(fe_model, dimfes_embed, state_embed, x, y, dim, fes, ub, lb, device_fe):
    with torch.no_grad():
        ela = fe_model(x, y).to(device_fe)
        dimfes = torch.tensor([np.log10(dim)/5, np.log10(fes)/10, ub / 100., lb / 100.]).to(device_fe)
        pinfo = dimfes_embed(dimfes.float())
        return state_embed(torch.concat([ela, pinfo])) 

def problem_embed(agent, x, y, dim, fes, ub, lb):
    for p in tqdm(range(1), desc = 'problem_embed RAY', leave=False, position=1):
        object_refs = [ray_problem_embed.remote(agent.fe_model, agent.dimfes_embed, agent.state_embed, x[j], y[j], dim[j], fes[j], ub[j], lb[j], agent.device_fe) for j in range(len(dim))]
        results = ray.get(object_refs)
    state = []
    # process results
    for p in tqdm(range(len(dim)), desc = 'problem_embed RAY post process', leave=False, position=1):
        state.append(results[p])
    return torch.stack(state)


def train_batch(
        rank,
        problem,
        agent,
        confgix: ConfigX,
        epoch,
        pre_step,
        tb_logger,
        opts,
        batch_id,
        pbar,
        ):
    
    # setup
    agent.train()
    ts = time.time()
    # initial samplings for ELA features
    # elas = []
    ela_fes = []
    ref_cost = []
    ref_x = []
    
    elas = []
    problem_info = []
    Xs, Ys, seds = [], [], []
    # Sampling
    for p in tqdm(range(len(problem)), desc = 'ELA sampling', leave=False, position=1):
        sed = opts.seed + epoch*1024 + batch_id*opts.batch_size + p
        rng = np.random.RandomState(sed)
        problem[p].reset()
        sample = rng.rand(opts.ela_sample*problem[p].dim, problem[p].dim) * (problem[p].ub - problem[p].lb) + problem[p].lb
        sample_y = problem[p].eval(sample)
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
    ela_features = torch.concat([torch.tensor(elas).to(opts.device), torch.tensor(problem_info).float().to(opts.device)], -1)
    
    
    trng = torch.random.get_rng_state()
    ref_cost = np.array(ref_cost)
    # try:
    action_features, logp, modules, alg_len, entropy = agent.actor(ela_features)

    
    torch.random.set_rng_state(trng)
    tact = time.time()
    optimizers = []
    for i, p in enumerate(tqdm(problem, desc = 'Construct envs', leave=False, position=1)):
        opt = Optimizer(opts, p, copy.deepcopy(modules[i]), MaxFEs=p.MaxFEs, skipped_FEs=ela_fes[i], ref_x=ref_x[i], ref_cost=ref_cost[i])
        optimizers.append(opt)
    rewards, collect_gbest, collect_curve, traj_len = configx_rollout_ray_v2(optimizers, confgix, opts, opts.use_configx)
    rewards = torch.from_numpy(rewards.mean(-1))
    costs = np.mean(collect_gbest, -1)

    agent.optimizer.zero_grad()
    loss = -(rewards.to(opts.device) * logp.to(opts.device)).mean()
    # loss = Variable(loss, requires_grad = True)
    loss.backward()
    grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)  # todo: try remove norm
    agent.optimizer.step()
    tloss = time.time()

    # logging
    mini_step = pre_step+1
    tb_logger.add_scalar('train/Return', rewards.cpu().mean().item(), mini_step)
    tb_logger.add_scalar('train/learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)
    grad_norms, grad_norms_clipped = grad_norms
    tb_logger.add_scalar('grad/grad', grad_norms[0].cpu(), mini_step)
    tb_logger.add_scalar('grad/grad_clipped', grad_norms_clipped[0], mini_step)
    tb_logger.add_scalar('train/entropy', entropy.mean().item(), mini_step)
    tb_logger.add_scalar('loss/loss', loss.item(), mini_step)
    tb_logger.add_scalar('loss/nll', -logp.detach().mean().item(), mini_step)
    with open(tb_logger.logdir + f'/modules_train_{mini_step}.pkl', 'wb') as f:
        # np.save(f, modules, allow_pickle=True)
        pickle.dump(modules, f)
        
    pbar.update()
    
    return elas, action_features, alg_len, rewards, costs, optimizers, traj_len


def train_configx(rank, agent, tb_logger):  
    print("begin training")
    opts = agent.opts
    warnings.filterwarnings("ignore")

    # training_problem, test_problem = make_dataset(opts)
    training_problem, test_problem = make_synthetic_dataset()
    
    # load ConfigX model
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
    cfgx_memory = Memory()
    
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(srg_opt, srg_opt.lr_decay, last_epoch=-1,)
    new_training_problem = Synthetic_Dataset(np.random.choice(training_problem.data, size=cfg_opt.trainsize, replace=False), batch_size=opts.batch_size)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # move optimizer's data onto chosen device
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)
    optimizers = []
    alg_lens = []
    train_module_record = {}
    module_stat = {}
    for batch_id, problem in enumerate(new_training_problem):
        ela_fes = []
        ref_cost = []
        ref_x = []
        
        elas = []
        problem_info = []
        Xs, Ys, seds = [], [], []
        # Sampling
        for p in tqdm(range(len(problem)), desc = 'ELA sampling', leave=False, position=1):
            sed = opts.seed + batch_id*opts.batch_size + p
            rng = np.random.RandomState(sed)
            problem[p].reset()
            sample = rng.rand(opts.ela_sample*problem[p].dim, problem[p].dim) * (problem[p].ub - problem[p].lb) + problem[p].lb
            sample_y = problem[p].eval(sample)
            Xs.append(sample)
            Ys.append(sample_y)
            seds.append(sed)
            # ela, fes, tim = get_ela_feature(problem[p], sample, sample_y, sed)
            # elas.append(ela)
            # ela_fes.append(fes + sample.shape[0])
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
        ela_features = torch.concat([torch.tensor(elas).to(opts.device), torch.tensor(problem_info).float().to(opts.device)], -1)
            
        with torch.no_grad():
            action_features, logp, modules, alg_len, entropy = agent.actor(ela_features, rollout=True)
            alg_lens += alg_len.tolist()
        for i, p in enumerate(tqdm(problem, desc = 'Construct envs', leave=False, position=1)):
            # opt = Optimizer(opts, p, modules[i], NPmax=params['NPmax'][i], NPmin=params['NPmin'][i], MaxFEs=min(p.dim**2 * opts.FEs_ratio, opts.MaxFEs) - ela_fes[i])
            opt = Optimizer(opts, p, copy.deepcopy(modules[i]), MaxFEs=p.MaxFEs, skipped_FEs=ela_fes[i], ref_cost=ref_cost[i], ref_x=ref_x[i])
            optimizers.append(opt)
    for i, p in enumerate(problem):
        if not isinstance(p, Composition) and not isinstance(p, Hybrid):
            if p.__class__.__name__ not in train_module_record.keys():
                train_module_record[p.__class__.__name__] = {'dim': p.dim, 'maxfes': p.MaxFEs, 'bound': p.ub, 'mod': copy.deepcopy(modules[i])}
        else:
            name = ""
            for sp in p.sub_problems:
                name += sp.__class__.__name__ + "+"
            if name not in train_module_record.keys():
                train_module_record[name] = {'dim': p.dim, 'maxfes': p.MaxFEs, 'bound': p.ub, 'mod': copy.deepcopy(modules[i])}
        for m in modules[i]:
            if isinstance(m, list):
                for mm in m:
                    if isinstance(mm, Multi_strategy):
                        for op in mm.ops:
                            if op.__class__.__name__ not in module_stat.keys():
                                module_stat[op.__class__.__name__] = 1
                            else:
                                module_stat[op.__class__.__name__] += 1
                    else:
                        if mm.__class__.__name__ not in module_stat.keys():
                            module_stat[mm.__class__.__name__] = 1
                        else:
                            module_stat[mm.__class__.__name__] += 1
            else:
                if m.__class__.__name__ not in module_stat.keys():
                    module_stat[m.__class__.__name__] = 1
                else:
                    module_stat[m.__class__.__name__] += 1

    print('training modules', module_stat)
    best_epoch=None
    best_avg_best_cost=None
    best_epoch_list=[]
    mean_per_list=[]
    sigma_per_list=[]
    rew_per_list=[]

    epoch = 0
    agent.save(epoch)
    avg_best,sigma,rew = 0, 0, 0

    rew, avg_best, curve, _=rollout(test_problem,opts,agent,cfgX, tb_logger,epoch_id=epoch)
    sigma = np.mean(np.std(avg_best, -1))
    rew = np.mean(rew)
    
    mean_per_list.append(np.mean(avg_best))
    sigma_per_list.append(sigma)
    rew_per_list.append(rew)
    if epoch==opts.epoch_start:
        best_avg_best_cost=avg_best
        best_avg_best_rew=rew
        best_epoch=epoch
    elif avg_best > best_avg_best_cost:
        best_avg_best_cost=avg_best
        best_epoch=epoch
    best_epoch_list.append(best_epoch)

    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        # Training mode
        # logging
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'],
                                                                                     opts.run_name) , flush=True)

        # start training
        
        cfgX.update(optimizers, np.zeros(len(optimizers)), alg_lens, tb_logger, upper_epoch=epoch, device='cpu')

        if (epoch-opts.epoch_start) %  opts.checkpoint_epochs == 0:
            cfgX.save(epoch+1)
        
                    
        if ((epoch + 1)-opts.epoch_start) % opts.update_best_model_epochs==0 or epoch == opts.epoch_end-1:
            # validate the new model
            rew, avg_best,curve, srg_info=rollout(test_problem,opts,agent,cfgX, tb_logger,epoch_id=epoch)
            sigma = np.mean(np.std(avg_best, -1))
            mean_per_list.append(np.mean(avg_best))
            rew = np.mean(rew)
            sigma_per_list.append(sigma)
            rew_per_list.append(rew)
            if epoch+1==opts.epoch_start:
                best_avg_best_cost=avg_best
                best_avg_best_rew=rew
                best_epoch=epoch+1
            elif avg_best > best_avg_best_cost:
                best_avg_best_cost=avg_best
                best_epoch=epoch+1
            best_epoch_list.append(best_epoch)
            
