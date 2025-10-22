import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
from utils.utils import set_seed
import numpy as np
import torch, copy, ray
from tqdm import tqdm
from utils.logger import log_to_val
import os, time
from env import DummyVectorEnv,SubprocVectorEnv, RayVectorEnv
from env.ela_feature import get_ela_feature
from env.optimizer_env import Optimizer
from components.Population import *
from components.operators import *
import pickle, warnings, ray
from env.basic_problem import *


@ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
def ray_ela(sample, sample_y, rng):
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLS_NUM_THREADS'] = '1'
    os.environ['GOTO_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TORCH_NUM_THREADS'] = '1'
    os.environ['RAY_num_server_call_thread'] = '1'
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


# interface for rollout
@torch.no_grad
def rollout(dataloader,opts,agent, confgix, tb_logger=None, run_name=None, epoch_id=0):
    if run_name is None:
        run_name = opts.run_name
    # run_time = time.strftime("%Y%m%dT%H%M%S")
    # opts.repeat = 1
    rollout_name=f'MTRL_{run_name}_{epoch_id}'
    agent.eval()
    # T = opts.MaxGen // opts.skip_step
    batch_size = min(len(dataloader), dataloader.batch_size)
    # to store the whole rollout process
    time_eval=0
    collect_curve=np.zeros((len(dataloader), opts.repeat, opts.MaxFEs//opts.record_internal + 1))
    collect_gbest=np.zeros((len(dataloader), opts.repeat))
    collect_R = np.zeros((len(dataloader), opts.repeat))
    
    all_elas = []
    all_acts = []
    all_lens = []
    all_rews = []
    all_mods = []
    
    # module_record = {}
    module_record = []
    module_stat = {}
    # set the same random seed before rollout for the sake of fairness
    set_seed(opts.testseed)

    pbar = tqdm(total=np.ceil(len(dataloader) / batch_size) * opts.repeat, desc = rollout_name + ' Rollout', position=0)
    for bat_id,problem in enumerate(dataloader):
        # figure out the backbone algorithm
        for r in range(opts.repeat):
            # reset the backbone algorithm
            is_end=False

            # trng = torch.random.get_rng_state()

            # elas = []
            ela_fes = []
            ref_cost = []
            ref_x = []

            elas = []
            Xs, Ys, seds = [], [], []
            problem_info = []
            for p in tqdm(range(len(problem)), desc = 'ELA', leave=False, position=1):
                sed = opts.testseed + r
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
            ela_features = torch.concat([torch.tensor(elas).to(opts.device), torch.tensor(problem_info).float().to(opts.device)], -1).to(opts.device)

            feature_record.append(ela_features.numpy())

            all_elas.append(elas)
            trng = torch.random.get_rng_state()
            with torch.no_grad():
                action_features, logp, modules, alg_len, entropy = agent.actor(ela_features, rollout=True)

            for i, p in enumerate(problem):
                name = p.__class__.__name__
                if isinstance(p, Composition) or isinstance(p, Hybrid):
                    for sp in p.sub_problems:
                        name += "+" + sp.__class__.__name__
                modes = []
                for m in modules[i]:
                    if isinstance(m, list):
                        modes.append([])
                        for mm in m:
                            if isinstance(mm, Multi_strategy):
                                modes[-1].append([])
                                for op in mm.ops:
                                    modes[-1][-1].append(op.__str__())
                                    if op.__str__() not in module_stat.keys():
                                        module_stat[op.__str__()] = 1
                                    else:
                                        module_stat[op.__str__()] += 1
                            else:
                                modes[-1].append(mm.__str__())
                                if mm.__str__() not in module_stat.keys():
                                    module_stat[mm.__str__()] = 1
                                else:
                                    module_stat[mm.__str__()] += 1
                    else:
                        modes.append(m.__str__())
                        if m.__str__() not in module_stat.keys():
                            module_stat[m.__str__()] = 1
                        else:
                            module_stat[m.__str__()] += 1
                        
                module_record.append({'name': name, 'dim': p.dim, 'maxfes': p.MaxFEs, 'bound': p.ub, 'mod': copy.deepcopy(modes)})
            
            torch.random.set_rng_state(trng)
            all_acts.append(action_features)
            all_lens.append(alg_len)
            all_mods += copy.deepcopy(modules)
            
            optimizers = []
            for i, p in enumerate(tqdm(problem, desc = 'Construct envs', leave=False, position=1)):
                opt = Optimizer(opts, p, copy.deepcopy(modules[i]), seed=opts.testseed + r + 1, MaxFEs=p.MaxFEs, skipped_FEs=ela_fes[i], ref_x=ref_x[i], ref_cost=ref_cost[i])
                optimizers.append(opt)

            rewards, cost, curve, _ = configx_rollout_ray_v2(optimizers, confgix, opts, use_configx=opts.use_configx, repeat=1)

            collect_R[batch_size*bat_id:batch_size*(bat_id+1),r] = rewards.squeeze()
            collect_gbest[batch_size*bat_id:batch_size*(bat_id+1),r] = cost.squeeze()
            collect_curve[batch_size*bat_id:batch_size*(bat_id+1),r,:] = curve.squeeze()
            all_rews.append(rewards.squeeze())
            pbar.update()
    saving_path=os.path.join(opts.log_dir, "rollout_{}_{}".format(epoch_id, run_name))
    save_dict={'gbest':collect_gbest,'process':collect_curve,'R_process': collect_R}
    np.save(saving_path,save_dict)

    feature_record = np.concatenate(feature_record, 0)
    
    # log to tensorboard if needed
    if tb_logger:
        log_to_val(tb_logger, np.mean(collect_gbest).item(), np.mean(collect_R).item(), epoch_id) 
        with open(tb_logger.logdir + f'/modules_test_{epoch_id}.pkl', 'wb') as f:
            # np.save(f, all_mods, allow_pickle=True)
            pickle.dump(all_mods, f)
    pbar.close()
    
    # calculate and return the mean and std of final cost
    return collect_R, collect_gbest,collect_curve, {'elas': all_elas, 'action_features': torch.concatenate(all_acts, 0), 'alg_len': torch.concatenate(all_lens, 0), 'rewards': torch.from_numpy(np.concatenate(all_rews, 0))}


@ray.remote(num_cpus=1, num_gpus=0)
def ray_all(agent, env, i, opts, rank, use_configx):
    warnings.filterwarnings("ignore")
    # rank = ray.get_runtime_context().get_actor_id()
    # print(hash(rank))
    # if rank == 0:
    #     step_pb = tqdm(total=2500, desc = 'ConfigX Rollout', leave=False, position=2)
    q_lengths = env.n_control
    
    traj_len = 0
    R = 0
    collect_gbest=0
    collect_curve = np.zeros(opts.MaxFEs//opts.record_internal + 1)
    
    is_end=False

    trng = torch.random.get_rng_state()
    env.seed(opts.testseed + i)
    state = env.reset()
    
    state=state.float().unsqueeze(0)

    torch.random.set_rng_state(trng)

    info = None
    # visualize the rollout process
    while not is_end:
        if use_configx:
            with torch.no_grad():
                logits = agent(state, 
                                    q_length=torch.tensor([q_lengths]),
                                    to_critic=False,
                                    detach_state=True,
                                    )
            trng = torch.random.get_rng_state()

            next_state,rewards,is_end,info = env.step(logits[0].detach())
            torch.random.set_rng_state(trng)
        else:
            next_state,rewards,is_end,info = env.step([None] * opts.maxCom)
        traj_len += 1
        R += rewards
        # put action into environment(backbone algorithm to be specific)
        state=next_state.float().unsqueeze(0)
    #     if rank == 0:
    #         step_pb.update()
    # if rank == 0:
    #     step_pb.close()
    collect_gbest=info['gbest_val']
    collect_curve[-len(info['curve']):] = np.array(info['curve'])
    collect_curve[:-len(info['curve'])] = info['curve'][0]
    return R, collect_gbest, collect_curve, traj_len
    
def configx_rollout_ray_v2(batch, configx, opts, use_configx=True, repeat=None):
    batch_size = len(batch)
    q_lengths = torch.zeros(batch_size)
    for i in range(batch_size):
        q_lengths[i] = batch[i].n_control
        
    if repeat is None:
        repeat = opts.repeat

    # list to store the final optimization result
    R = np.zeros((batch_size, repeat))
    collect_gbest=np.zeros((batch_size,repeat))
    collect_curve = np.zeros((batch_size,repeat, opts.MaxFEs//opts.record_internal + 1))
    traj_len = np.zeros(batch_size)
    # return (np.random.rand(batch_size, repeat) * 0.2) + 0.8, collect_gbest, collect_curve
    configx.to('cpu')

    for i in tqdm(range(repeat), desc = 'ConfigX Repeat', leave=False, position=1):
        object_refs = [ray_all.remote(configx, batch[j], i, opts, j, use_configx) for j in range(batch_size)]
        results = ray.get(object_refs)
        for j in range(batch_size):
            Ri, collect_gbesti, collect_curvei, traj_leni = results[j]
            R[j,i] = Ri
            collect_gbest[j,i] = collect_gbesti
            collect_curve[j,i,:] = collect_curvei
            traj_len[j] +=  traj_leni
        
    return R, collect_gbest, collect_curve, traj_len/repeat
