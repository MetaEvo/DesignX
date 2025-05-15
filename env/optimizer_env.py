import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
from sqlite3 import NotSupportedError
import time
import copy
import gym
from gym import spaces
import numpy as np
from scipy.spatial.distance import cdist
from components.operators import *
from components.Population import Population


class Optimizer(gym.Env):
    def __init__(self, problem, modules, seed=None,
                 skipped_FEs = 0, ref_x = None, ref_cost=None) -> None:

        self.problem = problem
        self.MaxFEs = problem.MaxFEs
        self.skip_step = 1
        self.skipped_FEs = skipped_FEs
        self.ref_cost = ref_cost
        self.ref_x = ref_x
        # ---------------------------- init some variables may needed ---------------------------- #
        self.op_strategy = None
        self.global_strategy = {}
        self.bound_strategy = []
        self.select_strategies = []
        self.regroup_strategy = None
        self.restart_strategies = []
        self.record_internal = 500
        self.maxCom = 48

        self.rng_seed = seed
        self.rng = np.random
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.rng = np.random.RandomState(seed)

        self.trng = torch.random.get_rng_state()

        pe_id = 0
        self.init_mod = modules[0]
        self.init_mod.pe_id = pe_id
        pe_id += 1
        self.nich_mod = None
        raw_module = modules[1:]  # ignore init
        self.Npop = 1
        if isinstance(modules[1], Niching):
            self.Npop = modules[1].Npop
            self.nich_mod = modules[1]
            self.nich_mod.pe_id = pe_id
            pe_id += 1
            raw_module = modules[2:]  # ignore init and niching
        self.NPmax = []
        self.NPmin = []
        self.NA = 2
        self.Vmax = 0.2
        self.arch_replace = 'oldest'
        self.n_component = 0
        self.n_control = 0
        self.n_subs = np.zeros(self.Npop)
        self.es_sample = np.zeros(self.Npop, dtype=bool)
        self.modules = [[] for _ in range(self.Npop)]
        for i, mod in enumerate(raw_module):
            for j, submod in enumerate(mod):
                if isinstance(submod, Pop_Size):
                    self.NPmax.append(submod.pop_size)
                    continue
                if isinstance(submod, Reduce_Size):
                    self.NPmin.append(submod.pop_size)
                    continue
                self.modules[i].append(raw_module[i][j])
                self.modules[i][-1].pe_id = pe_id
                pe_id += 1
                if isinstance(submod, Controllable):
                    self.modules[i][-1].act_index = self.n_control
                    self.n_control += 1
                self.n_component += 1
                self.n_subs[i] += 1
            if len(self.NPmin) < len(self.NPmax):
                self.NPmin.append(self.NPmax[-1])
        # -------------------------------- ob space -------------------------------- #
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_control, 12, ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(0, 100, shape=(self.n_component, 12 * 2))

    def seed(self, seed=None):
        self.rng_seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random
        self.trng = torch.random.get_rng_state()

    def reset(self):
        self.problem.reset()
        self.population = Population(self.problem,
                                     NPmax=self.NPmax, 
                                     NPmin=self.NPmin, 
                                     NA=self.NA,
                                     Xmax=self.problem.ub, 
                                     Xmin=self.problem.lb, 
                                     Vmax=self.Vmax, 
                                     multiPop=self.Npop, 
                                     arch_replace=self.arch_replace,
                                     MaxFEs=self.MaxFEs,
                                     rng=self.rng,
                                     )
        self.population.FEs += self.skipped_FEs
        self.population = self.init_mod(self.population, self.rng, self.rng_seed)
        if self.ref_cost is not None and self.ref_x is not None:
            id = np.argmax(self.population.cost[0])
            self.population.group[0][id] = copy.deepcopy(self.ref_x)
            self.population.cost[0][id] = self.ref_cost
        self.population.init_best()
        
        if self.nich_mod is not None:
            self.population = self.nich_mod(self.population, self.rng)
        self.gbest = self.pre_gb = self.init_gb = self.population.gbest  # For reward
        if self.ref_cost is not None:
            self.init_gb = self.ref_cost
            self.pre_gb = min(self.pre_gb, self.ref_cost)
        
        for i in range(self.Npop):
            if self.es_sample[i]:
                self.population.group[i] = self.rng.standard_normal(size=self.population.group[i].shape)
                self.population.cost[i] = self.problem.eval(self.population.group[i])
                self.population.trail[i] = copy.deepcopy(self.population.group[i])
                self.population.trail_cost[i] = copy.deepcopy(self.population.trail_cost[i])
        self.population.init_best()
        
        self.stag_count = 0
        # self.FEs = self.population.NP
        self.step_count = 0
        self.curve_record = []
        self.record_index = 0
        while self.population.FEs >= self.record_index * self.record_internal:
            self.curve_record.append(self.gbest)
            self.record_index += 1
        # self.curve_record.append(self.init_gb)
        return self.get_state()
            
    def cal_feature(self, group, cost, gbest, gbest_solution, cbest, cbest_solution):
        features = [] # 9

        gbest_ = np.log10(max(1e-8, gbest) + 0)
        cbest_ = np.log10(max(1e-8, cbest) + 0)
        cost_ = cost.copy()
        cost_[cost_ < 1e-8] = 1e-8
        cost_ = np.log10(cost_ + 0)
        init_max = np.log10(self.population.init_max + 0)
        features.append(gbest_ / init_max)
        features.append(cbest_ / init_max)
        features.append(np.mean(cost_ / init_max))
        features.append(np.std(cost_ / init_max))

        dist = np.sqrt(np.sum((group[None,:,:] - group[:,None,:]) ** 2, -1))
        features.append(np.max(dist) / (self.population.Xmax - self.population.Xmin) / np.sqrt(self.problem.dim))
        top10 = np.argsort(cost)[:int(max(1, 0.1*len(cost)))]
        dist10 = np.sqrt(np.sum((group[top10][None,:,:] - group[top10][:,None,:]) ** 2, -1))
        features.append((np.mean(dist10) - np.mean(dist)) / (self.population.Xmax - self.population.Xmin) / np.sqrt(self.problem.dim))

        # FDC
        d_lbest = np.sqrt(np.sum((group - gbest_solution) ** 2, -1))
        c_lbest = cost - gbest
        features.append(np.mean((c_lbest - np.mean(c_lbest)) * (d_lbest - np.mean(d_lbest))) / (np.std(c_lbest) * np.std(d_lbest) + 0.00001))
        d_cbest = np.sqrt(np.sum((group - cbest_solution) ** 2, -1))
        c_cbest = cost - cbest
        features.append(np.mean((c_cbest - np.mean(c_cbest)) * (d_cbest - np.mean(d_cbest))) / (np.std(c_cbest) * np.std(d_cbest)+ 0.00001))

        # features = []
        features.append((self.MaxFEs - self.population.FEs) / self.MaxFEs)
        # features.append((self.MaxGen - self.step_count) / self.MaxGen)
        
        features = torch.tensor(features)

        return features

    def get_state(self):
        states = []
        for i, ops in enumerate(self.modules):
            local_state = self.cal_feature(self.population.group[i], 
                                           self.population.cost[i], 
                                           self.population.lbest[i], 
                                           self.population.lbest_solution[i], 
                                           self.population.cbest, 
                                           self.population.cbest_solution)
            for io, op in enumerate(ops):
                if isinstance(op, Uncontrollable):
                    continue
                states.append(torch.concat((torch.tensor([op.pe_id]), op.get_id(), local_state), -1))

        states = torch.stack(states)
        if states.shape[0] < self.maxCom:  # mask 
            mask = torch.zeros(self.maxCom - states.shape[0], states.shape[-1])
            states = torch.concat((states, mask), 0)
        return states

    def get_reward(self):
        return max(self.pre_gb - self.gbest, 0) / self.init_gb

    def step(self, logits):
        rewards = 0
        if self.population.FEs >= self.MaxFEs:
            info = {}
            info['action_values'] = []
            info['logp'] = torch.tensor(0.)
            info['entropy'] = [torch.tensor(0.)]
            info['gbest_val'] = self.population.gbest
            info['gbest_sol'] = self.population.gbest_solution
            info['init_gb'] = self.init_gb
            info['curve'] = self.curve_record
            info['MaxFEs'] = self.MaxFEs
            info['FEs'] = self.population.FEs
            info['rec_index'] = self.record_index
            return self.get_state(),0,True,info
        state,reward,is_end,info = self.one_step(logits)
        rewards += reward
        if self.skip_step < 2:
            return state,rewards,is_end,info
        for t in range(1, self.skip_step):
            _,reward,is_end,_ = self.one_step([None]*logits.shape[0], had_action=info['had_action'])
            rewards += reward
        return self.get_state(),rewards,is_end,info

    def one_step(self, logits, had_action=None):
        torch.random.set_rng_state(self.trng)

        if had_action is None:
            had_action = [None for _ in range(self.n_control)]
        had_action_rec = [None for _ in range(self.n_control)]
        # reproduction
        action_values = [[] for _ in range(self.n_control)]
        logp_values = 0
        entropys = []
        syn_bar = np.zeros(self.Npop)
        while not (syn_bar >= self.n_subs).all():
            for ip in range(self.Npop):
                self.population.process_ip = ip
                # for io, op in enumerate(self.modules[ip]):
                st = int(syn_bar[ip])
                for io in range(st, int(self.n_subs[ip])):
                    op = self.modules[ip][io]
                    # subpops[ip]['problem'] = self.problem
                    if op.sym_bar_require and io > syn_bar[ip]:
                        syn_bar[ip] = io
                        break
                    syn_bar[ip] += 1
                    if isinstance(op, Controllable):
                        act_index = op.act_index
                        res = op(logits[act_index], self.population, softmax=False, rng=self.rng, had_action=had_action[act_index])
                        # print(ip, io, i_component, op)
                        if had_action[act_index] is None:
                            action_values[act_index] = res['actions']
                            logp_values += np.sum(res['logp'])
                            entropys += res['entropy']
                            had_action_rec[act_index] = res['had_action']
                        self.population = res['result']
                    else:
                        self.population = op(self.population)
        self.population.update_subpops()

        self.step_count += 1

        self.pre_gb = min(self.pre_gb, self.gbest)
        if self.gbest > self.population.gbest:
            self.gbest = min(self.gbest, self.population.gbest)
            self.stag_count = 0
        else:
            self.stag_count += 1
        while self.population.FEs >= self.record_index * self.record_internal and self.record_index <= self.MaxFEs//self.record_internal:
            self.curve_record.append(self.gbest)
            self.record_index += 1
        info = {}
        info['action_values'] = action_values
        info['logp'] = logp_values
        info['entropy'] = entropys
        info['gbest_val'] = self.population.gbest
        info['gbest_sol'] = self.population.gbest_solution
        info['init_gb'] = self.init_gb
        info['curve'] = self.curve_record
        info['had_action'] = had_action_rec
        info['MaxFEs'] = self.MaxFEs
        info['FEs'] = self.population.FEs
        info['rec_index'] = self.record_index
        is_done = self.population.FEs >= self.MaxFEs
        self.trng = torch.random.get_rng_state()

        return self.get_state(), self.get_reward(), is_done, info
