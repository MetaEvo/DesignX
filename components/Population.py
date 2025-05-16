import os
import scipy.stats as stats
import numpy as np
import copy
from scipy.stats import qmc


class Population:
    def __init__(self, problem, NPmax=[100], NPmin=[4], NA=3, Xmax=100, Xmin=-100, Vmax=0.2,  multiPop=1, arch_replace='oldest', MaxGen=500, MaxFEs=50000, rng=None):
        self.NPmax = NPmax                         # the upperbound of population size
        self.NPmin = NPmin                          # the lowerbound of population size
        self.multiPop = max(1, multiPop)
        # self.NPmax -= self.NPmax % self.multiPop
        # self.NPmin -= self.NPmin % self.multiPop
        # assert self.NPmin > 0 and self.NPmax >= self.NPmin
        self.NP = np.sum(NPmax)                            # the population size
        self.subNP = NPmax
        self.NA = int(NA * self.NP)                          # the size of archive(collection of replaced individuals)
        self.problem = problem
        self.dim = problem.dim                          # the dimension of individuals
        self.Vmax = Vmax
        self.Xmin = Xmin         # the upperbound of individual value
        self.Xmax = Xmax          # the lowerbound of individual value
        self.arch_replace = arch_replace
        self.MaxFEs = MaxFEs
        self.MaxGen = MaxGen

        if rng is None:
            rng = np.random
        self.cbest = np.inf                       # the best cost in current population, initialize as 1e15
        self.cbest_solution = np.zeros(self.dim)                      # the index of individual with the best cost
        self.gbest = self.init_max = self.pre_gb = np.inf                       # the global best cost
        self.gbest_solution = np.zeros(self.dim)     # the individual with global best cost
        self.lbest = np.ones(self.multiPop) * np.inf                       # the global best cost
        self.lbest_solution = np.zeros((self.multiPop, self.dim))
        self.lbest_velocity = np.zeros((self.multiPop, self.dim))
        self.pbest = [np.ones(self.subNP[i]) * np.inf for i in range(self.multiPop)]                        # the global best cost
        self.pbest_solution = [np.zeros((self.subNP[i], self.dim)) for i in range(self.multiPop)]
        self.velocity = [(rng.rand(self.NP, self.dim) * 2 - 1) * self.Vmax]

        self.group = []    # the population numpop x np x dim
        self.cost = []           # the cost of individuals
        self.archive = []             # the archive(collection of replaced individuals)
        self.archive_cost = []             # the archive(collection of replaced individuals)
        self.archive_index = -1
        self.process_ip = 0
        self.trail = None
        self.trail_cost = None
        self.FEs = 0
        self.step_count = 0
        # self.pop_id = np.zeros(self.NP)

    # initialize cost
    def initialize_costs(self):
        self.cost = []
        for i in range(len(self.group)):
            self.cost.append(self.problem.eval(self.group[i]))
            self.FEs += self.group[i].shape[0]
        self.trail_cost = copy.deepcopy(self.cost)
        self.init_best()
        
    def init_best(self):
        self.gbest = self.cbest = self.pre_gb = np.min(np.concatenate(self.cost))
        self.init_max = np.max(np.concatenate(self.cost))
        self.cbest_solution = self.gbest_solution = np.concatenate(self.group)[np.argmin(np.concatenate(self.cost))]
        self.pbest = copy.deepcopy(self.cost)
        self.pbest_solution = copy.deepcopy(self.group)
        self.pbest_velocity = copy.deepcopy(self.velocity)
        self.update_lbest()

    def evaluation(self):
        self.trail_cost[self.process_ip] = self.problem.eval(self.trail[self.process_ip])
        self.FEs += self.trail[self.process_ip].shape[0]
        
    # update archive, join new individual
    def update_archive(self, xs, ys, rng=None):
        for x, y in zip(xs, ys):
            self.archive_insert(x, y, rng)

    def archive_insert(self, x, y, rng=None):
        if rng is None:
            rng = np.random
        if len(self.archive) < self.NA:
            self.archive.append(x)
            self.archive_cost.append(y)
        else:
            if self.arch_replace == 'worst':
                archive_index = np.argmax(self.archive_cost)
            elif self.arch_replace == 'oldest':
                archive_index = self.archive_index
                self.archive_index = (self.archive_index + 1) % self.NA
            else:
                archive_index = rng.randint(self.NA)
            self.archive[archive_index] = x
            self.archive_cost[archive_index] = y

    def update_lbest(self):
        for i in range(len(self.group)):
            lb = np.argmin(self.cost[i])
            self.lbest[i] = self.cost[i][lb]
            self.lbest_solution[i] = self.group[i][lb].copy()
            self.lbest_velocity[i] = self.velocity[i][lb].copy()

    def update_subpops(self):
        old_x = copy.deepcopy(self.group)
        old_y = copy.deepcopy(self.cost)
        new_xs = copy.deepcopy(self.trail)
        new_ys = copy.deepcopy(self.trail_cost)

        for id in range(self.multiPop):
            new_x, new_y = new_xs[id], new_ys[id]
            self.group[id] = new_x
            self.cost[id] = new_y
            self.pbest_solution[id][self.cost[id] < self.pbest[id]] = new_x[self.cost[id] < self.pbest[id]]
            self.pbest[id] = np.minimum(self.pbest[id], self.cost[id])
        self.update_lbest()

        self.cbest = np.min(np.concatenate(self.cost))
        self.cbest_solution = np.concatenate(self.group)[np.argmin(np.concatenate(self.cost))]
        
        self.pre_gb = self.gbest
        if self.gbest > self.cbest:
            self.gbest = self.cbest
            self.gbest_solution = self.cbest_solution.copy()
        self.step_count += 1
        self.trail = copy.deepcopy(self.group)
        self.trail_cost = copy.deepcopy(self.cost)

    def reduce_subpop(self, removed_id):
        self.trail[self.process_ip] = np.delete(self.trail[self.process_ip], removed_id, 0)
        self.trail_cost[self.process_ip] = np.delete(self.trail_cost[self.process_ip], removed_id, 0)
        self.pbest[self.process_ip] = np.delete(self.pbest[self.process_ip], removed_id, 0)
        self.pbest_solution[self.process_ip] = np.delete(self.pbest_solution[self.process_ip], removed_id, 0)
        self.pbest_velocity[self.process_ip] = np.delete(self.pbest_velocity[self.process_ip], removed_id, 0)
        self.velocity[self.process_ip] = np.delete(self.velocity[self.process_ip], removed_id, 0)
