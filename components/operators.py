import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
from xml.dom import NotSupportedErr
import numpy as np
import copy
import torch
from torch.distributions import Normal, Categorical
from sklearn.cluster import KMeans
from scipy.stats import qmc
import torch
import scipy.stats as stats
from .Population import *
import cma, time
from scipy.stats import norm


################################ BASE CLASS ################################
class Module:
    def __init__(self) -> None:
        self.topo_type = ''  # the module class, to identify the topological rule in algorithm structure, most are the same as the module, except modules for spercific ECs, such as modules only for DE
        self.id = []  # will be a 3-element list indicating controllable?, module id and sub-module id, respectively
        self.pool_id = 0  # the id for masking in module selection
        self.mod_type = ''  # an identifier when the module is registerred into module pool, which catagoriy should it belong to, so that we can assign an incremental id for it
        self.topo_rule = []  # the list of valid module classes after current module
        self.pe_id = 0  # the positional encoding index in the algorithm structure
        self.sym_bar_require = False  # some operations need the results from other sub populations, stop the process of current population before here until other populations are barred or completed

    def get_id(self, mod_bin=6, sub_bin=9):
        id = torch.zeros(1 + mod_bin + sub_bin)
        con, mod, sub = self.id
        id[0] = con
        for i in range(mod_bin, 0, -1):
            if mod < 1:
                break
            id[i] = mod & 1
            mod //= 2
        for i in range(mod_bin + sub_bin, mod_bin, -1):
            if sub < 1:
                break
            id[i] = sub & 1
            sub //= 2
        return id
    
    def get_id_hash(self, mod_bin=6, sub_bin=9):
        hash = self.id[2] + self.id[1] * np.power(2, sub_bin) + self.id[0] * np.power(2, sub_bin + mod_bin)
        return hash
    
    def set_id(self, id):
        self.id.append(id)

    def get_type(self):
        return self.topo_type
    
    def get_topo_rule(self, former_type=None):  # maybe some modules need the module class of the former module to determine its successor range
        return self.topo_rule

    def exec(self,):
        pass

    def __call__(self, *args, **kwds):
        return self.exec(*args, **kwds)
    
    def __str__(self):
        return self.__class__.__name__



class Uncontrollable(Module):
    def __init__(self) -> None:
        super().__init__()
        self.id = [0]
        self.topo_rule = []

    def get_rule(self):
        return self.topo_rule
    

class Initialization(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(1)
        self.topo_rule = ['Niching', 'Pop_Size']
        self.topo_type = 'Initialization'
        self.mod_type = 'Initialization'

    def init(self, NP, dim, Xmax=5, Xmin=-5, rng=None, seed=None):
        pass

    def exec(self, population: Population, rng=None, seed=None):
        population.group = [np.clip(self.init(population.NP, population.dim, population.Xmax, population.Xmin, rng, seed), population.Xmin, population.Xmax)]
        population.initialize_costs()
        population.trail = copy.deepcopy(population.group)
        return population


class Niching(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(2)
        self.topo_rule = ['Pop_Size']
        self.topo_type = 'Niching'
        self.mod_type = 'Niching'

    def nich(self):
        pass

    def exec(self, population: Population, rng=None, seed=None):
        group = np.concatenate(population.group)
        cost = np.concatenate(population.cost)
        pop_id = self.nich(group, cost, population.NPmax, rng=rng)
        velocity = np.concatenate(population.velocity)
        pbest = np.concatenate(population.pbest)
        pbest_solution = np.concatenate(population.pbest_solution)
        pbest_velocity = np.concatenate(population.pbest_velocity)
        population.group = []
        population.cost = []
        population.velocity = []
        population.pbest = []
        population.pbest_solution = []
        population.pbest_velocity = []
        for i in range(self.Npop):
            population.group.append(group[pop_id == i])
            population.cost.append(cost[pop_id == i])
            population.velocity.append(velocity[pop_id == i])
            population.pbest_solution.append(pbest_solution[pop_id == i])
            population.pbest.append(pbest[pop_id == i])
            population.pbest_velocity.append(pbest_velocity[pop_id == i])
        population.trail = copy.deepcopy(population.group)
        population.trail_cost = copy.deepcopy(population.cost)
        population.update_lbest()
        return population


class BC(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(3)
        self.topo_rule = ['GA_Selection', 'DE_Selection', 'PSO_Selection']
        self.topo_type = 'BC'
        self.mod_type = 'BC'

    def bc(self, ):
        pass

    def get_topo_rule(self, former_type=None):
        if former_type == 'GA_Mutation':
            return ['GA_Selection']
        elif former_type == 'DE_Crossover':
            return ['DE_Selection']
        elif former_type == 'PSO_update' or former_type == 'ES_sample':
            return ['PSO_Selection']
        else:
            return self.topo_rule

    def exec(self, population: Population, rng=None, seed=None):
        population.trail[population.process_ip] = np.clip(population.trail[population.process_ip], 10*population.Xmin, 10*population.Xmax)
        population.trail[population.process_ip] = self.bc(population, rng)
        return population


class Selection(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(4)
        self.topo_rule = ['Restart', 'Reduction', 'Sharing', 'Termination']
        self.mod_type = 'Selection'
        
class GA_Selection(Selection):
    def __init__(self) -> None:
        super().__init__()
        self.topo_type = 'GA_Selection'

class DE_Selection(Selection):
    def __init__(self) -> None:
        super().__init__()
        self.topo_type = 'DE_Selection'

class PSO_Selection(Selection):
    def __init__(self) -> None:
        super().__init__()
        self.topo_type = 'PSO_Selection'


class Restart(Uncontrollable):
    def __init__(self, reinit_method) -> None:
        super().__init__()
        self.id.append(5)
        self.topo_rule = ['Termination']
        self.topo_type = 'Restart'
        self.mod_type = 'Restart'
        self.reinit_method = reinit_method()

    def check_restart(self):
        pass

    def exec(self, population: Population, rng=None, seed=None):
        if self.check_restart(population):
            population.trail[population.process_ip] = self.reinit_method.init(population.trail[population.process_ip].shape[0], population.dim, population.Xmax, population.Xmin, rng, seed)
            population.evaluation()
        return population


class Reduction(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(6)
        self.topo_type = 'Reduction'
        self.mod_type = 'Reduction'
        self.topo_rule = ['Reduce_Size']


class Termination(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(7)
        self.topo_type = 'Termination'
        self.mod_type = 'Termination'


class Pop_Size(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()     
        self.id.append(8)   
        self.topo_type = 'Pop_Size'
        self.mod_type = 'Pop_Size'
        self.topo_rule = ['GA_Crossover', 'DE_Mutation', 'PSO_update', 'ES_sample']


class Reduce_Size(Uncontrollable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(9)
        self.topo_type = 'Reduce_Size'
        self.mod_type = 'Reduce_Size'
        self.topo_rule = ['Restart', 'Termination']


class Controllable(Module):
    def __init__(self) -> None:
        super().__init__()
        self.id = [1]
        self.topo_rule = []
        self.config_space = {}
        self.act_index = 0

    def get_config(self):
        pass

    def set_config(self):
        pass


class Mutation(Controllable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(1)
        self.mod_type = 'Mutation'

class GA_Mutation(Mutation):
    def __init__(self) -> None:
        super().__init__()
        self.topo_rule = ['BC']
        self.topo_type = 'GA_Mutation'

class DE_Mutation(Mutation):
    allow_qbest=False
    allow_archive=False
    Nxn = 0
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__()
        self.topo_rule = ['DE_Crossover']
        self.topo_type = 'DE_Mutation'
        self.use_qbest = use_qbest and self.allow_qbest
        self.use_archive = use_archive and self.allow_archive
        self.united_qbest = united_qbest and self.use_qbest and self.use_archive
        self.archive_id = archive_id
        self.united_id = united_id        
        
    def __str__(self):
        name = self.__class__.__name__
        if self.use_qbest:
            name += "+qb"
        if self.use_archive:
            name += "+A"
        return name


class Crossover(Controllable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(2)
        self.mod_type = 'Crossover'

class GA_Crossover(Crossover):
    def __init__(self) -> None:
        super().__init__()
        self.topo_rule = ['GA_Mutation']
        self.topo_type = 'GA_Crossover'

class DE_Crossover(Crossover):
    allow_qbest=False
    allow_archive=False
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False) -> None:
        super().__init__()
        self.topo_rule = ['BC']
        self.topo_type = 'DE_Crossover'
        self.use_qbest = use_qbest and self.allow_qbest
        self.use_archive = use_archive and self.allow_archive
        self.united_qbest = united_qbest and self.use_qbest and self.use_archive

    def __str__(self):
        name = self.__class__.__name__
        if self.use_qbest:
            name += "+qb"
        if self.use_archive:
            name += "+A"
        return name


class PSO_update(Controllable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(3)
        self.topo_rule = ['BC']
        self.topo_type = 'PSO_update'
        self.mod_type = 'PSO_update'


class ES_sample(Controllable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(6)
        self.topo_rule = ['BC']
        self.topo_type = 'ES_sample'
        self.mod_type = 'ES_sample'



class Multi_strategies(Controllable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(4)
        self.topo_rule = []
        self.mod_type = 'Multi_strategy'


class Sharing(Controllable):
    def __init__(self) -> None:
        super().__init__()
        self.id.append(5)
        self.topo_rule = ['Reduction', 'Termination']
        self.topo_type = 'Sharing'
        self.mod_type = 'Sharing'
        self.sym_bar_require = True
################################ BASE CLASS ################################


################################ TOOL FUNCTION ################################
def qbest(q, group, cost):
    NP, dim = group.shape
    order = np.argsort(cost)
    group = group[order]
    cost = cost[order]
    if isinstance(q, float):
        qN = min(group.shape[0], max(2, int(np.round(q * NP))))
    else:
        qN = np.minimum(group.shape[0], np.maximum(2, np.round(q * NP)))
    return qN, group, cost


def with_archive(group, archive, cost=None, a_cost=None):
    if cost is not None and a_cost is not None:
        if len(archive) < 1:
            return group, cost
        return np.concatenate((group, archive), 0), np.concatenate((cost, a_cost), 0)
    if len(archive) < 1:
        return group
    return np.concatenate((group, archive), 0)


def get_xn(population, num_xn, use_archive=False, archive_id=[], united_id=[], rng=None):
    if rng is None:
        rng = np.random
    group = population.trail[population.process_ip].copy()
    NP, dim = group.shape
    rn = np.ones((NP, num_xn)) * -1
    xn = np.zeros((NP, num_xn, dim))
    arange = np.arange(NP)
    if use_archive:
        archive = population.archive
        united_archive = with_archive(group, archive)
        NPa = len(archive)
        NPu = united_archive.shape[0]

    for i in range(num_xn):
        if use_archive and ((len(archive_id) > 0 and i in archive_id and NPa > 0) or (len(united_id) > 0 and i in united_id)):
            continue
        r = rng.randint(NP, size=NP)
        duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(i)], 0))[0]
        while duplicate.shape[0] > 0 and NP > num_xn:
            r[duplicate] = rng.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(i)], 0))[0]
        rn[:,i] = r
        xn[:, i, :] = group[r]

    if use_archive and len(archive_id) > 0 and NPa > 0:
        for ia in archive_id:
            if ia >= num_xn:
                continue
            r = rng.randint(NPa, size=(NP))
            rn[:,ia] = r
            xn[:, ia, :] = archive[r]

    if use_archive and len(united_id) > 0:
        for iu in united_id:
            if iu >= num_xn:
                continue
            r = rng.randint(NPu, size=(NP))
            duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(num_xn)], 0))[0]
            while duplicate.shape[0] > 0:
                r[duplicate] = rng.randint(NPu, size=duplicate.shape[0])
                duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(num_xn)], 0))[0]
            rn[:,iu] = r
            xn[:, iu, :] = united_archive[r]

    return xn


def get_xqb(population, q, united_qbest=False, rng=None):
    if rng is None:
        rng = np.random
    group = population.trail[population.process_ip].copy()
    NP, dim = group.shape
    if united_qbest:
        group, _ = with_archive(population.trail[population.process_ip], population.archive, population.trail_cost[population.process_ip], population.archive_cost)
    qN, qpop, q_cost = qbest(q, group, population.trail_cost[population.process_ip])
    xqb = qpop[rng.randint(qN, size=NP)]
    return xqb


def get_qb_parent(population, q, united_qbest=False, rng=None):
    if rng is None:
        rng = np.random
    group = population.group[population.process_ip]
    NP, dim = group.shape
    if united_qbest:
        group, _ = with_archive(population.group[population.process_ip], population.archive, population.cost[population.process_ip], population.archive_cost)
    qN, qpop, q_cost = qbest(q, group, population.cost[population.process_ip])
    xqb = qpop[rng.randint(qN, size=NP)]
    return xqb


def parent_selection(action, population, rng=None):
    if rng is None:
        rng = np.random
    group, cost = population.group[population.process_ip], population.cost[population.process_ip]
    NP, dim = group.shape
    if action == 'rand':
        permu = rng.permutation(NP)
        return permu[:NP//2], permu[NP//2:]
    elif action == 'inter':
        arange = np.arange(NP//2)
        return arange * 2, arange * 2 + 1
    elif action == 'parti':
        return np.arange(NP//2), np.arange(NP//2) + NP//2
    elif action == 'rank_inter':
        order = np.argsort(cost)
        return order[:NP//2], order[NP//2:]
    else:
        order = np.argsort(cost)
        arange = np.arange(NP//2)
        return order[arange * 2], order[arange * 2 + 1]


def action_interpret(config_space, logits, softmax=False, fixed_action=None):
    actions = {}
    action_value = []
    logp = []
    entropy = []
    for i, key in enumerate(config_space.keys()):
        param = config_space[key]
        mu, sigma = logits[i*2:i*2+2]
        try:
            policy = Normal(mu, sigma)
        except ValueError:
            print('config_space', config_space)
            print('logits', logits)
            exit()
        if param['type'] == 'float':
            if fixed_action is None:
                action = policy.sample()
            else:
                action = fixed_action[i]
            logp.append(policy.log_prob(action))
            entropy.append(policy.entropy())
            action_value.append(action)
            actions[key] = torch.clip(action, 0, 1).item() * (param['range'][1] - param['range'][0]) + param['range'][0]
        else:
            rang = (torch.arange(len(param['range']))) / (len(param['range']) - 1)
            probs = torch.exp(policy.log_prob(rang))
            if softmax:
                cate = Categorical(torch.softmax(probs/torch.sum(probs), -1))
            else:
                cate = Categorical(probs/torch.sum(probs))
            if fixed_action is None:
                if torch.max(probs) < 1e-3:
                    action = torch.argmin((rang - mu) ** 2)
                else:
                    action = cate.sample()
            else:
                action = fixed_action[i]
            
            logp.append(cate.log_prob(torch.tensor(action)))
            action_value.append(action)
            entropy.append(policy.entropy())
            actions[key] = param['range'][action]
    return actions, action_value, logp, entropy

################################ TOOL FUNCTION ################################


################################ SUB-MODULE ################################
"""
Initialization: gaussian, sobol, lhs, halton, uniform
"""
class Gaussian_Init(Initialization):
    def __init__(self) -> None:
        super().__init__()

    def init(self, NP, dim, Xmax=5, Xmin=-5, rng=None, seed=None):
        if rng is None:
            rng = np.random
        return rng.normal(loc=(Xmax + Xmin) / 2, scale=(Xmax - Xmin) * 0.2, size=(NP, dim))


class Sobol_Init(Initialization):
    def __init__(self) -> None:
        super().__init__()

    def init(self, NP, dim, Xmax=5, Xmin=-5, rng=None, seed=None):
        if rng is None:
            rng = np.random
        sampler = qmc.Sobol(d=dim, seed=seed)
        return sampler.random(NP) * (Xmax - Xmin) + Xmin


class LHS_Init(Initialization):
    def __init__(self) -> None:
        super().__init__()

    def init(self, NP, dim, Xmax=5, Xmin=-5, rng=None, seed=None):
        if rng is None:
            rng = np.random
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        return sampler.random(NP) * (Xmax - Xmin) + Xmin


class Halton_Init(Initialization):
    def __init__(self) -> None:
        super().__init__()

    def init(self, NP, dim, Xmax=5, Xmin=-5, rng=None, seed=None):
        if rng is None:
            rng = np.random
        sampler = qmc.Halton(d=dim, seed=seed)
        return sampler.random(NP) * (Xmax - Xmin) + Xmin


class Uniform_Init(Initialization):
    def __init__(self) -> None:
        super().__init__()

    def init(self, NP, dim, Xmax=5, Xmin=-5, rng=None, seed=None):
        if rng is None:
            rng = np.random
        return rng.rand(NP, dim) * (Xmax - Xmin) + Xmin


"""
Niching
"""
class Rand_Nich(Niching):
    def __init__(self, Npop) -> None:
        super().__init__()
        self.Npop = Npop

    def nich(self, x, y, subsize=None, rng=None):
        NP = y.shape[0]
        if subsize is None:
            subsize = np.ones(self.Npop) * NP//self.Npop
            if np.sum(subsize) < NP:
                subsize[-1] += NP - np.sum(subsize)
        pop_id = np.zeros(NP)
        order = rng.permutation(NP)
        pre = 0
        for i in range(self.Npop):
            pop_id[order[int(pre):int(pre+subsize[i])]] = i
            pre += subsize[i]
        return pop_id


class Rank_Nich(Niching):
    def __init__(self, Npop) -> None:
        super().__init__()
        self.Npop = Npop

    def nich(self, x, y, subsize=None, rng=None):
        NP = y.shape[0]
        if subsize is None:
            subsize = np.ones(self.Npop) * NP//self.Npop
            if np.sum(subsize) < NP:
                subsize[-1] += NP - np.sum(subsize)
        pop_id = np.zeros(NP)
        rank = np.argsort(y)
        pre = 0
        for i in range(self.Npop):
            pop_id[(pre <= rank) * (rank < pre+subsize[i])] = i
            pre += subsize[i]
        return pop_id


class Distance_Nich(Niching):
    def __init__(self, Npop) -> None:
        super().__init__()
        self.Npop = Npop

    def nich(self, x, y, subsize=None, rng=None):
        NP = y.shape[0]
        if subsize is None:
            subsize = np.ones(self.Npop) * NP//self.Npop
            if np.sum(subsize) < NP:
                subsize[-1] += NP - np.sum(subsize)
        pop_id = np.zeros(NP)
        ban = []
        for i in range(self.Npop):
            for j in range(NP):
                if j not in ban:
                    ban.append(j)
                    break
            dist = np.argsort(np.sum((x[j,None] - x) ** 2, -1))
            pop_id[j] = i
            count = 1
            for k in range(NP):
                if dist[k] not in ban:
                    pop_id[dist[k]] = i
                    ban.append(dist[k])
                    count += 1
                if count >= subsize[i]:
                    break
        return pop_id
    

"""
Population size
"""
class Pop_Size_50(Pop_Size):
    def __init__(self) -> None:
        super().__init__()
        self.pop_size = 50


class Pop_Size_100(Pop_Size):
    def __init__(self) -> None:
        super().__init__()
        self.pop_size = 100


class Pop_Size_200(Pop_Size):
    def __init__(self) -> None:
        super().__init__()
        self.pop_size = 200


"""
Reduced Pop size
"""
class Reduce_Size_5(Reduce_Size):
    def __init__(self) -> None:
        super().__init__()
        self.pop_size = 5


class Reduce_Size_10(Reduce_Size):
    def __init__(self) -> None:
        super().__init__()
        self.pop_size = 10


class Reduce_Size_20(Reduce_Size):
    def __init__(self) -> None:
        super().__init__()
        self.pop_size = 20


"""
Boundary control
"""
class Clip_BC(BC):
    def __init__(self) -> None:
        super().__init__()

    def bc(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        trail = population.trail[population.process_ip]
        return np.clip(trail, population.Xmin, population.Xmax)
    

class Rand_BC(BC):
    def __init__(self) -> None:
        super().__init__()

    def bc(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        trail = population.trail[population.process_ip]
        bounds = [population.Xmin, population.Xmax]
        trail[(trail < bounds[0]) + (trail > bounds[1])] = rng.rand(np.sum((trail < bounds[0]) + (trail > bounds[1]))) * (bounds[1] - bounds[0]) + bounds[0]
        return trail
    

class Periodic_BC(BC):
    def __init__(self) -> None:
        super().__init__()

    def bc(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        trail = copy.deepcopy(population.trail[population.process_ip])
        bounds = [population.Xmin, population.Xmax]
        while np.min(trail) < bounds[0] or np.max(trail) > bounds[1]:
            trail[(trail < bounds[0]) + (trail > bounds[1])] = bounds[0] + (trail[(trail < bounds[0]) + (trail > bounds[1])] - bounds[1]) % (bounds[1] - bounds[0])
        return trail
    

class Reflect_BC(BC):
    def __init__(self) -> None:
        super().__init__()

    def bc(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        trail = population.trail[population.process_ip]
        bounds = [population.Xmin, population.Xmax]
        while np.min(trail) < bounds[0] or np.max(trail) > bounds[1]:
            trail[(trail < bounds[0])] = 2 * bounds[0] - trail[(trail < bounds[0])]
            trail[(trail > bounds[1])] = 2 * bounds[1] - trail[(trail > bounds[1])]
        return trail
    

class Halving_BC(BC):
    def __init__(self) -> None:
        super().__init__()

    def bc(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        trail = population.trail[population.process_ip]
        bounds = [population.Xmin, population.Xmax]
        group = population.group[population.process_ip]
        while np.min(trail) < bounds[0] or np.max(trail) > bounds[1]:
            trail[trail < bounds[0]] = (group[trail < bounds[0]] - bounds[0]) / 2 + bounds[0]
            trail[trail > bounds[1]] = (group[trail > bounds[1]] - bounds[1]) / 2 + bounds[1]
        return trail


"""
Selection
"""
class DE_like(DE_Selection):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None, seed=None):
        if rng is None:
            rng = np.random
        population.evaluation()
        new_x = copy.deepcopy(population.group[population.process_ip])
        new_y = copy.deepcopy(population.cost[population.process_ip])
        trail = population.trail[population.process_ip]
        trail_cost = population.trail_cost[population.process_ip]
        replace_id = np.where(trail_cost < population.cost[population.process_ip])[0]
        replaced_x = new_x[replace_id]
        replaced_y = new_y[replace_id]
        new_x[replace_id] = trail[replace_id]
        new_y[replace_id] = trail_cost[replace_id]
        population.trail[population.process_ip] = new_x
        population.trail_cost[population.process_ip] = new_y
        population.update_archive(replaced_x, replaced_y, rng)
        return population
    

class Crowding(DE_Selection):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None, seed=None):
        if rng is None:
            rng = np.random
        population.evaluation()
        new_x = copy.deepcopy(population.group[population.process_ip])
        new_y = copy.deepcopy(population.cost[population.process_ip])
        trail = population.trail[population.process_ip]
        trail_cost = population.trail_cost[population.process_ip]
        NP, dim = new_x.shape
        replaced_x, replaced_y = [], []
        have_replaced = np.zeros(NP)
        for i in range(NP):
            dist = np.sum((trail[i][None, :] - new_x) ** 2, -1)
            replace_id = np.argmin(dist)
            if trail_cost[i] < new_y[replace_id]:
                if not have_replaced[replace_id]:
                    replaced_x.append(new_x[replace_id])
                    replaced_y.append(new_y[replace_id])
                have_replaced[replace_id] = True
                new_x[replace_id] = trail[i]
                new_y[replace_id] = trail_cost[i]
        replaced_x = np.array(replaced_x)
        population.trail[population.process_ip] = new_x
        population.trail_cost[population.process_ip] = new_y
        population.update_archive(replaced_x, replaced_y, rng)
        return population
    

class PSO_like(PSO_Selection):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None, seed=None):
        if rng is None:
            rng = np.random
        population.evaluation()
        new_x = copy.deepcopy(population.group[population.process_ip])
        new_y = copy.deepcopy(population.cost[population.process_ip])
        trail = population.trail[population.process_ip]
        trail_cost = population.trail_cost[population.process_ip]
        replaced_x = copy.deepcopy(population.group[population.process_ip])
        replaced_y = copy.deepcopy(population.cost[population.process_ip])
        new_x = copy.deepcopy(trail)
        new_y = copy.deepcopy(trail_cost)
        population.trail[population.process_ip] = new_x
        population.trail_cost[population.process_ip] = new_y
        population.update_archive(replaced_x, replaced_y, rng)
        return population
    

class Ranking(GA_Selection):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None, seed=None):
        if rng is None:
            rng = np.random
        population.evaluation()
        new_x = copy.deepcopy(population.group[population.process_ip])
        new_y = copy.deepcopy(population.cost[population.process_ip])
        trail = population.trail[population.process_ip]
        trail_cost = population.trail_cost[population.process_ip]
        NP, dim = new_x.shape
        total = np.concatenate((new_x, trail), 0)
        total_cost = np.concatenate((new_y, trail_cost), 0)
        order = np.argsort(total_cost)
        new_x = total[order[:NP]]
        new_y = total_cost[order[:NP]]
        replace_id = order[NP:]
        replaced_x = population.group[population.process_ip][replace_id[replace_id < NP]]
        replaced_y = population.cost[population.process_ip][replace_id[replace_id < NP]]
        population.trail[population.process_ip] = new_x
        population.trail_cost[population.process_ip] = new_y
        population.update_archive(replaced_x, replaced_y, rng)
        return population
    

class Tournament(GA_Selection):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None, seed=None):
        if rng is None:
            rng = np.random
        population.evaluation()
        new_x = copy.deepcopy(population.group[population.process_ip])
        new_y = copy.deepcopy(population.cost[population.process_ip])
        trail = population.trail[population.process_ip]
        trail_cost = population.trail_cost[population.process_ip]
        NP, dim = new_x.shape
        total = np.concatenate((new_x, trail), 0)
        total_cost = np.concatenate((new_y, trail_cost), 0)
        new_x, new_y = [], []
        indices = np.arange(total.shape[0])
        for i in range(NP):
            ia, ib = rng.choice(indices.shape[0], size=2, replace=False)
            a, b = indices[ia], indices[ib]
            if total_cost[a] < total_cost[b]:
                new_x.append(total[a])
                new_y.append(total_cost[a])
                np.delete(indices, ia)
            else:
                new_x.append(total[b])
                new_y.append(total_cost[b])
                np.delete(indices, ib)
        new_x, new_y = np.array(new_x), np.array(new_y)
        replaced_x = population.group[population.process_ip][indices[indices < NP]]
        replaced_y = population.cost[population.process_ip][indices[indices < NP]]
        population.trail[population.process_ip] = new_x
        population.trail_cost[population.process_ip] = new_y
        population.update_archive(replaced_x, replaced_y, rng)
        return population
    

class Roulette(GA_Selection):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None, seed=None):
        if rng is None:
            rng = np.random
        population.evaluation()
        new_x = copy.deepcopy(population.group[population.process_ip])
        new_y = copy.deepcopy(population.cost[population.process_ip])
        trail = population.trail[population.process_ip]
        trail_cost = population.trail_cost[population.process_ip]
        NP, dim = new_x.shape
        total = np.concatenate((new_x, trail), 0)
        total_cost = np.concatenate((new_y, trail_cost), 0)
        prob = (np.max(total_cost) - total_cost + 1e-8)/(np.sum(np.max(total_cost) - total_cost) + 1e-8)
        prob /= np.sum(prob)
        new_id = rng.choice(total.shape[0], size=NP, replace=False, p=prob)
        new_x = total[new_id]
        new_y = total_cost[new_id]
        replace_id = np.delete(np.arange(NP), new_id[new_id < NP])
        replaced_x, replaced_y = population.group[population.process_ip][replace_id], population.cost[population.process_ip][replace_id]
        population.trail[population.process_ip] = new_x
        population.trail_cost[population.process_ip] = new_y
        population.update_archive(replaced_x, replaced_y, rng)
        return population
    

"""
Restart
"""
class Stagnation(Restart):
    def __init__(self, max_stag=50, eps_y=1e-8, reinit_method=Uniform_Init) -> None:
        super().__init__(reinit_method)
        self.eps_y = eps_y
        self.max_stag = max_stag
        self.stagnation_count = 0
        self.gbest = None

    def check_restart(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        if self.gbest is None:
            self.gbest = population.lbest[population.process_ip]
            return False
        if np.abs(self.gbest - population.lbest[population.process_ip]) < self.eps_y:
            self.stagnation_count += 1
        if self.stagnation_count >= self.max_stag:
            self.stagnation_count = 0
            self.gbest = population.lbest[population.process_ip]
            return True
        return False


class Conver_x(Restart):
    def __init__(self, eps_x=0.005, reinit_method=Uniform_Init) -> None:
        super().__init__(reinit_method)
        self.eps_x = eps_x

    def check_restart(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        dist = np.sqrt(np.sum((np.max(population.trail[population.process_ip], 0) - np.min(population.trail[population.process_ip], 0)) ** 2))
        if dist < self.eps_x * (population.Xmax - population.Xmin) * np.sqrt(population.dim):
            return True
        return False


class Conver_y(Restart):
    def __init__(self, Myeqs=0.25, eps_y=1e-8, reinit_method=Uniform_Init) -> None:
        super().__init__(reinit_method)
        self.eps_y = eps_y
        self.Myeqs = Myeqs

    def check_restart(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        scost = np.sort(population.trail_cost[population.process_ip])[:max(1, int(self.Myeqs * population.trail[population.process_ip].shape[0]))]
        if np.max(scost) - np.min(scost) < self.eps_y:
            return True
        return False


class Conver_xy(Restart):
    def __init__(self, eps_x=0.005, eps_y=1e-8, reinit_method=Uniform_Init) -> None:
        super().__init__(reinit_method)
        self.eps_y = eps_y
        self.eps_x = eps_x

    def check_restart(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        dist = np.sqrt(np.sum((np.max(population.trail[population.process_ip], 0) - np.min(population.trail[population.process_ip], 0)) ** 2))
        if dist < self.eps_x * (population.Xmax - population.Xmin) * np.sqrt(population.dim) and np.max(population.trail_cost[population.process_ip]) - np.min(population.trail_cost[population.process_ip]) < self.eps_y:
            return True
        return False


"""
Reduction
"""
class Linear(Reduction):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        NP, dim = population.trail[population.process_ip].shape
        step_ratio = min(1, population.FEs / population.MaxFEs)
        # step_ratio = population.step_count / population.MaxGen
        num = max(0, NP - int(np.round((population.NPmax[population.process_ip] - population.NPmin[population.process_ip]) * (1 - step_ratio) + population.NPmin[population.process_ip])))
        if num < 1:
            return population
        order = np.argsort(population.trail_cost[population.process_ip])
        removed_id = order[-num:]
        population.reduce_subpop(removed_id)
        return population


class Non_Linear(Reduction):
    def __init__(self) -> None:
        super().__init__()

    def exec(self, population: Population, rng=None):
        if rng is None:
            rng = np.random
        NP, dim = population.trail[population.process_ip].shape
        step_ratio = min(1, population.FEs / population.MaxFEs)
        # step_ratio = population.step_count / population.MaxGen
        num = max(0, NP - int(np.round(population.NPmax[population.process_ip] + (population.NPmax[population.process_ip] - population.NPmin[population.process_ip]) * np.power(step_ratio, 1-step_ratio))))
        if num < 1:
            return population
        order = np.argsort(population.trail_cost[population.process_ip])
        removed_id = order[-num:]
        population.reduce_subpop(removed_id)
        return population



"""
Mutation
"""
class rand1(DE_Mutation):
    allow_qbest=False
    allow_archive=True
    Nxn = 3

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        xn = get_xn(population, self.Nxn, rng)

        NP, dim = population.trail[population.process_ip].shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)

        trail = xn[:,0] + action['F1'] * (xn[:,1] - xn[:,2])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}
    
class rand2(DE_Mutation):
    allow_qbest=False
    allow_archive=True
    Nxn = 5

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             'F2': {'type': 'float', 'range': [0, 1], 'default': 0.5}}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        xn = get_xn(population, self.Nxn, rng)

        NP, dim = population.trail[population.process_ip].shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        trail = xn[:,0] + action['F1'] * (xn[:,1] - xn[:,2]) + action['F2'] * (xn[:,3] - xn[:,4])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class best1(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 2

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             }
        if self.use_qbest:
            self.config_space['q'] = {'type': 'float', 'range': [0, 1], 'default': 0.05}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']

        xn = get_xn(population, self.Nxn, rng)
        if self.use_qbest:
            xqb = get_xqb(population, action['q'], rng=rng)
            trail = xqb + action['F1'] * (xn[:,0] - xn[:,1])
        else:
            trail = population.gbest_solution + action['F1'] * (xn[:,0] - xn[:,1])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}
    
class best2(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 4

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             'F2': {'type': 'float', 'range': [0, 1], 'default': 0.5}}
        if self.use_qbest:
            self.config_space['q'] = {'type': 'float', 'range': [0, 1], 'default': 0.05}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']

        xn = get_xn(population, self.Nxn, rng)
        if self.use_qbest:
            xqb = get_xqb(population, action['q'], rng=rng)
            trail = xqb + action['F1'] * (xn[:,0] - xn[:,1]) + action['F2'] * (xn[:,2] - xn[:,3])
        else:
            trail = population.gbest_solution + action['F1'] * (xn[:,0] - xn[:,1]) + action['F2'] * (xn[:,2] - xn[:,3])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class current2best(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 2

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             'F2': {'type': 'float', 'range': [0, 1], 'default': 0.5}}
        if self.use_qbest:
            self.config_space['q'] = {'type': 'float', 'range': [0, 1], 'default': 0.05}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        xn = get_xn(population, self.Nxn, rng)

        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        if self.use_qbest:
            xqb = get_xqb(population, action['q'], rng=rng)
            trail = group + action['F1'] * (xqb - group) + action['F2'] * (xn[:,0] - xn[:,1])
        else:
            trail = group + action['F1'] * (population.gbest_solution - group) + action['F2'] * (xn[:,0] - xn[:,1])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class current2rand(DE_Mutation):
    allow_qbest=False
    allow_archive=True
    Nxn = 3   

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             'F2': {'type': 'float', 'range': [0, 1], 'default': 0.5}}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        xn = get_xn(population, self.Nxn, rng)
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        trail = group + action['F1'] * (xn[:,0] - group) + action['F2'] * (xn[:,1] - xn[:,2])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class rand2best(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 2

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             }
        if self.use_qbest:
            self.config_space['q'] = {'type': 'float', 'range': [0, 1], 'default': 0.05}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        xn = get_xn(population, self.Nxn, rng)
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)

        if self.use_qbest:
            xqb = get_xqb(population, action['q'], rng=rng)
            trail = xn[:,0] + action['F1'] * (xqb - xn[:,1])
        else:
            trail = xn[:,0] + action['F1'] * (population.gbest_solution - xn[:,1])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class weighted_rand2best(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 2

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.config_space = {'F1': {'type': 'float', 'range': [0, 1], 'default': 0.5},
                             'F2': {'type': 'float', 'range': [0, 1], 'default': 0.5}}
        if self.use_qbest:
            self.config_space['q'] = {'type': 'float', 'range': [0, 1], 'default': 0.05}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        xn = get_xn(population, self.Nxn, rng)
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        if self.use_qbest:
            xqb = get_xqb(population, action['q'], rng=rng)
            trail = action['F1'] * xn[:,0] + action['F1'] * action['F2'] * (xqb - xn[:,1])
        else:
            trail = action['F1'] * xn[:,0] + action['F1'] * action['F2'] * (population.gbest_solution - xn[:,1])

        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

# GA ======================================================================
class gaussian(GA_Mutation):
    def __init__(self, ) -> None:
        super().__init__()
        self.config_space = {'sigma': {'type': 'float', 'range': [0.05, 1], 'default': 0.1}}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']

        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape

        trail = group + rng.normal(0, action['sigma'], size=(NP, dim))
        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class polynomial(GA_Mutation):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {'n': {'type': 'discrete', 'range': np.arange(5)+1, 'default': 2}}

    def __call__(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape

        rvs = rng.rand(NP, dim)
        trail = np.zeros((NP, dim))
        d1 = np.power(2 * rvs, 1/(action['n']+1)) - 1
        d2 = 1 - np.power(2 * (1 - rvs), 1/(action['n']+1))
        C1 = group + d1 * (group - population.Xmin)
        C2 = group + d2 * (population.Xmax - group)

        trail[rvs <= 0.5] = C1[rvs <= 0.5]
        trail[rvs > 0.5] = C2[rvs > 0.5]
        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class random_mutation(GA_Mutation):
    def __init__(self, ) -> None:
        super().__init__()
        self.config_space = {'Cr': {'type': 'float', 'range': [0, 1], 'default': 0.1}}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        rvs = rng.rand(NP, dim)
        rand = rng.rand(NP, dim) * (population.Xmax - population.Xmin) + population.Xmin
        trail = copy.deepcopy(group)
        trail[rvs < action['Cr']] = rand[rvs < action['Cr']]
        population.trail[population.process_ip] = trail
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}


"""
Crossover
"""
# DE ======================================================================
class binomial(DE_Crossover):
    allow_qbest=True
    allow_archive=True
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False) -> None:
        super().__init__(use_qbest, united_qbest, use_archive)
        self.use_archive = self.united_qbest
        self.config_space = {'Cr': {'type': 'float', 'range': [0, 1], 'default': 0.9},}
        if self.use_qbest:
            self.config_space['q'] = {'type': 'float', 'range': [0, 1], 'default': 0.5}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape

        if isinstance(action['Cr'], np.ndarray) and len(action['Cr'].shape) < 2:
            action['Cr'] = action['Cr'][:,None].repeat(dim, -1)

        jrand = rng.randint(dim, size=NP)
        parent = population.group[population.process_ip]
        if self.use_qbest:
            parent = get_qb_parent(population, action['q'], rng=rng)
        u = np.where(rng.rand(NP, dim) < action['Cr'], group, parent)
        u[np.arange(NP), jrand] = group[np.arange(NP), jrand]
        population.trail[population.process_ip] = u
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class exponential(DE_Crossover):
    allow_qbest=True
    allow_archive=True
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False) -> None:
        super().__init__(use_qbest, united_qbest, use_archive)
        self.use_archive = self.united_qbest
        self.config_space = {'Cr': {'type': 'float', 'range': [0, 1], 'default': 0.9},}
        if self.use_qbest:
            self.config_space['q'] = {'type': 'float', 'range': [0, 1], 'default': 0.5}

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        if isinstance(action['Cr'], np.ndarray) and len(action['Cr'].shape) < 2:
            action['Cr'] = action['Cr'][:,None].repeat(dim, -1)
        parent = population.group[population.process_ip]
        if self.use_qbest:
            parent = get_qb_parent(population, action['q'], rng=rng)
        u = parent.copy()
        L = rng.randint(dim-1, size=(NP, 1))
        L = np.repeat(L, dim, -1)
        rvs = rng.rand(NP, dim)
        index = np.repeat([np.arange(dim)], NP, 0)
        rvs[index < L] = -1 # masked as no crossover (left side)
        R = np.argmax(rvs > action['Cr'], -1)
        R[np.max(rvs, -1) < action['Cr']] = dim
        R = np.repeat(R[:,None], dim, -1)
        rvs[index >= R] = -1 # masked as no crossover (right side)
        u[rvs > 0] = group[rvs > 0]
        population.trail[population.process_ip] = u
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

# GA ======================================================================
class sbx(GA_Crossover):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
            'n': {'type': 'discrete', 'range': np.arange(5)+1, 'default': 2},
        }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        
        index1, index2 = parent_selection('rand', population, rng=rng)
        ext = None
        if len(index1) != len(index2):  # odd number population
            if len(index1) > len(index2):
                ext = index1[0]
                index1 = index1[1:]
            else:
                ext = index2[0]
                index2 = index2[1:]
        rvs = rng.rand(NP//2, dim)
        b = np.power(2 * rvs, 1 / (action['n'] + 1))
        b[rvs > 0.5] = np.power(2 * (1 - rvs[rvs > 0.5]), -1 / (action['n'] + 1))
        C1 = 0.5 * (group[index2] + group[index1]) - 0.5 * b * (group[index2] - group[index1])
        C2 = 0.5 * (group[index2] + group[index1]) + 0.5 * b * (group[index2] - group[index1])

        new_x = np.zeros((NP, dim))
        new_x[index1] = C1
        new_x[index2] = C2
        if ext is not None:
            new_x[ext] = group[ext]
        population.trail[population.process_ip] = new_x
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class mpx(GA_Crossover):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
            'Cr': {'type': 'float', 'range': [0, 1], 'default': 0.5},
        }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']

        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        index1, index2 = parent_selection('rand', population, rng=rng)
        ext = None
        if len(index1) != len(index2):  # odd number population
            if len(index1) > len(index2):
                ext = index1[0]
                index1 = index1[1:]
            else:
                ext = index2[0]
                index2 = index2[1:]
        rvs = rng.rand(NP//2, dim)
        C1 = population.trail[population.process_ip][index1]
        C2 = population.trail[population.process_ip][index2]
        C1[rvs < action['Cr']], C2[rvs < action['Cr']] = C2[rvs < action['Cr']], C1[rvs < action['Cr']]

        new_x = np.zeros((NP, dim))
        new_x[index1] = C1
        new_x[index2] = C2
        if ext is not None:
            new_x[ext] = population.trail[population.process_ip][ext]
        population.trail[population.process_ip] = new_x
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class arithmetic(GA_Crossover):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
            'alpha': {'type': 'float', 'range': [0, 1], 'default': 0.5},
        }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        
        index1, index2 = parent_selection('rand', population, rng=rng)
        ext = None
        if len(index1) != len(index2):  # odd number population
            if len(index1) > len(index2):
                ext = index1[0]
                index1 = index1[1:]
            else:
                ext = index2[0]
                index2 = index2[1:]
        C1 = action['alpha'] * group[index1] + (1 - action['alpha']) * group[index2]
        C2 = action['alpha'] * group[index2] + (1 - action['alpha']) * group[index1]

        new_x = np.zeros((NP, dim))
        new_x[index1] = C1
        new_x[index2] = C2
        if ext is not None:
            new_x[ext] = population.trail[population.process_ip][ext]
        population.trail[population.process_ip] = new_x
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

# PSO ======================================================================
class vanilla_PSO(PSO_update):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
                'w': {'type': 'float', 'range': [0, 1], 'default': 0.7},
                'c1': {'type': 'float', 'range': [0, 2], 'default': 1.49445},
                'c2': {'type': 'float', 'range': [0, 2], 'default': 1.49445},
        }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']

        for k in action.keys():
            if isinstance(action[k], np.ndarray) and len(action[k].shape) < 2:
                action[k] = action[k][:,None].repeat(dim, -1)

        gbest_solutions = population.lbest_solution[population.process_ip]
        vel = action['w'] * population.velocity[population.process_ip] + \
              action['c1'] * np.repeat(rng.rand(NP, 1), dim, -1) * (gbest_solutions - group) + \
              action['c2'] * np.repeat(rng.rand(NP, 1), dim, -1) * (population.pbest_solution[population.process_ip] - group)
        vel = np.clip(vel, -population.Vmax, population.Vmax)
        new_x = group + vel
        population.trail[population.process_ip] = new_x
        population.velocity[population.process_ip] = vel
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class FDR_PSO(PSO_update):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
                'w': {'type': 'float', 'range': [0, 1], 'default': 0.729},
                'c1': {'type': 'float', 'range': [0, 2], 'default': 1},
                'c2': {'type': 'float', 'range': [0, 2], 'default': 1},
                'c3': {'type': 'float', 'range': [0, 2], 'default': 2},
        }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']

        for k in action.keys():
            if isinstance(action[k], np.ndarray) and len(action[k].shape) < 2:
                action[k] = action[k][:,None].repeat(dim, -1)

        gbest_solutions = population.lbest_solution[population.process_ip]
        dist = np.sqrt(np.sum((group[None,:,:] - group[:,None,:]) ** 2, -1))
        df = population.cost[population.process_ip].reshape(-1, 1) - population.cost[population.process_ip]
        df[df == 0] += 1e-8
        # df -= -1e9 * np.eye(NP)
        fdr = df / (dist + 1e-8)
        nbest = np.argmax(fdr, -1)
        nbest_solutions = group[nbest]
        vel = action['w'] * population.velocity[population.process_ip] + \
              action['c1'] * np.repeat(rng.rand(NP, 1), dim, -1) * (gbest_solutions - group) + \
              action['c2'] * np.repeat(rng.rand(NP, 1), dim, -1) * (population.pbest_solution[population.process_ip] - group) + \
              action['c3'] * np.repeat(rng.rand(NP, 1), dim, -1) * (nbest_solutions - group)
        vel = np.clip(vel, -population.Vmax, population.Vmax)
        new_x = group + vel
        population.trail[population.process_ip] = new_x
        population.velocity[population.process_ip] = vel
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class CLPSO(PSO_update):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
                'w': {'type': 'float', 'range': [0, 1], 'default': 0.729},
                'c1': {'type': 'float', 'range': [0, 2], 'default': 1},
                'c2': {'type': 'float', 'range': [0, 2], 'default': 1},
        }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']

        for k in action.keys():
            if isinstance(action[k], np.ndarray) and len(action[k].shape) < 2:
                action[k] = action[k][:,None].repeat(dim, -1)

        gbest_solutions = population.lbest_solution[population.process_ip]
        a = 0.05
        b = 0.45
        Pc = a + b * (np.exp(10 * np.arange(NP) / (NP - 1)) - 1) / (np.exp(10) - 1)
        rbest = rng.random(NP) <= Pc
        pbest_solutions = copy.deepcopy(population.pbest_solution[population.process_ip])
        permu = np.argsort(rng.rand(rbest.sum(), NP), -1)
        tournament = population.trail_cost[population.process_ip][permu]
        while permu.shape[-1] > 1:
            # tournament[:,:tournament.shape[-1]//2] = np.maximum(tournament[:,:tournament.shape[-1]//2], tournament[:,-(tournament.shape[-1]//2):])
            # tournament = tournament[:,:-(tournament.shape[-1]//2)]
            permu[:,:tournament.shape[-1]//2] = np.where(tournament[:,:tournament.shape[-1]//2] < tournament[:,-(tournament.shape[-1]//2):], permu[:,:tournament.shape[-1]//2], permu[:,-(tournament.shape[-1]//2):])
            tournament[:,:tournament.shape[-1]//2] = np.maximum(tournament[:,:tournament.shape[-1]//2], tournament[:,-(tournament.shape[-1]//2):])
            permu = permu[:,:-(tournament.shape[-1]//2)]
            tournament = tournament[:,:-(tournament.shape[-1]//2)]
        # pbest_solutions[rbest] = group[tournament.reshape(-1)]
        pbest_solutions[rbest] = group[permu.reshape(-1)]
        vel = action['w'] * population.velocity[population.process_ip] + \
              action['c1'] * np.repeat(rng.rand(NP, 1), dim, -1) * (gbest_solutions - group) + \
              action['c2'] * np.repeat(rng.rand(NP, 1), dim, -1) * (pbest_solutions - group)
        vel = np.clip(vel, -population.Vmax, population.Vmax)
        new_x = group + vel
        population.trail[population.process_ip] = new_x
        population.velocity[population.process_ip] = vel
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

# ES =======================================================================
class LTO(ES_sample):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
                'cc': {'type': 'float', 'range': [0.1, 1.], 'default': 1.},
                'cs': {'type': 'float', 'range': [0.1, 1.], 'default': 1.},
        }
        self.strategy = None
    
    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random.RandomState()
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        np.random.set_state(rng.get_state())
        if self.strategy is None:
            centroid = np.mean(group, 0)
            self.strategy = cma.CMAEvolutionStrategy(centroid, 1., {'popsize': NP,
                                                                       'bounds': [population.Xmin, population.Xmax],
                                                                       'maxfevals': population.MaxFEs,
                                                                       'BoundaryHandler': cma.s.ch.BoundTransform,
                                                                       'verbose': -10,
                                                                       'randn': rng.randn,
                                                                       'seed': 1,}) 
            self.strategy.ask(NP, )
            
        self.strategy.sp.cc = action['cc']
        self.strategy.adapt_sigma.cs = action['cs']
        
        self.strategy.tell(group, population.trail_cost[population.process_ip])

        population.trail[population.process_ip] = np.array(self.strategy.ask(number=NP))
        rng.set_state(np.random.get_state())

        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}


class SepCMAES(ES_sample):
    def __init__(self):
        super().__init__()
        self.config_space = {
                'cc': {'type': 'float', 'range': [0.1, 1.], 'default': 1.},
                'cs': {'type': 'float', 'range': [0.1, 1.], 'default': 1.},
        }
        self.initialized = False
        
    def _set_c_cov(self, dim):
        c_cov = (1.0/self._mu_eff)*(2.0/np.power(dim + np.sqrt(2.0), 2)) + (
            (1.0 - 1.0/self._mu_eff)*np.minimum(1.0, (2.0*self._mu_eff - 1.0)/(
                np.power(dim + 2.0, 2) + self._mu_eff)))
        c_cov *= (dim + 2.0)/3.0  # for faster adaptation
        return c_cov

    def _set_d_sigma(self, dim):
        d_sigma = np.maximum((self._mu_eff - 1.0)/(dim + 1.0) - 1.0, 0.0)
        return 1.0 + self.c_s + 2.0*np.sqrt(d_sigma)

    def initialize(self, x):
        NP, dim = x.shape
        self.n_individuals = NP
        self.n_parents = NP // 2
        w_base, w = np.log((self.n_individuals + 1.0)/2.0), np.log(np.arange(self.n_parents) + 1.0)
        # positive weight coefficients for weighted intermediate recombination (Nikolaus Hansen, 2023)
        #   [assigning different weights should be interpreted as a selection mechanism]
        self._w = (w_base - w)/(self.n_parents*w_base - np.sum(w))
        # variance effective selection mass (Nikolaus Hansen, 2023)
        #   effective sample size of the selected samples
        self._mu_eff = 1.0/np.sum(np.square(self._w))  # μ_eff (μ_w)
        self.c_c = 4.0/(dim + 4.0)
        self.c_s = (self._mu_eff + 2.0)/(dim + self._mu_eff + 3.0)
        self.c_cov = self._set_c_cov(dim)
        self.d_sigma = self._set_d_sigma(dim)
        self._s_1 = 1.0 - self.c_s
        self._s_2 = np.sqrt(self._mu_eff*self.c_s*(2.0 - self.c_s))
        self.mean = np.mean(x, 0)
        self.z = (x - self.mean) / np.std(x, 0)  # np.zeros((NP, dim))
        self.s = np.zeros((dim,))  # evolution path for CSA
        self.p = np.zeros((dim,))  # evolution path for CMA
        self.c = np.ones((dim,))  # diagonal elements for covariance matrix
        self.d = np.ones((dim,))  # diagonal elements for covariance matrix
        self._e_chi = np.sqrt(dim)*(  # E[||N(0,I)||]: expectation of chi distribution
                1.0 - 1.0/(4.0*dim) + 1.0/(21.0*np.square(dim)))
        self.sigma = 1.0

    def iterate(self, x, rng):
        NP, dim = x.shape
        self.z = rng.standard_normal((NP, dim))
        res = self.z * self.d * self.sigma + self.mean        
        return res

    def _update_distribution(self, x, y, n_generations):
        NP, dim = x.shape
        if NP < self.n_individuals:
            self.z = self.z[:NP]
            self.n_individuals = NP
            self.n_parents = NP // 2
        order = np.argsort(y)
        zeros = np.zeros((dim,))
        z_w, self.mean, dz_w = np.copy(zeros), np.copy(zeros), np.copy(zeros)
        
        for k in range(self.n_parents):
            z_w += self._w[k]*self.z[order[k]]
            self.mean += self._w[k]*x[order[k]]  # update distribution mean
            dz = self.d*self.z[order[k]]
            dz_w += self._w[k]*dz*dz
        self.s = self._s_1*self.s + self._s_2*z_w
        if (np.linalg.norm(self.s)/np.sqrt(1.0 - np.power(1.0 - self.c_s, 2.0*(n_generations + 1)))) < (
                (1.4 + 2.0/(dim + 1.0))*self._e_chi):
            h = np.sqrt(self.c_c*(2.0 - self.c_c))*np.sqrt(self._mu_eff)*self.d*z_w
        else:
            h = 0
        self.p = (1.0 - self.c_c)*self.p + h
        self.c = (1.0 - self.c_cov)*self.c + (1.0/self._mu_eff)*self.c_cov*self.p*self.p + (
                self.c_cov*(1.0 - 1.0/self._mu_eff)*dz_w)
        self.sigma *= np.exp(self.c_s/self.d_sigma*(np.linalg.norm(self.s)/self._e_chi - 1.0))
        if np.any(self.c <= 0):  # undefined in the original paper
            cc = np.copy(self.c)
            cc[cc <= 0] = 1.0
            self.d = np.sqrt(cc)
        else:
            self.d = np.sqrt(self.c)

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        group = population.trail[population.process_ip].copy()
        NP, dim = group.shape
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        
        if not self.initialized:
            self.initialize(group)
            self.initialized = True

        self._update_distribution(group, population.trail_cost[population.process_ip], population.step_count, )

        self.c_c = action['cc']
        self.c_s = action['cs']
        self._s_1 = 1.0 - self.c_s
        self._s_2 = np.sqrt(self._mu_eff*self.c_s*(2.0 - self.c_s))
        self.d_sigma = self._set_d_sigma(dim)
        
        population.trail[population.process_ip] = self.iterate(group, rng)
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

      
class MMES(ES_sample):
    def __init__(self):
        super().__init__()
        self.config_space = {
                'cc': {'type': 'float', 'range': [0.1, 1.], 'default': 1.},
                'cs': {'type': 'float', 'range': [0.1, 1.], 'default': 1.},
        }
        self.initialized = False

    def initialize(self, x):
        NP, dim = x.shape
        self.n_individuals = NP
        self.n_parents = NP // 2
        # set number of candidate direction vectors
        self.m = 2*int(np.ceil(np.sqrt(dim)))
        # set learning rate of evolution path
        self.c_c = 0.4/np.sqrt(dim)
        self.ms = 4  # mixing strength (l)
        # set for paired test adaptation (PTA)
        self.c_s = 0.3  # learning rate of global step-size adaptation
        self.a_z = 0.05  # target significance level
        # set minimal distance of updating evolution paths (T)
        self.distance = int(np.ceil(1.0/self.c_c))
        # set success probability of geometric distribution (different from 4/n in the original paper)
        self.c_a = 3.8/dim  # same as the official Matlab code
        self.gamma = 1.0 - np.power(1.0 - self.c_a, self.m)
        self._z_1 = np.sqrt(1.0 - self.gamma)
        self._z_2 = np.sqrt(self.gamma/self.ms)
        self._p_1 = 1.0 - self.c_c
        self._p_2 = np.sqrt(self.c_c*(2.0 - self.c_c))
        self._w_1 = 1.0 - self.c_s
        self._w_2 = np.sqrt(self.c_s*(2.0 - self.c_s))
        self.sigma = 1.
        
        self._n_mirror_sampling = int(np.ceil(NP/2))
        self.mean = np.zeros(dim)
        self.p = np.zeros((dim,))  # evolution path
        self.w = 0.0
        self.q = np.zeros((self.m, dim))  # candidate direction vectors
        self.t = np.zeros((self.m,))  # recorded generations
        self.v = np.arange(self.m)  # indexes to evolution paths
        # unify these following settings in the base class for *consistency* and *simplicity*
        w_base, w = np.log((self.n_individuals + 1.0)/2.0), np.log(np.arange(self.n_parents) + 1.0)
        # positive weight coefficients for weighted intermediate recombination (Nikolaus Hansen, 2023)
        #   [assigning different weights should be interpreted as a selection mechanism]
        self._w = (w_base - w)/(self.n_parents*w_base - np.sum(w))
        # variance effective selection mass (Nikolaus Hansen, 2023)
        #   effective sample size of the selected samples
        self._mu_eff = 1.0/np.sum(np.square(self._w))  # μ_eff (μ_w)
    
    def iterate(self, group, rng):
        NP, dim = group.shape
        self._n_mirror_sampling = int(np.ceil(NP/2))

        nms = self._n_mirror_sampling
        # zq = np.zeros((self._n_mirror_sampling, dim))
        zq = (rng.standard_normal(size=(nms, self.ms)).reshape(nms, self.ms, 1).repeat(dim, -1)*self.q[self.v[(self.m - rng.geometric(self.c_a, size=(nms * self.ms)) % self.m) - 1]].reshape(nms, self.ms, dim)).sum(1)
        z = self._z_1 * rng.standard_normal(size=(nms, dim))
        z += self._z_2 * zq
        group[:nms] = self.mean + self.sigma*z
        group[nms:] = self.mean - self.sigma*z[:NP-nms]
        return group

    def _update_distribution(self, x, n_generations, y):
        NP, dim = x.shape
        y_bak = copy.deepcopy(y)
        self.n_individuals = NP
        self.n_parents = NP // 2
        order = np.argsort(y)[:self.n_parents]
        y.sort()
        mean_w = np.dot(self._w[:self.n_parents], x[order])
        self.p = self._p_1*self.p + self._p_2*np.sqrt(self._mu_eff)*(mean_w - self.mean)/self.sigma
        self.mean = mean_w
        if n_generations < self.m:
            self.q[n_generations] = self.p
        else:
            k_star = np.argmin(self.t[self.v[1:]] - self.t[self.v[:(self.m - 1)]])
            k_star += 1
            if self.t[self.v[k_star]] - self.t[self.v[k_star - 1]] > self.distance:
                k_star = 0
            self.v = np.append(np.append(self.v[:k_star], self.v[(k_star + 1):]), self.v[k_star])
            self.t[self.v[-1]], self.q[self.v[-1]] = n_generations, self.p
        # conduct success-based mutation strength adaptation
        l_w = np.dot(self._w, y_bak[:self.n_parents] > y[:self.n_parents])
        self.w = self._w_1*self.w + self._w_2*np.sqrt(self._mu_eff)*(2*l_w - 1)
        self.sigma *= np.exp(norm.cdf(self.w) - 1.0 + self.a_z)
    
    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        group = population.trail[population.process_ip].copy()
        cost = population.trail_cost[population.process_ip].copy()
        NP, dim = group.shape
        action_value = None
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = action_interpret(self.config_space, logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        else:  # both None -> default
            action = {}
            for param in self.config_space.keys():
                action[param] = self.config_space[param]['default']
        
        if not self.initialized:
            self.initialize(group)
            self.initialized = True
            
        self.c_c = action['cc']
        self.c_s = action['cs']
        self.distance = int(np.ceil(1.0/self.c_c))
        self._p_1 = 1.0 - self.c_c
        self._p_2 = np.sqrt(self.c_c*(2.0 - self.c_c))
        self._w_1 = 1.0 - self.c_s
        self._w_2 = np.sqrt(self.c_s*(2.0 - self.c_s))
        
        self._update_distribution(group, population.step_count, cost)
        population.trail[population.process_ip] = self.iterate(group, rng)
        return {'result': population, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}
    


class Multi_strategy(Multi_strategies):
    def __init__(self, op_list) -> None:
        super().__init__()
        self.ops = op_list
        self.nop = len(op_list)
        self.config_space = {
            'op': {'type': 'discrete', 'range': np.arange(self.nop), 'default': 0},
        }
        # Intersection of all sub operator topo_rules
        self.topo_rule = op_list[0].topo_rule
        self.topo_type = op_list[0].topo_type
        for op in op_list:
            self.topo_rule = list(set(self.topo_rule) & set(op.topo_rule))
            # All sub operators should have the same type
            assert self.topo_type == op.topo_type, 'All sub operators should have the same type.'

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        
        logp = []
        actions = []
        entropy = []
        had_action_rec = {}

        if logits is not None and had_action is None:
            action, action_value, logp1, e = action_interpret(self.config_space, logits[:2], softmax, fixed_action)
            iop = action['op']
            actions.append(iop)
            had_action_rec['op_select'] = iop
            logp += logp1
            entropy += e
            op = self.ops[iop]
            if isinstance(op, Controllable):
                res = op(logits[2:], population, softmax, fixed_action[1:] if fixed_action is not None else None, rng=rng)
                actions += res['actions']
                had_action_rec['op_list'] = [None for _ in range(self.nop)]
                had_action_rec['op_list'][iop] = res['had_action']
                logp += res['logp']
                entropy += res['entropy']
                population = res['result']
            else:
                population = op(population, rng=rng)
        elif had_action is not None:
            iop = had_action['op_select']
            # assert isinstance(iop, int)
            if isinstance(self.ops[iop], Controllable):
                res = self.ops[iop](None, population, softmax, fixed_action=None, rng=rng, had_action=had_action['op_list'][iop])
                population = res['result']
            else:
                population = self.ops[iop](population, rng=rng)
        else:
            iop = rng.randint(self.nop)
            if isinstance(self.ops[iop], Controllable):
                res = self.ops[iop](None, population, softmax, fixed_action=None, rng=rng, had_action=None)
                population = res['result']
            else:
                population = self.ops[iop](population, rng=rng)
        # population.trail[population.process_ip] = trail
        return {'result': population, 'actions': actions, 'had_action': had_action_rec, 'logp': logp, 'entropy': entropy}


"""
Information Sharing
"""
class Comm(Sharing):
    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
            'target': {'type': 'discrete', 'range': [], 'default': 0},
        }

    def exec(self, logits, population: Population, softmax=False, fixed_action=None, rng=None, had_action=None):
        if rng is None:
            rng = np.random
        logp = []
        actions = []
        entropy = []
        nop = population.multiPop
        pid = population.process_ip
        self.config_space['target']['range'] = np.arange(nop)
        if logits is not None and had_action is None:
            action, action_value, logp1, entropy = action_interpret(self.config_space, logits[:2], softmax, fixed_action)
            iop = action['target']
            actions.append(iop)
            logp.append(logp1)
        elif had_action is not None:
            iop = had_action
        else:
            iop = rng.randint(nop)
        if iop != pid:
            tar_x = population.lbest_solution[iop]
            tar_f = population.lbest[iop]
            tar_v = population.lbest_velocity[iop]
            cost = population.trail_cost[pid].copy()
            worst = np.argmax(cost)
            population.trail[pid][worst] = tar_x.copy()
            population.trail[pid][worst] = tar_f
            population.trail[pid][worst] = tar_v.copy()
        
        return {'result': population, 'actions': actions, 'had_action': iop, 'logp': logp, 'entropy': entropy}

################################ SUB-MODULE ################################

# """
# Parameter Adaption
# """        
# class Param_adaption:
#     def __init__(self) -> None:
#         self.history_value = None

#     def reset(self):
#         self.history_value = None

#     def get(self,):
#         pass

#     def update(self, old_cost, new_cost, ratio=None):
#         self.history_value = None


# class SHA(Param_adaption):
#     def __init__(self, size, sample='Cauchy', init_value=0.5, fail_value=-1, sigma=0.1, bound=[0, 1], ubc='clip', lbc='clip', p=2, m=1, self_param_ada=None) -> None:
#         super().__init__()
#         self.size = size
#         self.sample = sample
#         self.sigma = sigma
#         self.bound = bound
#         self.ubc = ubc
#         self.lbc = lbc
#         self.fail_value = fail_value
#         self.init_p = self.p = p
#         self.init_m = self.m = m
#         self.self_param_ada = None if self_param_ada is None else eval(self_param_ada['class'])(*self_param_ada['args'])
#         self.init_value = init_value
#         self.memory = np.ones(size) * self.init_value
#         self.update_index = 0
#         self.history_value = None
#         self.history_id = None

#     def reset(self, size=None):
#         if size is not None:
#             self.size = size
#         self.memory = np.ones(self.size) * self.init_value
#         self.update_index = 0
#         self.history_value = None
#         self.history_id = None
#         self.p = self.init_p
#         self.m = self.init_m

#     def update(self, old_cost, new_cost, ratio=None):
#         old_cost = np.concatenate(old_cost)
#         new_cost = np.concatenate(new_cost)
#         if self.history_value is None:
#             return
#         updated_id = np.where((new_cost < old_cost) * (self.history_value > 1e-8))[0]
#         succ_param = self.history_value[updated_id]
#         d_fitness = (old_cost[updated_id] - new_cost[updated_id]) / (old_cost[updated_id] + 1e-9)
#         if succ_param.shape[0] < 1 or np.max(succ_param) < 1e-8:
#             self.memory[self.update_index] = self.fail_value if self.fail_value > 0 else self.memory[self.update_index]
#         else:
#             w = d_fitness / np.sum(d_fitness)
#             if ratio is not None and self.self_param_ada is not None:
#                 self.p = self.self_param_ada.get(ratio)
#             self.memory[self.update_index] = np.sum(w * (succ_param ** self.p)) / np.sum(w * (succ_param ** (self.p - self.m)))
#         self.update_index = (self.update_index + 1) % self.size
#         self.history_value = None
#         self.history_id = None

#     def get(self, size, ids=None, rng=None):
#         if rng is None:
#             rng = np.random

#         if ids is None:
#             ids = rng.randint(self.size, size=size)

#         mrs = self.memory[ids]

#         def cal_value(mr, sz):
#             if self.sample == 'Cauchy':
#                 values = stats.cauchy.rvs(loc=mr, scale=self.sigma, size=sz, random_state=rng)
#             else:
#                 values = rng.normal(loc=mr, scale=self.sigma, size=sz)
#             return values
        
#         values = cal_value(mrs, size)

#         if self.ubc == 'regenerate' and self.lbc == 'regenerate':
#             exceed = np.where((values < self.bound[0]) + (values > self.bound[1]))[0]
#             while exceed.shape[0] > 0:
#                 values[exceed] = cal_value(mrs[exceed], exceed.shape[0])
#                 exceed = np.where((values < self.bound[0]) + (values > self.bound[1]))[0]
#         elif self.ubc == 'regenerate':
#             exceed = np.where(values > self.bound[1])[0]
#             while exceed.shape[0] > 0:
#                 values[exceed] = cal_value(mrs[exceed], exceed.shape[0])
#                 exceed = np.where(values > self.bound[1])[0]
#         elif self.lbc == 'regenerate':
#             exceed = np.where(values < self.bound[0])[0]
#             while exceed.shape[0] > 0:
#                 values[exceed] = cal_value(mrs[exceed], exceed.shape[0])
#                 exceed = np.where(values < self.bound[0])[0]

#         values = np.clip(values, self.bound[0], self.bound[1])
#         self.history_value = values.copy()
#         self.history_id = ids
#         return values
    
#     def get_ids(self, size, rng=None):
#         if rng is None:
#             rng = np.random
#         ids = rng.randint(self.size, size=size)
#         return ids


# class jDE(Param_adaption):
#     def __init__(self, size, init_value=None, bound=[0.1, 0.9], tau=0.1) -> None:
#         super().__init__()
#         self.size = size
#         self.bound = bound
#         self.tau = tau
#         self.init_value = init_value
#         if init_value is None:
#             self.init_value = (bound[0] + bound[1]) / 2
#         self.memory = np.zeros(size) + self.init_value

#     def reset(self):
#         self.memory = np.zeros(self.size) + self.init_value

#     def update(self, old_cost, new_cost, rng=None):
#         if rng is None:
#             rng = np.random
#         updated_id = np.where(new_cost < old_cost)[0]
#         rvs = rng.rand(len(updated_id))
#         self.memory[updated_id[rvs < self.tau]] = rng.uniform(low=self.bound[0], high=self.bound[1], size=len(updated_id))

#     def get(self, size=None):
#         if size is None:
#             size = self.size
#         return self.memory[:size]
    
#     def reduce(self, reduce_id):
#         assert np.max(reduce_id) < self.size
#         self.memory = np.delete(self.memory, reduce_id)
#         self.size = self.memory.shape[0]

#     def reorder(self, new_order):
#         self.memory = self.memory[new_order]
        

# class Linear(Param_adaption):
#     def __init__(self, init_value, final_value) -> None:
#         super().__init__()
#         self.init_value = init_value
#         self.final_value = final_value

#     def get(self, ratio, size=1):
#         value = self.init_value + ratio * (self.final_value - self.init_value)
#         if size == 1:
#             return value
#         return np.ones(size) * value


# class Bound_rand(Param_adaption):
#     def __init__(self, low, high) -> None:
#         super().__init__()
#         self.low = low
#         self.high = high

#     def get(self, size=1, rng=None):
#         if rng is None:
#             rng = np.random
#         return rng.uniform(low=self.low, high=self.high, size=size)


# # for DMS-PSO
# class DMS(Param_adaption):
#     def __init__(self, init, final, switch_point=0.9) -> None:
#         super().__init__()
#         self.init = init
#         self.final = final
#         self.switch_point = switch_point

#     def get(self, ratio, size=1):
#         if ratio < self.switch_point:
#             value = self.init
#         else:
#             value = self.final
#         if size == 1:
#             return value
#         return np.ones(size) * value

# """
# Operator Selection
# """
# class Probability_rand():
#     def __init__(self, opm, probs) -> None:
#         self.opm = opm
#         self.probs = probs
#         assert len(probs) >= opm
#         self.probs = self.probs[:opm]
#         assert np.sum(probs) > 0
#         self.probs /= np.sum(self.probs)
#         self.init_probs = self.probs.copy()
#         self.history_value = None
    
#     def reset(self):
#         self.probs = self.init_probs.copy()
#         self.history_value = None

#     def get(self, size=1, rng=None):
#         if rng is None:
#             rng = np.random
#         value = rng.choice(self.opm, size=size, p=self.probs)
#         self.history_value = value.copy()
#         return value
    
#     def update(self, old_cost, new_cost, ratio=None):
#         # assert self.history_value is not None
#         # d_fitness = np.zeros(self.opm)
#         # for i in range(self.opm):
#         #     id = np.where(self.history_value == i)[0]
#         #     d_fitness[i] = np.mean(np.maximum(0, (old_cost[id] - new_cost[id]) / (old_cost[id] + 1e-9)))
#         # self.probs = d_fitness / np.sum(d_fitness)
#         # self.history_value = None
#         pass


# class Fitness_rand():
#     def __init__(self, opm, prob_bound=[0.1, 0.9]) -> None:
#         self.opm = opm
#         self.probs = np.ones(opm) / opm
#         self.prob_bound = prob_bound
#         self.history_value = None

#     def reset(self):
#         self.probs = np.ones(self.opm) / self.opm
#         self.history_value = None

#     def get(self, size=1, rng=None):
#         if rng is None:
#             rng = np.random
#         value = rng.choice(self.opm, size=size, p=self.probs)
#         self.history_value = value.copy()
#         return value
    
#     def update(self, old_cost, new_cost, ratio=None):
#         if self.history_value is None:
#             return
#         df = np.maximum(0, old_cost - new_cost)
#         count_S = np.zeros(self.opm)
#         for i in range(self.opm):
#             count_S[i] = np.mean(df[self.history_value == i] / old_cost[self.history_value == i])
#         if np.sum(count_S) > 0:
#             self.probs = np.maximum(self.prob_bound[0], np.minimum(self.prob_bound[1], count_S / np.sum(count_S)))
#             self.probs /= np.sum(self.probs)
#         else:
#             self.probs = np.ones(self.opm) / self.opm
#         self.history_value = None


# class Random_select():
#     def __init__(self, opm) -> None:
#         self.opm = opm

#     def reset(self):
#         pass

#     def get(self, size, rng=None):
#         if rng is None:
#             rng = np.random
#         value = rng.randint(self.opm) * np.ones(size)
#         self.history_value = value.copy()
#         return value
    
#     def update(self, old_cost, new_cost, ratio):
#         pass