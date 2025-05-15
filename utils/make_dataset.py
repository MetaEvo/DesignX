from utils.utils import *
from torch.utils.data import Dataset
import numpy as np
from components.operators import *
from env.basic_problem import Synthetic_Dataset


module_dict = {
    'Uncontrollable': {
        'Initialization': [
            {'class': 'Gaussian_Init', 'param': {}},
            {'class': 'Sobol_Init', 'param': {}},
            {'class': 'LHS_Init', 'param': {}},
            {'class': 'Halton_Init', 'param': {}},
            {'class': 'Uniform_Init', 'param': {}},
        ],
        'Niching': [
            {'class': 'Rand_Nich', 'param': {'Npop': 2}},
            {'class': 'Rank_Nich', 'param': {'Npop': 2}},
            {'class': 'Distance_Nich', 'param': {'Npop': 2}},
            {'class': 'Rand_Nich', 'param': {'Npop': 3}},
            {'class': 'Rank_Nich', 'param': {'Npop': 3}},
            {'class': 'Distance_Nich', 'param': {'Npop': 3}},
            {'class': 'Rand_Nich', 'param': {'Npop': 4}},
            {'class': 'Rank_Nich', 'param': {'Npop': 4}},
            {'class': 'Distance_Nich', 'param': {'Npop': 4}},
        ],
        'BC': [
            {'class': 'Clip_BC', 'param': {}},
            {'class': 'Rand_BC', 'param': {}},
            {'class': 'Periodic_BC', 'param': {}},
            {'class': 'Reflect_BC', 'param': {}},
            {'class': 'Halving_BC', 'param': {}},
        ],
        'Selection': [
            {'class': 'DE_like', 'param': {}},
            {'class': 'Crowding', 'param': {}},
            {'class': 'PSO_like', 'param': {}},
            {'class': 'Ranking', 'param': {}},
            {'class': 'Tournament', 'param': {}},
            {'class': 'Roulette', 'param': {}},
        ],
        'Restart': [
            {'class': 'Stagnation', 'param': {}},
            {'class': 'Conver_x', 'param': {}},
            {'class': 'Conver_y', 'param': {}},
            {'class': 'Conver_xy', 'param': {}},
        ],
        'Reduction': [
            {'class': 'Linear', 'param': {}},
            {'class': 'Non_Linear', 'param': {}},
        ],
        'Termination': [{'class': 'Termination', 'param': {}}],
        'Pop_Size': [
            {'class': 'Pop_Size_50', 'param': {}},
            {'class': 'Pop_Size_100', 'param': {}},
            {'class': 'Pop_Size_200', 'param': {}},
        ],
        'Reduce_Size': [
            {'class': 'Reduce_Size_5', 'param': {}},
            {'class': 'Reduce_Size_10', 'param': {}},
            {'class': 'Reduce_Size_20', 'param': {}},
        ],
    },
    'Controllable': {
        'Mutation': [
            {'class': 'rand1', 'param': {}},
            {'class': 'rand2', 'param': {}},
            {'class': 'best1', 'param': {}},
            {'class': 'best2', 'param': {}},
            {'class': 'current2best', 'param': {}},
            {'class': 'current2rand', 'param': {}},
            {'class': 'rand2best', 'param': {}},
            {'class': 'current2best', 'param': {'use_qbest': True, }},  # current-to-pbest
            {'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},  # current-to-pbest/1+archive
            {'class': 'weighted_rand2best', 'param': {'use_qbest': True}},  # weighted-rand-to-pbest
            {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]}},  # current-to-rand/1+archive
            {'class': 'rand2best', 'param': {'use_qbest': True}},
            {'class': 'best2', 'param': {'use_qbest': True,}},
            {'class': 'rand2', 'param': {'use_archive': True, 'archive_id': [4]}},
            {'class': 'rand2best', 'param': {'use_qbest': True,}},
            {'class': 'rand2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},
            {'class': 'gaussian', 'param': {}},
            {'class': 'polynomial', 'param': {}},
            {'class': 'random_mutation', 'param': {}},
        ],
        'Crossover': [
            {'class': 'binomial', 'param': {}},
            {'class': 'exponential', 'param': {}},
            {'class': 'binomial', 'param': {'use_qbest': True, }},  # qbest-Binomial
            {'class': 'binomial', 'param': {'use_qbest': True, 'united_qbest': True, 'use_archive': True}},  # qbest-Binomial+archive
            {'class': 'sbx', 'param': {}},
            {'class': 'arithmetic', 'param': {}},
            {'class': 'mpx', 'param': {}},
        ],
        'PSO_update': [
            {'class': 'vanilla_PSO', 'param': {}},
            {'class': 'FDR_PSO', 'param': {}},
            {'class': 'CLPSO', 'param': {}},
        ],
        'ES_sample':[
            {'class': 'LTO', 'param': {}},
            {'class': 'SepCMAES', 'param': {}},
            {'class': 'MMES', 'param': {}},
        ],
        'Multi_strategy': [
            {'class': 'Multi_strategy', 'param': {'op_list':   # Multi_BC
                                           [{'class': 'Clip_BC', 'param': {}},
                                            {'class': 'Rand_BC', 'param': {}},
                                            {'class': 'Periodic_BC', 'param': {}},
                                            {'class': 'Reflect_BC', 'param': {}},
                                            {'class': 'Halving_BC', 'param': {}},],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_1
                                            [{'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},  # current-to-pbest/1+archive
                                             {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]}},  # current-to-rand/1+archive
                                             {'class': 'weighted_rand2best', 'param': {'use_qbest': True}}],}},  # weighted-rand-to-pbest
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_2
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'rand2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_3
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_4
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_5
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'best1', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_6
                                            [{'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_7
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_8
                                            [{'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_9
                                            [{'class': 'current2best', 'param': {}},
                                             {'class': 'rand2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_10
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_11
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {'use_qbest': True,}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_12
                                            [{'class': 'current2best', 'param': {'use_qbest': True, }},  # current-to-pbest
                                             {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]},}],}},  # current-to-rand/1+archive
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_13
                                            [{'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},  # current-to-pbest/1+archive
                                             {'class': 'rand1', 'param': {}},
                                             {'class': 'best1', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_14
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'rand2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_15
                                            [{'class': 'best1', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_16
                                            [{'class': 'rand2best', 'param': {'use_qbest': True}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_17
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {'use_qbest': True,}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_18
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'rand2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_19
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {'use_qbest': True,}},
                                             {'class': 'rand2best', 'param': {'use_qbest': True,},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_20
                                            [{'class': 'rand2', 'param': {'use_archive': True, 'archive_id': [4]}},
                                             {'class': 'best2', 'param': {'use_qbest': True,},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_21
                                            [{'class': 'rand2best', 'param': {}},
                                             {'class': 'rand2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_22
                                            [{'class': 'rand2', 'param': {'use_archive': True, 'archive_id': [4]}},
                                             {'class': 'best2', 'param': {'use_qbest': True,}},
                                             {'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},
                                             {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_23
                                            [{'class': 'best1', 'param': {}},
                                             {'class': 'best2', 'param': {},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_24
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'rand2', 'param': {},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_25
                                            [{'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_26
                                            [{'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_27 GA
                                            [{'class': 'polynomial', 'param': {}},
                                             {'class': 'gaussian', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_28 GA
                                            [{'class': 'polynomial', 'param': {}},
                                             {'class': 'random_mutation', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_29 GA
                                            [{'class': 'random_mutation', 'param': {}},
                                             {'class': 'gaussian', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_30 GA
                                            [{'class': 'polynomial', 'param': {}},
                                             {'class': 'random_mutation', 'param': {}},
                                             {'class': 'gaussian', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_1
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_2
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'binomial', 'param': {'use_qbest': True, 'united_qbest': True, 'use_archive': True}}],}},  # qbest-Binomial+archive
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_3
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'binomial', 'param': {'use_qbest': True, },}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_4
                                            [{'class': 'binomial', 'param': {'use_qbest': True, }},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_5
                                            [{'class': 'binomial', 'param': {'use_qbest': True, 'united_qbest': True, 'use_archive': True}},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_6
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'binomial', 'param': {'use_qbest': True}},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_7 GA
                                            [{'class': 'sbx', 'param': {}},
                                             {'class': 'arithmetic', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_8 GA
                                            [{'class': 'sbx', 'param': {}},
                                             {'class': 'mpx', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_9 GA
                                            [{'class': 'mpx', 'param': {}},
                                             {'class': 'arithmetic', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_10 GA
                                            [{'class': 'sbx', 'param': {}},
                                             {'class': 'mpx', 'param': {}},
                                             {'class': 'arithmetic', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_1
                                            [{'class': 'FDR_PSO', 'param': {}},
                                             {'class': 'CLPSO', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_2
                                            [{'class': 'vanilla_PSO', 'param': {}},
                                             {'class': 'CLPSO', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_3
                                            [{'class': 'vanilla_PSO', 'param': {}},
                                             {'class': 'FDR_PSO', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_4
                                            [{'class': 'vanilla_PSO', 'param': {}},
                                             {'class': 'CLPSO', 'param': {}},
                                             {'class': 'FDR_PSO', 'param': {}}],}},
        ],
        'Sharing': [
            {'class': 'Comm', 'param': {}},
        ],
    }
}


class Module_pool():
    def __init__(self, module_dict=module_dict, ban_range=[]) -> None:
        self.module_dict = module_dict  # data backup
        self.ban_range = ban_range  # ignore some modules in topo_type
        self.pool = {}  # topo_type: list [Module ]
        self.mod_list = []  # for algorithm construction from actions
        self.id_count = {}  # count the id of modules in each topo_type
        self.N = 0  # total number of modules
        if module_dict is not None:
            for cab in module_dict.keys():
                for mod in module_dict[cab].keys():
                    for submod_dict in module_dict[cab][mod]:
                        submod = submod_dict['class']
                        if mod == 'Multi_strategy':
                            op_list = []
                            for op_dict in submod_dict['param']['op_list']:
                                op = op_dict['class']
                                sub_item = eval(op)(**op_dict['param'])
                                op_list.append(sub_item)
                            item = eval(mod)(op_list)
                        else:
                            item = eval(submod)(**submod_dict['param'])
                        topo_type = item.topo_type
                        if item.mod_type not in self.id_count.keys():
                            self.id_count[item.mod_type] = 0
                        self.id_count[item.mod_type] += 1
                        if item.__class__.__name__ in ban_range or item.topo_type in ban_range:
                            continue
                        item.id.append(self.id_count[item.mod_type])
                        if topo_type not in self.pool.keys():
                            self.pool[topo_type] = []
                        item.pool_id = self.N + 1
                        self.pool[topo_type].append(item)
                        self.mod_list.append(item)
                        self.N += 1
            print(f'Load {self.N} sub-modules.')

    def register(self, dict):
        pass

    def get(self, topo_type, rng=None) -> Module:
        if rng is None:
            rng = np.random
        item = rng.choice(self.pool[topo_type])
        return item
    
    def get_mask(self, topo_rule):
        mask = torch.zeros(self.N, dtype=torch.bool)
        for mod in topo_rule:
            for submod in self.pool[mod]:
                mask[submod.pool_id-1] = True
        return mask
    
    def action_to_type(self, action):
        return self.mod_list[action].topo_type
    
    def actions_to_types(self, actions):
        types = []
        for act in actions:
            types.append(self.action_to_type(act))
        return types
    
    def action_to_mod(self, action):
        return self.mod_list[action]
    
    def actions_to_mods(self, actions):
        mods = []
        for act in actions:
            mods.append(self.action_to_mod(act))
        return mods
    
    def action_to_algorithm(self, actions):
        pass
    

class Taskset(Dataset):
    def __init__(self,
                 data,
                 batch_size=16):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(train_set, test_set,
                     train_batch_size=1,
                     test_batch_size=1,
                     ):
        # with open(path, 'rb') as f:
        #     data = pickle.load(f)
        # train_set, test_set = data['train'], data['test']
        return Taskset(train_set, train_batch_size), Taskset(test_set, test_batch_size)

    def __getitem__(self, item):
        # if self.batch_size < 2:
        #     return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Taskset'):
        return Taskset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)


def make_synthetic_dataset():
    return Synthetic_Dataset.get_datasets(train_batch_size=128, test_batch_size=128)


def get_test_problems():
    _, test_problem = make_synthetic_dataset()
    pid = np.array([0, 78, 124, 153, 210, 239, 325, 410, 544, 1044, 1138, 1199, 1555, 1652, 1686, 2067, 2389, 2472, 2894, 2985])
    return np.array(test_problem.data)[pid]