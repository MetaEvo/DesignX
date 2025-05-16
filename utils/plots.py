from matplotlib import pyplot as plt
import io
import numpy as np
from components.operators import *


def display_optimizer(module: list):
    Npop = 1
    if not isinstance(module[1], list):
        Npop = module[1].Npop
    
    print(module[0])
    print(' |')
    print(' |')
    if Npop > 1:
        print(module[1])
        print(' |' , end='')
        spaces = []
        lengths = []
        module_strs = []
        for subpop in module[2:]:
            module_str = []
            space = 0
            length = 0
            for mod in subpop:
                # if isinstance(mod, Pop_Size) or isinstance(mod, Reduce_Size):
                #     continue
                if isinstance(mod, Multi_strategy):
                    module_str.append('Multi_strategy')
                    module_str.append([])
                    sp = 0
                    for m in mod.ops:
                        sp += len(m.__str__())
                        sp += 4  # space between sub-operators
                        module_str[-1].append(m.__str__())
                    # sp -= 4
                    length += 1
                else:
                    module_str.append(mod.__str__())
                    sp = len(mod.__str__())
                space = max(space, sp)
                length += 1
            spaces.append(space)
            lengths.append(length)
            module_strs.append(module_str)
            
        print('-' * (np.sum(spaces[:-1]) + 8 * (Npop - 1)))
        # print(' |' + ' '*(np.sum(spaces[:-1]) - 1 + 8 * (Npop - 1)) + '|')
        
        for i in range(np.max(lengths)):
            upp = ''
            row = ''
            low = ''
            for j, subpop in enumerate(module_strs):
                if i >= len(subpop):
                    row += ' '*(spaces[j] + 4)
                    upp += ' '*(spaces[j] + 4)
                    if i < np.max(lengths) - 1:
                        lo += ' '*(spaces[j] + 4)
                    continue
                mod = subpop[i]
                if isinstance(mod, list):
                    up = ' |  '
                    mm = ' |  '
                    lo = ''
                    for m in mod:
                        up += ' |' + ' ' * (len(m) - 2) + ' '*4
                        mm += m + ' '*4
                    up = up[:-4]
                    mm = mm[:-4]
                    upp += up
                    row += mm
                    upp += ' ' * int(np.sum(spaces[:(j+1)]) - len(mm) + 8)
                    row += ' ' * int(np.sum(spaces[:(j+1)]) - len(mm) + 8)
                else:
                    upp += ' |' + ' ' * (len(mod) - 2)
                    row += mod
                    upp += ' ' * int(np.sum(spaces[:(j+1)]) - len(mod) + 8)
                    row += ' ' * int(np.sum(spaces[:(j+1)]) - len(mod) + 8)
                if i < np.max(lengths) - 1:
                    lo = ' |'
                    if mod == 'Multi_strategy':
                        lo += '   |'
                        for k in range(len(module_strs[j][i+1]) - 1):
                            lo += '-' * (len(module_strs[j][i+1][k]) + 4)
                        lo += ' ' * (spaces[j] - len(lo))
                    else:
                        lo += ' '*(spaces[j] - 2)
                    low += lo + ' ' * 8
            print(upp)
            print(row)
            print(low)
                    
    else:
        for i, mod in enumerate(module[1]):
            # if isinstance(mod, Pop_Size) or isinstance(mod, Reduce_Size):
            #     continue
            print(mod)
            if i < len(module[1]) - 1:
                print(' |')
                print(' |')
        
