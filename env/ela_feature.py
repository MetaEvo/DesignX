import os
from env.pflacco_v1.classical_ela_features \
import calculate_ela_meta, calculate_ela_distribution,\
        calculate_information_content,calculate_nbc
import math
import numpy as np

select_ela_features = ['ela_meta.lin_simple.intercept',
                       'ela_meta.quad_simple.adj_r2',
                        'ela_meta.lin_w_interact.adj_r2',
                        'ic.m0',
                        'ic.h_max',
                        'ic.eps_ratio',
                        'nbc.nn_nb.mean_ratio',
                        'nbc.dist_ratio.coeff_var',
                        'ela_distr.number_of_peaks'
                        ]

                        # 'ela_meta.quad_w_interact.adj_r2',

def get_ela_feature( Xs, Ys,random_state):
    total_calculation_time_cost = 0
    all_features = []
    
    all_ela_keys = []
    
    # 计算下列特征时 对Y进行归一化
    # 对目标值进行归一化
    Ys = (Ys - Ys.min()) / (Ys.max() - Ys.min() + 1e-15)
    

    ela_meta_full_results = calculate_ela_meta(Xs,Ys)
    total_calculation_time_cost += ela_meta_full_results['ela_meta.costs_runtime']
    for k in ela_meta_full_results.keys():
        if k in select_ela_features:
            v = ela_meta_full_results[k]
            # print(f"{k}: {v}")
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)
            all_ela_keys.append(k)
        # elif k == 'ela_meta.costs_runtime':
        #     print("meta feature costs : ",ela_meta_full_results[k])

    

    ela_ic_full_results = calculate_information_content(Xs,Ys,seed=random_state)
    total_calculation_time_cost += ela_ic_full_results['ic.costs_runtime']
    for k in ela_ic_full_results.keys() :
        if  k in select_ela_features:
            v = ela_ic_full_results[k]
            # print(f"{k}: {v}")
            if v is None:
                v = 0.
            elif math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)
            all_ela_keys.append(k)
        # elif k == 'ic.costs_runtime':
        #     print("ic feature costs : ",ela_ic_full_results[k])

    if Ys.max() - Ys.min() < 1e-8:
        all_features += [1.]
    else:
        ela_dist_full_results = calculate_ela_distribution(Xs,Ys)
        total_calculation_time_cost += ela_dist_full_results['ela_distr.costs_runtime']
        for k in ela_dist_full_results.keys() :
            if  k in select_ela_features:
                v = ela_dist_full_results[k]
                if math.isnan(v):
                    v = 0.
                elif math.isinf(v):
                    v = 1.
                all_features.append(v)
                all_ela_keys.append(k)
            # elif k == 'ela_distr.costs_runtime':
            #     print("dist feature costs : ",ela_dist_full_results[k])

    nbc_full_results = calculate_nbc(Xs,Ys)
    total_calculation_time_cost += nbc_full_results['nbc.costs_runtime']
    for k in nbc_full_results.keys() :
        if k in select_ela_features:
            v = nbc_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)
            all_ela_keys.append(k)
        # elif k == 'nbc.costs_runtime':
        #     print("nbc feature costs : ",nbc_full_results[k])


    return np.array(all_features), # all_ela_keys, total_calculation_time_cost





