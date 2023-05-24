from load_data import *
from metrics import *
from algorithms import *
import json
import datetime

# PARAMETERS:
n = 30 #number of experimetns
exponent = 1.7
squared = True
killer_threshold = 1



for percentage in [0.1, 0.2, 0.5]:
    print('%= ', percentage)

    R2 = {}
    algos = ['mean_over_cells', 'mean_over_drugs', 'fixed_effect',
            'cf_cells', 'cf_drugs', 'np_cells', 'np_drugs', 'cf_fe_cells', 'cf_fe_drugs',
            'squared_si_cells','squared_si_drugs', 'squared_csi_cells', 'squared_csi_drugs',
            'squared_kernel_si_cells', 'squared_kernel_si_drugs',
            'nnm', 'nnm_fe']
    for alg in algos: R2[alg] = []

    seeds = [i for i in range(n)]

    for s in seeds:
        print('Iteration', s)
        x = load_matrix(s, killer_threshold)
        W = observed_entries(x, exponent, squared, percentage)

        for alg in algos:
            print(alg, datetime.datetime.now())
            pred = locals()[alg](W, x)
            R2[alg].append(r2(W, x, pred))

    json.dump(R2, open('R2_squared_' + str(percentage) +'.txt', 'w'))


#json.load(open("R2.txt"))
