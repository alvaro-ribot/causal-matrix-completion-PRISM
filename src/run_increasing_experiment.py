from load_data import *
from metrics import *
from algorithms import *
import json
import datetime

# PARAMETERS:
n = 20 #number of experimetns
exponent = 1.7
squared = True


def CV(x, p, fe = False, lambdas = [10**i for i in range(-4, 0)], K = 5):

    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    W = observed_entries(x, squared = True, percentage = p)
    
    r2max = -3
    lmax = 0
    
    W2 = [np.zeros(W.shape)]*K
    prob = np.sum(W)/(x.shape[0]*x.shape[1])

    for k in range(K):
        W2[k] = np.random.binomial(1, prob, W.shape)*W

    
    for l in lambdas:
        avg = 0
        for k in range(K):

            pred = nnm(W2[k], x, lambdak = l, fixed_effects = fe, max_it = 100, tol = 1e-6)
            avg += r2(W2[k] + (1-W), x, pred)
        avg /= K
        print(l,avg)
        if avg > r2max:
            r2max = avg
            lmax = l
    
    print(lmax, r2max)
    return lmax





algos = ['mean_over_cells', 'mean_over_drugs', 'fixed_effect', 'squared_si_cells', 'squared_csi_cells', 'squared_csi_drugs','squared_si_fe', 'nnm', 'nnm_fe']


for alg in algos:
    R2, MSE = {}, {}

    print('-'*20)
    print(alg)
    print('-'*20)

    for columns in range(5, 280, 5):
    #for columns in range(1, 31): # for low_data regime
        print(columns, datetime.datetime.now())

        R2[str(columns)] = []
        MSE[str(columns)] = []

        seeds = [i for i in range(n)]

        if alg in ['nnm', 'nnm_fe']:
            x = load_matrix(1234, killer_threshold = 1)
            x = x[:columns + 284, :columns + 284]
            if 'fe' in alg: lmax = CV(x, columns/(columns + 284), fe = True)
            else: lmax = CV(x, columns/(columns + 284), fe = False)

        for s in seeds:
            #print('Iteration', s)
            x = load_matrix(s, killer_threshold = 1)
            x = x[:columns + 284, :columns + 284]

            W = np.ones(x.shape)
            W[columns:, columns:] = 0

            if alg == 'nnm': pred = nnm(W, x, lambdak = lmax, fixed_effects = False)
            elif alg == 'nnm_fe': pred = nnm(W, x, lambdak = lmax, fixed_effects = True)
            else: pred = locals()[alg](W, x)

            R2[str(columns)].append(r2(W, x, pred))
            MSE[str(columns)].append(mse(W, x, pred))

    json.dump(R2, open('R2_low_' + str(alg) +'.txt', 'a'))
    json.dump(MSE, open('MSE_low_' + str(alg) +'.txt', 'a'))


#json.load(open("R2.txt"))
