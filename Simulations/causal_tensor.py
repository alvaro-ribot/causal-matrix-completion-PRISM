from dag_conditioning import *

import math
from tensorly.decomposition import parafac, tucker


"""
The following function creates an mxnxp tensor sampled from a Gaussian DAG after intervening and conditioning
m (rows) is the number of different actions (intervening)
n (columns) is the number different contexts (conditioning)
p is the number of observed variables
"""
def causal_tensor(m, n, gdag, setting: dict[str, list], size = 1, noisy = True):
    p = len(setting['X'])
    q = len(setting['A'])
    r = len(setting['C'])
    
    A = 10*np.random.rand(m, q)
    C = 10*np.random.rand(n, r)
    
    X = np.zeros((m, n, size, p))

    for i, a in enumerate(A):
        interventions = {}
        for idx in range(q): interventions.update({setting['A'][idx]: a[idx]})
        
        for j, c in enumerate(C):
            conditions = {}
            for idx in range(r): conditions.update({setting['C'][idx]: c[idx]})

            # we get "size" samples and then take the average
            X[i,j] = sample_condition_intervention(gdag, conditions, interventions, size, noisy)[:,setting['X']]
        
    X = X.squeeze()  # matrix case or size = 1
    return X


"""
Returns the rank of a tensor using the Candecomp-Parafac decomposition
Warning! the value is approximate, as it depends on the initialization of parafac
"""
def tensor_rank(X, tol = 1e-3, attempts = 3):
    if len(X.shape) == 2: return np.linalg.matrix_rank(X)   # matrix case
    
    max_rank = int(math.prod(X.shape)/max(X.shape))
    for r in range(1, max_rank+1):
        for _ in range(attempts):
            Y = parafac(X, rank = r, n_iter_max=500).to_tensor()
            error = np.max(Y - X)
            if error < tol: return r

    return 0