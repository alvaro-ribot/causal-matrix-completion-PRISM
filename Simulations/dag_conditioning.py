# Author: Alvaro Ribot
"""
The following functions let us to sample from a Gaussian DAG after conditioning
and intervening on some variables. For the intervened variables, we will sample counterfactually.

The main difference with the causaldag setup is that, after conditioning, the noise variables are non longer
independent, so we need the Covariance matrix instead of just a variance vector.
"""

import numpy as np
import random
import causaldag as cd




"""
Use Schur complement to predict the expected value and variance of a set of
Gaussian random variables after conditioning on some of them
"""
def schur_complement(gdag, conditions : dict[int, float]):
    cond_vars = np.array(list(conditions.keys()))
    cond_values = np.array(list(conditions.values()))
    
    M = gdag.covariance
    other_vars = [i for i in range(len(M)) if i not in cond_vars]
    
    A = M[np.ix_(other_vars, other_vars)]
    B = M[np.ix_(other_vars, cond_vars)]
    C = M[np.ix_(cond_vars, cond_vars)]
    
    E_prior = gdag.means()[np.ix_(other_vars)]
    E_cond = gdag.means()[np.ix_(cond_vars)]
    
    # compute the expected value and variance of X_A after conditioning
    E_post = E_prior + B @ np.linalg.inv(C) @ (cond_values - E_cond) 
    Cov_post = A - B @ np.linalg.inv(C) @ B.T
    
    # extend the quantitites computed before for the variables we condition on
    # (variance will be zero for those variables)
    E = np.zeros(len(M))
    for idx, var in enumerate(other_vars):
        E[var] = E_post[idx]
    for idx, var in enumerate(cond_vars):
        E[var] = cond_values[idx]
        
    Cov = np.zeros(M.shape)
    for idx1, var1 in enumerate(other_vars):
        for idx2, var2 in enumerate(other_vars):
            Cov[var1, var2] = Cov_post[idx1, idx2]
        
    return E, Cov


"""
B is the weight matrix of a Gaussian DAG
we apply some interventions and then sample (counterfactually for the intervened nodes) from the noise distribution
"""
def sample_hardintervention_counterfactual(B, E_noise, Cov_noise, interventions: dict[int, float], size = 1, noisy = True):
    for x in interventions:
        E_noise += B[x,:]*interventions[x]
        B[x,:] = np.zeros(len(B))
    
    if noisy: Eps = np.random.multivariate_normal(E_noise, Cov_noise, size = size)
    else: Eps = [E_noise]*size
    
    I = np.eye(len(B))
    X = Eps @ np.linalg.inv(I-B)
    
    return X

"""
sample from a Gaussian DAG after conditioning and intervening on some variables
"""
def sample_condition_intervention(gdag, conditions, interventions, size = 1, noisy = True):
    E, Cov = schur_complement(gdag, conditions)

    B = gdag.weight_mat
    I = np.eye(len(B))

    #compute expected value and variance of the noise
    E_noise = E @ (I - B)
    Cov_noise = (I-B).T @ Cov @ (I-B)

    X = sample_hardintervention_counterfactual(B, E_noise, Cov_noise, interventions, size, noisy)
    return X