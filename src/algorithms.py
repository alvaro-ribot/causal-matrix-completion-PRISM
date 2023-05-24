import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.kernel_ridge import KernelRidge

# Recall    rows: drugs
#           cols: cancer cell-line

def mean_over_drugs(W, x):    
    mean = np.sum(x*W, axis = 0)/np.sum(W, axis = 0)
    pred = np.outer(np.ones(x.shape[0]), mean)
    
    return pred    

def mean_over_cells(W, x):
    pred = mean_over_drugs(W.T, x.T) 
    return pred.T


###############################################################################
# FIXED EFFECTS
###############################################################################
def naive_fixed_effect_drug(W, x, a = 0):
    x = x*W
    shift = np.sum((x - x[a,:])*W, axis = 1)/np.sum(W, axis = 1)
    mean = (x*W)[a,:]
    pred = np.outer(np.ones(x.shape[0]), mean) + np.outer(shift, np.ones(x.shape[1]))
    return pred

def naive_fixed_effect_cell(W, x, c = 0):
    pred = naive_fixed_effect_drug(W.T, x.T, a = c)
    return pred.T

# this might not be symmetric depending on the missing data pattern
def fixed_effect(W, x):
    x = x*W
    pred = np.zeros(x.shape)
    for a in range(x.shape[0]):
        pred += naive_fixed_effect_drug(W, x, a) * W[a, :]
    pred /= np.sum(W, axis = 0)
    return pred


# Fixed effect + neirest neighbors
def neighbors(W, x, r, i, j):

    ball= []

    for k in range(x.shape[0]):
        if k == i: continue

        inter = W[i,:]*W[k,:]
        if np.sum(inter) == 0: continue

        xi = x[i,:]*inter
        xk = x[k,:]*inter

        if np.linalg.norm((xi - xk)*inter)**2/np.sum(inter) <= r: ball.append(k)
    
    return ball

def doubly_robust_nn(W, x, r1 = 1, r2 = None):
    if r2 == None: r2 = r1

    pred0 = fixed_effect(W, x) # we use this prediction if denominator == 0
    pred = np.zeros(x.shape)

    balls_rows = [-1]*x.shape[0] #distance between rows
    balls_cols = [-1]*x.shape[1] #distance between rows

    x = x*W
    missing = [(i,j) for i in range(W.shape[0]) for j in range(W.shape[1]) if W[i,j] == 0]

    for i,j in missing:
        if balls_rows[i] == -1:
            balls_rows[i] = neighbors(W, x, r1, i, j)
        if balls_cols[j] == -1:
            balls_cols[j] = neighbors(W.T, x.T, r2, j, i)

        '''
        num, den = 0, 0

        for k in balls_rows[i]:
            for l in balls_cols[j]:
                if W[i,l]*W[k,j]*W[k,l] == 1:
                    num += x[i,l] + x[k,j] - W[k,l]
                    den += 1
        
        if den == 0: pred[i,j] = pred0[i,j]
        else: pred[i,j] = num/den
        '''

        Wnn = W[np.ix_(balls_rows[i], balls_cols[j])]
        wi = W[i, balls_cols[j]]
        wj = W[balls_rows[i], j]
        Wnn = Wnn * np.outer(wj, wi)
        if np.sum(Wnn) == 0:
            pred[i,j] = pred0[i,j]
            continue
        
        xnn = x[np.ix_(balls_rows[i], balls_cols[j])]
        xi = x[i, balls_cols[j]]
        xj = x[balls_rows[i], j]
        xnn = (np.outer(xj, np.ones(len(xi))) + np.outer(np.ones(len(xj)), xi) - xnn)

        pred[i,j] = np.sum(xnn*Wnn)/np.sum(Wnn)
        

    return pred

###############################################################################
# Collaborative Filtering
###############################################################################
# cos similarity between rows i and k
def cos_sim(W, x, i, k):
    inter = W[i,:]*W[k,:]
    if np.sum(inter) == 0: return 0
    
    xi = x[i,:]*inter
    xk = x[k,:]*inter
    
    sim = np.dot(xi, xk)/(np.linalg.norm(xi)*np.linalg.norm(xk))
    return sim

# shift from row k to row i
def row_shift(W, x, i, k):
    inter = W[i,:]*W[k,:]
    if np.sum(inter) == 0: return 0
    
    return np.sum((x[i,:]-x[k,:])*inter)/np.sum(inter)


def cf_predict(W, x, pred, i, j, sim, shift, Neighbors = 0, fixed_effect = False):
    similarities = []
    for k in range(x.shape[0]):
        if W[k,j] == 1:
            if np.isnan(sim[i,k]):
                sim[i,k] = cos_sim(W, x, i, k)
                if fixed_effect: shift[i,k] = row_shift(W, x, i, k)
                else: shift[i,k] = 0
                    
            similarities.append((k, sim[i, k]))
            
    if Neighbors > 0:
        similarities = sorted(similarities, key = lambda tup: tup[1], reverse = True)[:Neighbors]

    aux, norm = 0, 0
    for k, s in similarities:
        aux += s*(x[k,j] + shift[i,k])
        norm += np.abs(s)
                
    if norm == 0: pred[i,j] = 0
    else: pred[i,j] = aux/norm
        
    return pred, sim, shift

        
def cf_drugs(W, x, Neighbors = 0, fixed_effect = False):
    x = x*W
    
    pred = np.zeros(x.shape)
    sim = np.empty((len(x), len(x))) #similarity between rows
    sim[:] = np.nan
    shift = np.empty((len(x), len(x))) #shift from row i to row k (when fixed_effects)
    shift[:] = np.nan
    
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if W[i,j] == 0:
                pred, sim, shift = cf_predict(W, x, pred, i, j, sim, shift, Neighbors, fixed_effect)
    return pred

def cf_cells(W, x, Neighbors = 0, fixed_effect = False):
    pred = cf_drugs(W.T, x.T, Neighbors, fixed_effect)
    return pred.T

def np_cells(W, x, Neighbors = 10, fixed_effect = False):
    pred = cf_cells(W, x, Neighbors, fixed_effect)
    return pred

def np_drugs(W, x, Neighbors = 10, fixed_effect = False):
    pred = cf_drugs(W, x, Neighbors, fixed_effect)
    return pred

def cf_fe_cells(W, x, Neighbors = 0):
    pred = cf_drugs(W.T, x.T, Neighbors, fixed_effect = True)
    return pred.T

def cf_fe_drugs(W, x, Neighbors = 0):
    pred = cf_drugs(W, x, Neighbors, fixed_effect = True)
    return pred


###############################################################################
# Synthetic Interventions
###############################################################################
def si_neighbors(W, i, j):
    neighbors = []
    for k in range(W.shape[0]):
        wk = W[k, :]
        wi = W[i, :]
        if np.sum(wk*wi) == np.sum(wi) and wk[j] == 1: neighbors.append(k)
    
    return neighbors
        
def si_cells(W, x, Ridge = True, kernel = 'linear'):
    x = x*W
    pred = np.zeros(x.shape)
    
    if Ridge: alphas = [10**k for k in range(-5,5)]
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if W[i,j] == 0:
                neighbors = si_neighbors(W,i,j)
                A = x[neighbors,:][:,[l for l in range(x.shape[1]) if W[i,l]]]
                B = x[neighbors, j]
                C = x[i, [l for l in range(x.shape[1]) if W[i,l]]]
                if kernel == 'linear':
                    if Ridge: reg = RidgeCV(alphas = alphas, fit_intercept = False).fit(A, B)
                    else: reg = LinearRegression(fit_intercept = False).fit(A, B)
                else:
                    # CV results for rbf computed in the missing square case
                    alpha, gamma, degree = 0.01, 1e-5, 0
                    reg = KernelRidge(alpha = alpha, kernel = kernel, gamma = gamma, degree = degree).fit(A, B)

                pred[i,j] = reg.predict([C])[0]
    return pred

def si_drugs(W, x, Ridge = True):
    pred = si_cells(W.T, x.T, Ridge)
    return pred.T

# Centered SI
def csi_drugs(W, x, Ridge = True):
    mean = mean_over_drugs(W, x)
    x_centered = x - mean
    pred = si_drugs(W, x_centered, Ridge) + mean
    return pred

def csi_cells(W, x, Ridge = True):
    pred = csi_drugs(W.T, x.T, Ridge)
    return pred.T

#Kernel SI
def kernel_si_cells(W, x, kernel = 'rbf'):
    pred = si_drugs(W, x, kernel = kernel)
    return pred

def kernel_si_drugs(W, x, kernel = 'rbf'):
    pred = kernel_si_cells(W.T, x.T, kernel = kernel)
    return pred.T


###############################################################################
# Nuclear Norm Minimization
###############################################################################
def shrink(A, lambdak):
    u, s, v = np.linalg.svd(A, full_matrices = False)
    s = s-lambdak
    s = s*(s>0)
    return u @ np.diag(s) @ v

def fe(W, x, L):
    x = x*W
    n = x.shape[0]
    A = np.zeros((x.shape[0] + x.shape[1], x.shape[0] + x.shape[1]))

    for i in range(x.shape[0]):
        A[i, i] = np.sum(W[i,:])
    for j in range(x.shape[1]):
        A[n + j, n + j] = np.sum(W[:, j])

    A[:n, n:] = W
    A[n:, :n] = W.T
    
    sumcols = np.sum((x-L)*W, axis = 1)
    sumrows = np.sum((x-L)*W, axis = 0)
    b = np.concatenate((sumcols, sumrows))

    result = np.linalg.pinv(A) @ b
    gamma, delta = result[:n], result[n:]
    
    return gamma, delta


def nnm(W, x, lambdak = 1e-4, fixed_effects = False, max_it = 200, tol = 1e-6, L0 = 0):
    x = x*W
    obs = np.sum(W)
    
    it = 0
    #if type(L0) == int and L0 == 0: L0 = W*x
    
    while it < max_it:
        if fixed_effects:
            fe_drugs, fe_cells = fe(W, x, L0)
            aux = (x - np.outer(fe_drugs, np.ones(x.shape[1])) - np.outer(np.ones(x.shape[0]), fe_cells))  
        else: aux = x
        
        L1 = shrink(W*aux + (1-W)*L0, lambdak*obs/2)
        
        if type(L0) == int and L0 == 0: L0 = x*W
        
        if np.linalg.norm(L1-L0, ord = 'fro')**2 < tol*np.linalg.norm(L0, ord = 'fro')**2: break
        else: L0 = L1
        
        it += 1
    
    #print('Convergence in', it, 'iterations for lamda =', lambdak)
    if fixed_effects:
        return L1 + np.outer(fe_drugs, np.ones(x.shape[1])) + np.outer(np.ones(x.shape[0]), fe_cells)
    else:
        return L1
    
def nnm_fe(W, x):
    return nnm(W, x, fixed_effects = True)


###############################################################################
# Synthetic Intervention for the square case (much faster)
###############################################################################


def squared_si_cells(W, x): 
    train = int(np.sum(W[-1,:]))
    y = np.copy(x)
    
    A = y[:train, :train]
    B = y[:train, train:]
    C = y[train:, :train]
    
    
    ridge_reg = RidgeCV(alphas = [10**k for k in range(-10,10)], fit_intercept = False).fit(A, B)
    pred = ridge_reg.predict(C)
    y[train:, train:] = pred
    return y

def squared_si_drugs(W, x):
    pred = squared_si_cells(W.T, x.T)
    return pred.T

# Centered SI
def squared_csi_drugs(W, x):
    mean = mean_over_drugs(W, x)
    x_centered = x - mean
    pred = squared_si_drugs(W, x_centered) + mean
    return pred

def squared_csi_cells(W, x):
    pred = squared_csi_drugs(W.T, x.T)
    return pred.T

# Kernel SI
def squared_kernel_si_cells(W, x, kernel = 'rbf'):
    alpha, gamma, degree = 0.01, 1e-5, 0 # CV
    train = int(np.sum(W[-1,:]))  
    y = np.copy(x)
    
    A = y[:train, :train]
    B = y[:train, train:]
    C = y[train:, :train]
    
    kernel_reg = KernelRidge(alpha = alpha, kernel = kernel, gamma = gamma, degree = degree).fit(A, B)
    pred = kernel_reg.predict(C)
    y[train:, train:] = pred
    return y

def squared_kernel_si_drugs(W, x, kernel = 'rbf'):
    pred = squared_kernel_si_cells(W.T, x.T, kernel = kernel)
    return pred.T


def squared_si_fe(W, x):
    fixe = fixed_effect(W, x)
    pred = fixe + squared_si_cells(W, x - fixe)
    return pred
