import numpy as np

def mse(W, x, pred):
    test_size = np.sum(1-W)
    y = x*(1-W)
    pred = pred*(1-W)
    
    #error = np.linalg.norm(y-pred)**2
    error = np.sum((y-pred)**2)
    return error/(test_size)

def r2(W, x, pred):
    test_size = np.sum(1-W)
    error = mse(W, x, pred)
    
    pred_base = (np.sum(x*(1-W))/test_size)*(1-W)
    baseline = mse(W, x, pred_base)
    
    return 1 - error/baseline

def accuracy(W, x, pred, binary = False):
    test_size = np.sum(1-W)
    if binary: threshold = 0.5
    else: threshold = 0
    
    comp = pred*x*(1-W)
    return np.sum(comp>0)/(test_size)

    
    
def print_metrics(W, x, pred, algorithm = "Unknown", binary = False):
    print('-'*30)
    print('Metrics for ' + algorithm)
    print('MSE: ' + '%.3f' % mse(W, x, pred))
    print('R^2: ' + '%.3f' % r2(W, x, pred))
    print('Acc: ' + '%.3f' % accuracy(W, x, pred, binary))
    print('-'*30)