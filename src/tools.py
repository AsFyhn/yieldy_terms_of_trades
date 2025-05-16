import numpy as np, pandas as pd
from numpy import random
from numpy import linalg as la
from scipy import optimize
from scipy.stats import norm
from tabulate import tabulate
import estimation

name = 'Logit'

DOCHECKS = True 

_colorway = [
    '#002b5c',  # Original (deep navy)
    '#5CBEAF',  # Original (teal)
    '#30B2E7',  # Original (bright blue)
    '#DDD08A',  # Original (muted yellow)
    '#014F59',  # Original (dark teal)
    '#DA5E4F',  # Original (warm red)
    '#001A37',  # Original (near-black blue)
    '#8D5524',  # New (rich brown)
    '#A3A3A3',  # New (neutral gray)
    '#FF6F91',  # New (soft coral)
    '#2E8B57',  # New (sea green)
    '#800080',  # New (deep purple)
    '#F4A261',  # New (warm orange)
    '#FFB347'   # New (bright amber)
]
colorway = _colorway + _colorway 

def G(z): 
    Gz = 1. / (1. + np.exp(-z))
    return Gz

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x):

    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.size == N
        assert theta.ndim == 1 
        assert theta.size == K 

    # 0. unpack parameters 
    # (trivial, we are just estimating the coefficients on x)
    beta = theta 
    
    # 1. latent index
    z = x@beta
    Gxb = G(z)
    
    # 2. avoid log(0.0) errors
    h = 1e-8 # a tiny number 
    Gxb = np.fmax(Gxb, h)     # truncate below at 1e-8 
    Gxb = np.fmin(Gxb, 1.0-h) # truncate above at 0.99999999

    ll = (y==1)*np.log(Gxb) + (y==0)*np.log(1.0 - Gxb) 
    return ll

def Ginv(u): 
    '''Inverse logistic cdf: u should be in (0;1)'''
    x = - np.log( (1.0-u) / u )
    return x

def starting_values(y,x): 
    b_ols = la.solve(x.T@x, x.T@y)
    return b_ols*4.0

def predict(theta, x): 
    # the "prediction" is the response probability, Pr(y=1|x)
    yhat = G(x@theta) 
    return yhat 

def sim_data(theta: np.ndarray, N:int): 
    '''sim_data: simulate a dataset of size N with true K-parameter theta

    Args. 
        theta: (K,) vector of true parameters (k=0 will always be a constant)
        N (int): number of observations to simulate 
    
    Returns
        tuple: y,x
            y (float): binary outcome taking values 0.0 and 1.0
            x: (N,K) matrix of explanatory variables
    '''
    
    # 0. unpack parameters from theta
    # (trivial, only beta parameters)
    beta = theta

    K = theta.size 
    assert K>1, f'Only implemented for K >= 2'
    
    # 1. simulate x variables, adding a constant 
    oo = np.ones((N,1))
    xx = np.random.normal(size=(N,K-1))
    x  = np.hstack([oo, xx]);
    
    # 2. simulate y values
    
    # 2.a draw error terms 
    uniforms = np.random.uniform(size=(N,))
    u = Ginv(uniforms)

    # 2.b compute latent index 
    ystar = x@beta + u
    
    # 2.b compute observed y (as a float)
    y = (ystar>=0).astype(float)

    # 3. return 
    return y, x


def average_partial_effect(x, betas, k=1):
    '''
    Compute the average partial effect of a binary variable in the logit model.
    '''
    
    me_lg = np.zeros(x.shape[0])
    # For i in x:
    for i in range(x.shape[0]):
        # Slice the data to get the ith observation
        x_me = x[i,:]
        # Compute the marginal effect for each observation
        me_lg[i] = np.exp(x_me @ betas) / ((1 + np.exp(x_me @ betas))**2) * betas[k] 

    # Average the marginal effects
    avg_marginal_effect = np.mean(me_lg)
    
    return avg_marginal_effect

def logit(y, x):
    # Get initial values for theta (using OLS)
    theta0 = starting_values(y, x)
    
    # 
    logit_res = estimation.estimate(
        q = q,
        theta0=theta0,
        y=y,
        x=x,
        cov_type='Sandwich',#'Sandwich',
        options={'disp':False}
    )
    class params: pass
    res = params
    res.params = logit_res['theta']
    return res

def bootstrap_se_with_fit(x, y, k=1, n_boot=1000, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    n = x.shape[0]
    boot_ames = np.zeros(n_boot)
    
    # Fit original model
    model = logit(y, x)
    ame = average_partial_effect(x, model.params, k)
    
    # Bootstrap
    for b in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        x_boot = x[idx, :]
        y_boot = y[idx]
        model_boot = logit(y_boot, x_boot)
        boot_ames[b] = average_partial_effect(x_boot, model_boot.params, k)
    
    se = np.std(boot_ames, ddof=1)
    return ame, se

def bootstrap_se(x, betas, k=1, n_boot=1000, seed=None):
    '''
    Bootstrap the standard error of the average partial effect.
    
    Parameters:
    - x: NumPy array of predictors (n x p)
    - betas: NumPy array of logit coefficients
    - k: Index of the variable for AME
    - n_boot: Number of bootstrap iterations
    - seed: Random seed for reproducibility
    
    Returns:
    - ame: Original average marginal effect
    - se: Bootstrap standard error
    '''
    if seed is not None:
        np.random.seed(seed)
    
    n = x.shape[0]
    boot_ames = np.zeros(n_boot)
    
    # Original AME
    ame = average_partial_effect(x, betas, k)
    
    # Bootstrap loop
    for b in range(n_boot):
        # Resample indices with replacement
        boot_idx = np.random.choice(n, size=n, replace=True)
        x_boot = x[boot_idx, :]
        # Compute AME for bootstrap sample
        boot_ames[b] = average_partial_effect(x_boot, betas, k)
    
    # Standard error is the standard deviation of bootstrap estimates
    se = np.std(boot_ames, ddof=1)
    
    return ame, se

