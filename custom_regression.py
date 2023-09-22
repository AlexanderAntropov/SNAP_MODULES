#https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/

from multiprocessing import allow_connection_pickling
import sys
import numpy as np
from monty.serialization import loadfn
from maml.utils import pool_from, convert_docs
import json
import os
import random
import subprocess
import glob, shutil

from scipy.optimize import minimize
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from time import time

np.set_printoptions(precision=4)

### Лист весов регуляризации (если это регуляризация к нулю) можно подавать как только для коэффициентов бета, так и для св.члена+коэффициенты бетта. Он сам поймет по его длине.
### Лист фиксированных таргетов (если они не должны быть нулевыми) можно подавать также как только для коэффициентов бета, так и для св.члена+коэффициенты бетта. 
### Длина этих листов должна совпадать, если поданы оба.

def my_mean_square_error(y_pred, y_true, sample_weight):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
        

    sample_weight = np.array(sample_weight)
    assert len(sample_weight) == len(y_true)
    return(np.dot(sample_weight, (np.square((y_true - y_pred)))))
    
loss_function = my_mean_square_error

def custom_ridge(X, y, sample_weight, regular_attraction, target_regular_list=None):
    t1 = time()
    from os import environ
    environ['OMP_NUM_THREADS'] = '64'
    """Ridge Regression model with intercept term.
    L2 penalty and intercept term included via design matrix augmentation.
    This augmentation allows for the OLS estimator to be used for fitting.
    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response
        l2 - L2 penalty tuning parameter (positive scalar) 
    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    if len(regular_attraction)!=len(target_regular_list):
        raise IOError("not consistent len(target_regular_list) and len(regular_attraction_list)")
    #XX = X
    XX = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    #print(XX)
    C = np.diag(np.array(sample_weight))
    #print(C)
    #print(X)
    m, n = np.shape(XX)
    #upper_half = np.hstack((np.ones((m, 1)), X))
    diag = np.zeros((n, n))
    if len(regular_attraction)==n-1:
        full_regular_attraction = np.concatenate((np.array([0.0]), regular_attraction), axis=0)
        target_regular_list = np.concatenate((np.array([0.0]), target_regular_list), axis=0)
    else:
        full_regular_attraction =  regular_attraction
    #print(full_regular_attraction)
    np.fill_diagonal(diag, full_regular_attraction)
    #print(diag)
    H = np.dot(XX.T, np.dot(C,XX))
    Hreg = H + diag
    #print(diag)
    #print(target_regular_coefs)
    if type(target_regular_list)==type(None):
        right = np.dot(XX.T, np.dot(C,y))
    else:
        right = np.dot(XX.T, np.dot(C,y)) + np.dot(diag, target_regular_list)       

    #lower_half = np.hstack((np.zeros((n, 1)), lower))
    #X =  np.vstack((upper_half, lower_half))
    #y = np.append(y, np.zeros(n))
    #print(X)
    res = np.linalg.solve(Hreg, right)
    return res

class FlexiSNAPLinearRegression:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """
    def __init__(self, regular_attraction, target_regular_list=None, loss_function=my_mean_square_error):
        self.regular_attraction = regular_attraction
        self.beta = None
        self.loss_function = loss_function
        self.target_regular_list = target_regular_list

    def predict(self, X):
        X = np.array(X)
        XX = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        prediction = np.matmul(XX, self.beta)
        return(prediction)            
    
    def fit(self, X=None, y=None, sample_weight=None, maxiter=250):
        from os import environ
        environ['OMP_NUM_THREADS'] = '128'
        res = custom_ridge(X, y, sample_weight, regular_attraction = self.regular_attraction, target_regular_list=self.target_regular_list) 
        self.beta = res
        self.coef_ = self.beta[1:]
        self.intercept_ = self.beta[0]
