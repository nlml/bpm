# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:58:18 2016

Module to estimate the bpm based on a vector of pulses

@author: liam
"""

import numpy as np

def linear_interp_nans(y):
    # Fit a linear regression to the non-nan y values

    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))
    
    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)
    
    # Estimate the coefficients of the linear regression
    beta = np.linalg.lstsq(X_fit.T, y_fit)[0]
    
    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y
    
def vec_to_bpm(vec, verbose=False, resolution=172, seconds=4):
    fps = resolution * 1. / seconds
    a = np.where(vec > 0.01)
    a = np.array(a[0])
    curr, means = [], []
    for idx, i in enumerate(a[:-1]):
        if a[idx] == a[idx + 1] - 1:
            curr.append(a[idx])
        else:
            if verbose:
                print 'curr', curr
            if idx == 0 or idx == (len(a) - 1):
                means.append(a[idx])
            else:
                means.append(np.mean(curr))
            curr = []
    means = linear_interp_nans(np.array(means))
    gap = means[1:] - means[:-1]# - 1\
    gap_in_seconds = gap / fps
    gap_in_seconds = np.mean(gap_in_seconds)
    return 60. / gap_in_seconds