# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:54:04 2020

@author: yooji
"""

import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_items, num_features, lam):
    """Compute cost and gradient for collaborative filtering."""
    X = params[:num_items*num_features].reshape(num_items, num_features)
    Theta = params[num_items*num_features:].reshape(num_users, num_features)

    J = np.sum((np.dot(X, Theta.T)*R - Y)**2)/2 +\
        lam/2*(np.sum(Theta**2) + np.sum(X**2))

    X_grad = (X.dot(Theta.T)*R-Y).dot(Theta) + lam*X
    Theta_grad = (X.dot(Theta.T)*R-Y).T.dot(X) + lam*Theta
    new_params = np.hstack([X_grad.flatten(), Theta_grad.flatten()])
    return J, new_params


def normRatings(Y, R):
    """Normalize ratings."""
    Ymean = np.sum(Y, axis=1)/np.sum(R, axis=1)
    Ynorm = Y-Ymean[:, None]*R
    return Ynorm, Ymean
