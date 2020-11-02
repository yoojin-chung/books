# -*- coding: utf-8 -*-
"""
Book recommender system.

Created on Fri Oct 30 10:45:06 2020
@author: yooji
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from scipy.optimize import minimize
import reco


# %% Load data
ratings = pd.read_csv('archive/ratings.csv')
ratings['rating'] = ratings.rating.astype('uint8')
ratings['book_id'] = ratings.book_id.astype('uint16')
ratings['user_id'] = ratings.user_id.astype('uint16')
print(ratings)

# %% Cull data and divide to train/test sets
bybook = ratings['rating'].groupby(ratings.book_id).count()
byuser = ratings[['user_id', 'rating']].groupby('user_id').count()

users_culled = byuser[byuser.rating >= 100].index.tolist()
ratings_culled = ratings[ratings.user_id.isin(users_culled)]

idx = np.random.rand(len(ratings_culled)) < 0.8
ratings_train = ratings_culled[idx]
ratings_test = ratings_culled[~idx]

F = open('train_ind.pkl', 'wb')
pickle.dump(idx, F)
F.close()

pivoted_train = ratings_train.pivot_table(index='book_id',
                                          columns='user_id',
                                          values='rating',
                                          fill_value=0)
ax = sns.heatmap(pivoted_train.sample(100))

# %% Normalize data, initialize parameters
train_mat = np.array(pivoted_train, dtype='float16')
Y = train_mat
R = Y!=0

Ynorm, Ymean = reco.normRatings(Y, R)
num_movies, num_users = Y.shape
num_features = 3

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

init_params = np.hstack([X.flatten(), Theta.flatten()])
lam = 1
X.shape
num_movies, num_users

# %%
theta = minimize(reco.cofiCostFunc,
                 x0=init_params,
                 args=(Ynorm, R, num_users, num_movies, num_features, lam),
                 method='TNC',
                 jac=True)

F = open('theta_trained.pkl', 'wb')
pickle.dump(theta, F)
F.close()