# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:43:26 2020

@author: yooji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import reco


# Load data & 
ratings = pd.read_csv('archive/ratings.csv')
bybook = ratings['rating'].groupby(ratings.book_id).count()
byuser = ratings[['user_id', 'rating']].groupby('user_id').count()

users_culled = byuser[byuser.rating >= 100].index.tolist()
ratings_culled = ratings[ratings.user_id.isin(users_culled)]

fp = open('train_ind.pkl', 'rb')
idx = pickle.load(fp)
fp.close()

fp = open('train_mat.pkl', 'rb')
Y = pickle.load(fp)
fp.close()

fp = open('theta_trained.pkl', 'rb')
theta = pickle.load(fp)
fp.close()

ratings_train = ratings_culled[idx]
ratings_test = ratings_culled[~idx]

R = Y!=0
Ynorm, Ymean = reco.normRatings(Y, R)
num_items, num_users = Y.shape
num_features = 3

X = theta.x[:num_items*num_features].reshape(num_items, num_features)
Theta = theta.x[num_items*num_features:].reshape(num_users, num_features)
p = X.dot(Theta.T) + Ymean.reshape(-1, 1)

# %% Compare training and testing set

users_train = list(set(ratings_train.user_id.tolist()))
users_test = list(set(ratings_test.user_id.tolist()))
print("Number of users in training set: %d" % len(users_train))
print("Number of users in testing set: %d" % len(users_test))

books_train = list(set(ratings_train.book_id.tolist()))
books_test = list(set(ratings_test.book_id.tolist()))
print("Number of books in training set: %d" % len(books_train))
print("Number of books in testing set: %d" % len(books_test))

# %% Remove items that are not in the training set

books_train = np.array(books_train)
books_test = np.array(books_test)

not_in_train = list()

for n in range(len(books_test)):
    tmp = np.argwhere(books_train == books_test[n])
    if tmp.size == 0:
        not_in_train.append(books_test[n])

pivoted_test = ratings_test.pivot_table(index='book_id',
                                        columns='user_id',
                                        values='rating',
                                        fill_value=0)

pivoted_test.drop(not_in_train, inplace=True)
test_mat = np.array(pivoted_test.reindex(books_train, fill_value=0))

# %% Show results
R = test_mat!=0
test_error = np.sqrt(np.sum((p[R]-test_mat[R])**2)/np.sum(R))
print("Test RMSE = %0.2f" % test_error)

R = Y!=0
train_error = np.sqrt(np.sum((p[R]-Y[R])**2)/np.sum(R))
print("Train RMSE = %0.2f" % train_error)