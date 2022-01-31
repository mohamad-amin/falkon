from sklearn import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import falkon

X, Y = datasets.load_boston(return_X_y=True)
num_train = int(X.shape[0] * 0.8)
num_test = X.shape[0] - num_train
shuffle_idx = np.arange(X.shape[0])
np.random.shuffle(shuffle_idx)
train_idx = shuffle_idx[:num_train]
test_idx = shuffle_idx[num_train:]

Xtrain, Ytrain = X[train_idx], Y[train_idx]
Xtest, Ytest = X[test_idx], Y[test_idx]

# convert numpy -> pytorch
Xtrain = torch.from_numpy(Xtrain)
Xtest = torch.from_numpy(Xtest)
Ytrain = torch.from_numpy(Ytrain)
Ytest = torch.from_numpy(Ytest)
# z-score normalization
train_mean = Xtrain.mean(0, keepdim=True)
train_std = Xtrain.std(0, keepdim=True)
Xtrain -= train_mean
Xtrain /= train_std
Xtest -= train_mean
Xtest /= train_std

import IPython; IPython.embed()