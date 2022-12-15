# Scientific and vector computation for python
import numpy as np
from numpy import asarray
from numpy import savetxt

# Plotting library
from matplotlib import pyplot
import matplotlib.image as mpimg
import matplotlib.cm as cm

# Optimization module in scipy
from scipy import optimize
from scipy import misc

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

import loadImages

from skimage import color
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

import os

progress = 1

# ========================== START CODE FROM ASSIGNMENTS ============================

def lrCostFunction(theta, X, y, lambda_):
    # Initialize some useful values
    m = y.size
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    h = utils.sigmoid(X.dot(theta.T))

    temp = theta
    temp[0] = 0

    J = (1/m) * ((-y.dot(np.log(h)))-(1-y).dot(np.log(1-h))) + \
        ((lambda_/(2*m)) * np.sum(np.square(temp)))
    grad = (1 / m) * (h - y).dot(X)
    grad = grad + (lambda_ / m) * temp
    # =============================================================
    return J, grad


def oneVsAll(X, y, num_labels, lambda_):

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))  # 10x401

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    #num_labels = 10
    # y = 5000, filled with 0s-9s
    for c in range(num_labels):
        initial_theta = np.zeros(n + 1)  # 401 x 1
        options = {'maxiter': 1000}
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X, (y == c), lambda_),
                                jac=True,
                                method='CG',
                                options=options)
        all_theta[c] = res.x
    # ============================================================
    return all_theta


def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    p = np.argmax(utils.sigmoid(X.dot(all_theta.T)), axis=1)
    # ============================================================
    return p


def main():
    x_train, y_train, x_val, y_val = loadImages.loadData()
    y = np.zeros(y_train.shape[0])
    for i in range(y_train.shape[0]):
        for j in range(len(y_train[i])):
            if (y_train[i][j] == 1):
                y[i] = j

    num_labels = 6

    lambda_ = 0.3
    all_theta = oneVsAll(x_train, y, num_labels, lambda_)
    pred = predictOneVsAll(all_theta, x_train)
    print('Training Set Accuracy: {:.2f}%'.format(
        np.mean(pred == y) * 100))

# ========================== END CODE FROM ASSIGNMENTS ============================

main()
