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


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))
    # ====================== YOUR CODE HERE ======================
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    # ============================================================
    return W


def sigmoidGradient(z):
    g = np.zeros(z.shape)
    # ====================== YOUR CODE HERE ======================
    g = sigmoid(z) * (1 - sigmoid(z))
    # =============================================================
    return g


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification. 

    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.

    input_layer_size : int
        Number of features for the input layer. 

    hidden_layer_size : int
        Number of hidden units in the second layer.

    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 

    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).

    y : array_like
        Dataset labels. A vector of shape (m,).

    lambda_ : float, optional
        Regularization parameter.

    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.

    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenatation of
        neural network weights Theta1 and Theta2.

    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.shape[0]

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # add the ones to the begining of the X variables.
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Compute a2 the outputs of the hidden layer
    a2 = utils.sigmoid(a1.dot(Theta1.T))
    # Add row of ones to the beginning of the hidden layer output, used for the bias.
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

    # compute output of outputlayer h(x)
    a3 = utils.sigmoid(a2.dot(Theta2.T))

    y_matrix = y

    temp1 = Theta1
    temp2 = Theta2

    # Add regularization term
    reg_term = (lambda_ / (2 * m)) * \
        (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))

    J = (-1 / m) * np.sum((np.log(a3) * y_matrix) +
                          np.log(1 - a3) * (1 - y_matrix)) + reg_term

    # Backpropogation
    delta_3 = a3 - y_matrix
    delta_2 = delta_3.dot(Theta2)[:, 1:] * sigmoidGradient(a1.dot(Theta1.T))

    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)

    # Add regularization to gradient
    Theta1_grad = (1 / m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]

    Theta2_grad = (1 / m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    global progress
    print("Completed Cost ", progress)
    progress += 1
    return J, grad


#gray_arrays, labels,j = loadImages.load(True)
x_train, y_train, x_val, y_val = loadImages.loadData()

#X_t = np.concatenate([np.ones((X_size, 1)), gray_arrays], axis=1)
# print(X_t)

#labels_int = labels.astype(int)
#X = x_train
#y_t = labels_int
#y = labels_int
# print(y)
# Setup the parameters you will use for this exercise
input_layer_size = 60*60  # 57600  # Input Images of Digits
hidden_layer_size = 100   # 25 hidden units
num_labels = 6         # 10 labels, from 0 to 9
print(len(y_train))
print(y_train)
print('Initializing Neural Network Parameters ...')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate(
    [initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
nn_params = initial_nn_params


# utils.checkNNGradients(nnCostFunction)

#  Check gradients by running checkNNGradients
lambda_ = 1
# utils.checkNNGradients(nnCostFunction, lambda_)

# Also output the costFunction debugging values
#debug_J, _  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x_train, y_train, lambda_)

#print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
#print('(for lambda = 3, this value should be about 0.576051)')

#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options = {'maxfun': 800}

#  You should also try different values of lambda
lambda_ = 3

# Create "short hand" for the cost function to be minimized


def costFunction(p): return nnCostFunction(p, input_layer_size,
                                           hidden_layer_size, num_labels, x_train, y_train, lambda_)


# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction, initial_nn_params,
                        jac=True, method='TNC', options=options)
print("Optimized")
# get the solution of the optimization
nn_params = res.x

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))
'''
Theta1 = np.load("arrays/theta1.npy")
Theta2 = np.load("arrays/theta2.npy")
'''
np.save("arrays/theta1_all_60_100_1600.npy", Theta1)
np.save("arrays/theta2_all_60_100_1600.npy", Theta2)


pred = utils.predict(Theta1, Theta2, x_train)
y = np.zeros(y_train.shape[0])
for i in range(y_train.shape[0]):
    for j in range(len(y_train[i])):
        if (y_train[i][j] == 1):
            y[i] = j

debug_J, _ = nnCostFunction(nn_params, input_layer_size,
                            hidden_layer_size, num_labels, x_train, y_train, lambda_)
print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' %
      (lambda_, debug_J))

print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

pred = utils.predict(Theta1, Theta2, x_val)
y = np.zeros(y_val.shape[0])
for i in range(y_val.shape[0]):
    for j in range(len(y_val[i])):
        if (y_val[i][j] == 1):
            y[i] = j
print('Validation Set Accuracy: %f' % (np.mean(pred == y) * 100))
