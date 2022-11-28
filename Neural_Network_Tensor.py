# used for manipulating directory paths
import os

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

# importing os module
import os

from skimage import color
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

import tensorflow as tf


def sigmoid(x):
    return 1/(1 + np.exp(-x))


X_size = 14284  # 2046 validation 14284 train 2039 valid which are 691200 and 14284 which are 691200
Gray_size = 57600  # 230400


def loadData(directory_path):
    labels_int = np.zeros((X_size))
    gray_arrays = np.zeros((X_size, Gray_size))
    directory_path
    ext = ('.jpg')
    i = 0
    j = 0
    for directory in os.listdir(directory_path):
        newPath = directory_path + '/' + directory
        for file in os.listdir(newPath):
            if file.endswith(ext):
                path = newPath + '/' + file

                img = io.imread(path)
                if (img.size == 691200):
                    imgGray = color.rgb2gray(img)
                    res_img = rescale(imgGray, 0.5, anti_aliasing=False)
                    imgn = np.reshape(res_img, (1, Gray_size), order='F')
                    gray_arrays[i] = imgn
                    type = directory
                    match type:
                        case 'd4':
                            labels_int[i] = 0
                        case 'd6':
                            labels_int[i] = 1
                        case 'd8':
                            labels_int[i] = 2
                        case 'd10':
                            labels_int[i] = 3
                        case 'd12':
                            labels_int[i] = 4
                        case 'd20':
                            labels_int[i] = 5
                else:
                    continue

                i = i + 1
                j = j+1
            else:
                continue
    i = 0

    return gray_arrays, labels_int, j


def main():
    print("TensorFlow version:", tf.__version__)

    gray_arrays_train, labels_int_train, j_train = loadData(
        r"./Data/dice-d4-d6-d8-d10-d12-d20/dice/train")

    x_train = gray_arrays_train
    y_train = labels_int_train

    gray_arrays_test, labels_int_test, j_test = loadData(
        r"./Data/dice-d4-d6-d8-d10-d12-d20/dice/valid")
    x_test = gray_arrays_test
    y_test = labels_int_test

    # Setup the parameters you will use for this exercise
    input_layer_size = 57600  # 240x240 Input Images
    hidden_layer_size = 10000   # 200 hidden units
    num_labels = 6          # 6 labels

    # Build a tf.keras.Sequential model by stacking layers.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(57600, 1)),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_labels)
    ])

    # For each example, the model returns a vector of logits or log-odds scores, one for each class.
    predictions = model(x_train[:1]).numpy()

    # The tf.nn.softmax function converts these logits to probabilities for each class:
    tf.nn.softmax(predictions).numpy()

    # Define a loss function for training using losses.SparseCategoricalCrossentropy,
    # which takes a vector of logits and a True index and returns a scalar loss for each example.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
    loss_fn(y_train[:1], predictions).numpy()

    # Before you start training, configure and compile the model using Keras Model.compile.
    # Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier
    # and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # Use the Model.fit method to adjust your model parameters and minimize the loss:
    model.fit(x_train, y_train, epochs=5)

    # The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".
    model.evaluate(x_test,  y_test, verbose=2)

    # If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    probability_model(x_test[:5])


main()
