import numpy as np
import loadImages


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate(
        [np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate(
        [np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p

def printConfusion(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(matrix[i][j])

def main():
    x_test = np.load("Data/arrays/x_test.npy")
    y_test = np.load("Data/arrays/y_test.npy")

    input_layer_size = 60*60
    hidden_layer_size = 250
    num_labels = 6
    lambda_ = 0.3
    maxParams = np.load("Data/arrays/optimized_lamba_0.300000_hidden_250.000000.npy")
    Theta1 = np.reshape(maxParams[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(maxParams[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    
    testPred = predict(Theta1, Theta2, x_test)
    y = np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        for j in range(len(y_test[i])):
            if (y_test[i][j] == 1):
                y[i] = j
    test_acc = np.mean(testPred == y) * 100
    print('Test Set Accuracy: %f' % (test_acc))
    confusion = np.zeros((num_labels,num_labels))
    for i,pred in enumerate(testPred):
        actual = round(y[i])
        confusion[actual][pred] += 1
    printConfusion(confusion)
    print(("Actual d4 value: %d, predicted value: %d")%(np.sum(y==0),np.sum(testPred==0)))
    print(("Actual d6 value: %d, predicted value: %d")%(np.sum(y==1),np.sum(testPred==1)))
    print(("Actual d8 value: %d, predicted value: %d")%(np.sum(y==2),np.sum(testPred==2)))
    print(("Actual d10 value: %d, predicted value: %d")%(np.sum(y==3),np.sum(testPred==3)))
    print(("Actual d12 value: %d, predicted value: %d")%(np.sum(y==4),np.sum(testPred==4)))
    print(("Actual d20 value: %d, predicted value: %d")%(np.sum(y==5),np.sum(testPred==5)))
    print(y.shape)

if __name__ == "__main__":
    main()