import numpy as np
import loadImages
import tensorflow as tf


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
    

    x_train, y_train, x_val, y_val = loadImages.loadTensorFlowData()

    model = tf.keras.models.load_model("TensorModels/model_3")
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1)
    predict = model.predict(val_dataset,batch_size=1)
    y=y_val
    predict = np.argmax(predict, axis=1)
    test_acc = np.mean(predict == y) * 100
    print(predict)

    print('Test Set Accuracy: %f' % (test_acc))
    confusion = np.zeros((6,6))
    for i,pred in enumerate(predict):
        actual = round(y[i])
        confusion[actual][pred] += 1
    printConfusion(confusion)
    print(("Actual d4 value: %d, predicted value: %d")%(np.sum(y==0),np.sum(predict==0)))
    print(("Actual d6 value: %d, predicted value: %d")%(np.sum(y==1),np.sum(predict==1)))
    print(("Actual d8 value: %d, predicted value: %d")%(np.sum(y==2),np.sum(predict==2)))
    print(("Actual d10 value: %d, predicted value: %d")%(np.sum(y==3),np.sum(predict==3)))
    print(("Actual d12 value: %d, predicted value: %d")%(np.sum(y==4),np.sum(predict==4)))
    print(("Actual d20 value: %d, predicted value: %d")%(np.sum(y==5),np.sum(predict==5)))
    print(y.shape)

    

if __name__ == "__main__":
    main()