import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2 as cv
import loadImages
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def main():
    print("TensorFlow version:", tf.__version__)
    
    x_train, y_train, x_val, y_val = loadImages.loadTensorFlowDataRGB()
    
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_val, y_val = unison_shuffled_copies(x_val, y_val)
    inputs = keras.Input(shape=(60,60,3))
    x = layers.Rescaling(1.0/255)(inputs)
    x = layers.Conv2D(filters=32,kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32,kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(6, activation="softmax")(x)
    model = keras.Model(inputs,outputs)
    model.summary()

    model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
    batch_size = 32

    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='./TensorModels/model_{epoch}',
        save_freq='epoch')
    ]

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    print("Fit on Dataset")
    history = model.fit(dataset, epochs=10)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1)
    #history = model.fit(dataset, epochs=1, validation_data=val_dataset)
    loss, acc = model.evaluate(val_dataset,batch_size=1)  # returns loss and metrics
    print("loss: %.2f" % loss)
    print("acc: %.2f" % acc)
    model.save("./TensorModels/model_4")
    predict = model.predict(val_dataset,batch_size=1)
    print(y_val[0:8])
    print(np.argmax(predict[0:8], axis=1))
    print(y_val[880:888])
    print(np.argmax(predict[880:888], axis=1))
    print(y_val[0])
    cv.imshow("random", x_val[0])
    cv.waitKey(100000)

if __name__ == "__main__":
    main()
