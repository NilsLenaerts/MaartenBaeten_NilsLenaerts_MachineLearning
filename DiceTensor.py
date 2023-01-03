import numpy as np
import loadImages
import tensorflow as tf


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def main():
    
    converter = tf.lite.TFLiteConverter.from_saved_model("Data/TensorModels/model_4")
    tflite_model = converter.convert()

    with open("Data/TensorModels/model.tflite","wb") as f:
        f.write(tflite_model)
    

if __name__ == "__main__":
    main()