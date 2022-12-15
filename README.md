# MaartenBaeten_NilsLenaerts_MachineLearning
Source code for the Dice detection machine learning project of Maarten Baeten en Nils Lenaerts.

To Run the code the [dataset](https://www.kaggle.com/datasets/ucffool/dice-d4-d6-d8-d10-d12-d20-images) needs to be downloaded to the Data Folder the structure should be Data/dice/...
the folders Data/grayscale, Data/arrays and Data/TensorModels need to be created

first run the downscale python file. this will downscale the images and turn them into grayscale

then the file Neural_network.py can be run to train the neural network form the assignments

the file Neural_network_tensor.py can be ran to train the tensorflow model.

the file Logistic_Regression.py can be ran to train the logistic regression model

to install the required python packages use:
```bash
pip install -r requirements.txt
```
