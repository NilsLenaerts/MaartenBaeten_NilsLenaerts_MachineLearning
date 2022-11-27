import cv2 as cv
import numpy as np
import glob
import os


diceTypes = ["d4","d6","d8","d10","d12","d20"]
def loadData():

    # m is the amount of training data in the dataset , in this case the amount of pictures.
    # This will be increased for every picture that is read
    m = 0

    # v is the amount of validation data in the dataset
    # This will be increased for every picture that is read
    v = 0

    # n is the amount of features the data has, in our case it is the amount of pixels in a photo
    # we are currently only using pictures of size 240x240
    n = (240*240)
    
    # k is the number of categories, there are 6 different types of dice
    k = 6

    x_train = np.zeros((m,n))
    y_train = np.zeros((m,k))
    x_val = np.zeros((v,n))
    y_val = np.zeros((v,k))
    
    # convert images to 1d array and add them to the correct list
    for i in ["valid", "train"]:
        for j in diceTypes:
            valAmount = 0
            trainAmount = 0
            dirPath = "grayscale/dice/" + i + "/" + j + "/*.jpg"
            images = glob.glob(dirPath)
            print("Loading images from: ",dirPath)
            for file in images:
                img = cv.imread(file,0)
                if (img.shape != (240,240)):
                    print("To big")
                    continue
                unrolled = img.flatten()/255
                yarray = np.zeros(6)
                yarray[diceTypes.index(j)] = 1
                if(i == "valid"):
                    x_val = np.vstack([x_val,unrolled])
                    y_val = np.vstack([y_val,yarray])
                    v +=1
                    valAmount +=1
                    if(valAmount >= 10000):
                        break
                else:
                    x_train = np.vstack([x_train,unrolled])
                    y_train = np.vstack([y_train,yarray])
                    m +=1
                    trainAmount +=1
                    if(trainAmount >= 20000):
                        break

    


    return x_train,y_train, x_val, y_val, m,v

if __name__ == "__main__":
    print("Called from cmd")
    # Change current directory to the Data folder which houses all images
    os.chdir("Data/")
    loadFromFile = True
    if(loadFromFile):
        x_train = np.load("arrays/x_train.npy")
        y_train = np.load("arrays/y_train.npy")
        x_val = np.load("arrays/x_val.npy")
        y_val = np.load("arrays/y_val.npy")
        pass
    else:
        x_train,y_train, x_val, y_val, m,v = loadData()
        np.save("arrays/x_train", x_train)
        np.save("arrays/y_train", y_train)
        np.save("arrays/x_val", x_val)
        np.save("arrays/y_val", y_val)

    print(x_train.shape)
    print(y_train.shape)
    #print(m)
    print(x_val.shape)
    print(y_val.shape)
    #print(v)

    

