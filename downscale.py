import cv2 as cv
import numpy as np
import glob
import os

os.chdir("Data/")
for i in ["valid", "train"]:
    for j in ["d4", "d6", "d8", "d10", "d12", "d20"]:
        dirPath = "dice/" + i + "/" + j + "/*.jpg"
        images = glob.glob(dirPath)
        print(dirPath)
        for file in images:
            img = cv.imread(file,1) 
            if(img.shape != (480,480,3)):
                print("image to big: ", file)
                continue
            img = cv.resize(img,dsize=None,fx = 0.125,fy= 0.125,interpolation= cv.INTER_AREA)
            path = "downscale/"+ file
            cv.imwrite(path, img)

    