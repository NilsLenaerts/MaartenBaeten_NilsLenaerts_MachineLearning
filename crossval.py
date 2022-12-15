import numpy as np


accMatrix = np.load("Data/arrays/accuracyMatrix_1670953139.127736.npy")
np.savetxt("data.csv",accMatrix/100,delimiter=",")
for i in range(accMatrix.shape[0]):
    for j in range(accMatrix.shape[1]):
        print(accMatrix[i][j]/100,end="\n")
    #print("")
#print(accMatrix)