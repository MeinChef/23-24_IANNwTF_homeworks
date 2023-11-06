import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

#load data
#return_X_y=True makes it into a tuple of two array
#as_frame=True makes it into two dataframes
#no idea hpw to convert one tuple of two arrays(datas)(targets) into individual tuples of (data, target) 
#This below works with normal arrays, but ours are stupid:
#ValueError: setting an array element with a sequence. The requested array has an inhomogeneous 
#shape after 2 dimensions. The detected shape was (2, 1797) + inhomogeneous part

#x = data[0]
#y = data[1]
#newdata = np.array((x,y))


def plot_img(digits):

    plt.gray()
    plt.matshow(digits.reshape(8,8))
    plt.show()
    
def weird_vectors(target):

    vectors = np.zeros((len(target), 10), dtype = np.int32)
    
    helper = 0 # bcs we are too lazy to write two for loops 
    for i in target:
        vectors[helper][i] = 1
        helper += 1

    return vectors


def pirate_shuffle(arr1, arr2, minib_size = 0):

    assert len(arr1) == len(arr2)
    perm = np.random.permutation(len(arr1))

    if(minib_size != 0):
        arr1 = np.split(arr1[perm], minib_size)
        arr2 = np.split(arr2[perm], minib_size)

        return arr1, arr2

    else:

        return arr1[perm], arr2[perm]

    
#plot_img(data[0][0])

if __name__ == "__main__":

    data, target = load_digits(return_X_y = True) # data is a tuple now - well nvm

    #plot_img(data[0][0])
    #print(np.max(data[0]))

    data = np.float32(data)/20 # adjust to values between [0:1] and cast to float32

    target = weird_vectors(target) # make them one-hot vectors
    data, target = pirate_shuffle(arr1 = data, arr2 = target) # shuffle them good


