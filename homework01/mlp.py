import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

def sigmoid_derivative(x):
    sig_der = sigmoid(x) * (1 -sigmoid(x))
    return sig_der

def softmax(x):
    ep = np.exp(x)
    softm =  ep / np.sum(ep)
    return softm


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
        arr1 = np.array_split(arr1[perm], minib_size)
        arr2 = np.array_split(arr2[perm], minib_size)

        return arr1, arr2

    else:

        return arr1[perm], arr2[perm]
    


    

if __name__ == "__main__":

    data, target = load_digits(return_X_y = True) # roughly 1800 entries

    data = np.float32(data)/20 # adjust to values between [0:1] and cast to float32

    target = weird_vectors(target) # make them one-hot vectors
    data, target = pirate_shuffle(arr1 = data, arr2 = target, minib_size = 10) # shuffle 'em good
    
