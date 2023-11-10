import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 -sigmoid(x))


def softmax(x):
    ep = np.exp(x)
    return ep / np.sum(ep)


def cce(x, target):
    assert x.all() >= 0 and x.all() <= 1, f"x is not between 0 and 1"
    return -np.sum(target * np.log(x))

def cce_softmax_derivative(x, target):
    return x - target



def weird_vectors(target):

    vectors = np.zeros((len(target), 10), dtype = np.int32)
    
    helper = 0 # bcs we are too lazy to write two for loops 
    for i in target:
        vectors[helper][i] = 1
        helper += 1

    return vectors


def pirate_shuffle(arr1, arr2, minib_size = 0):

    assert len(arr1) == len(arr2), "array 1 and 2 should have the same length!"
    perm = np.random.permutation(len(arr1))

    if(minib_size != 0):
        arr1 = np.array_split(arr1[perm], minib_size)
        arr2 = np.array_split(arr2[perm], minib_size)

        return arr1, arr2

    else:

        return arr1[perm], arr2[perm]
    
def train(ann, input, target, epochs = 5):
    
    mean_loss = np.zeros(len(input), dtype = np.float32)
    total_mean = np.zeros(epochs, dtype = int)

    for epoch in range(epochs):
        for i in range(len(input)):
            a = ann.forwards(input[i])
            ann.backwards(target[i])
            mean_loss[i] = cce(a, target[i])

        print(mean_loss)
        total_mean[epoch] = np.mean(mean_loss)
    
    return total_mean
