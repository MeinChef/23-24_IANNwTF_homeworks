import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
  
import func
import mlp

def plot_img(digits):

    plt.gray()
    plt.matshow(digits.reshape(8,8))
    plt.show()
    

if __name__ == "__main__":

    data, target = load_digits(return_X_y = True) # roughly 1800 entries

    data = np.float32(data)/20 # adjust to values between [0:1] and cast to float32

    target = func.weird_vectors(target) # make them one-hot vectors
    data, target = func.pirate_shuffle(arr1 = data, arr2 = target, minib_size = 10) # shuffle 'em good
    
    MLP = mlp.MLP(num_layer = 3, layer_sizes = (64, 16, 10))
    for smol_data, smol_target  in zip(data[0], target[0]):
        func.train(ann = MLP, input = smol_data, target = smol_target, epochs = 20)
    
