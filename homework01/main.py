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
    
    
