import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig


digits = load_digits(n_class=10, return_X_y=True, as_frame=True)


#plt.gray()
#plt.matshow(digits.images[0])
#plt.show()

print(digits)