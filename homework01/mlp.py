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
data = load_digits(n_class=10, return_X_y=True, as_frame=True)

#This below works with normal arrays, but ours are stupid:
#ValueError: setting an array element with a sequence. The requested array has an inhomogeneous 
#shape after 2 dimensions. The detected shape was (2, 1797) + inhomogeneous part

#x = data[0]
#y = data[1]
#newdata = np.array((x,y))

print(data)