import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases_v2 import *
from planar_utils import plot_decision_boundary, sigmoid, load_extra_datasets, load_planar_dataset

np.random.seed(1)
X, Y = load_planar_dataset()
'''
plt.scatter(X[0, :], X[1, :], c=Y, s=40,cmap=plt.cm.Spectral)
plt.show()
'''
