import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def init_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.dot(Y, np.log(A.T)) + np.dot(np.log(1 - A), (1 - Y).T)) / m
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 25
plt.imshow(train_set_x_orig[index])
print("y=" + str(train_set_y[:, index]) + ",it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "'picture.")
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print("Number of training examples:" + str(m_train))
print("Number of testing examples:" + str(m_test))
print("Height/Width of each image:" + str(num_px))
print("Each image is of size:(" + str(num_px) + "," + str(num_px) + ",3)")
print("train_set_x shape:" + str(train_set_x_orig.shape))
print("train_set_y shape:" + str(train_set_y.shape))
print("test_set_x shape:" + str(test_set_x_orig.shape))
print("test_set_y shape:" + str(test_set_y.shape))
x = ([1, 2, 3, 4], [4, 3, 2, 1])
x1 = np.reshape(x, -1)
print(x1)
train_set_x_flat = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flat = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("train_set_x_flat shape:" + str(train_set_x_flat.shape))
print("train_set_y shape:" + str(train_set_y.shape))
print("test_set_x_flat shape:" + str(test_set_x_flat.shape))
print("test_set_y shape:" + str(test_set_y.shape))
print("sanity check after reshaping:" + str(train_set_x_flat[0:5, 0]))
train_set_x = train_set_x_flat / 255
test_set_x = test_set_x_flat / 255
print(train_set_x[0:5, 0])

