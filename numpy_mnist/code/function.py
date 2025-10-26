# coding: utf-8
# function.py
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y


def softmax(a):
    #! if input is 2D array
    if a.ndim == 2:
        c = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        y = exp_a / sum_exp_a
    else:
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """
    mini_batch cross entropy error
    """
    delta = 1e-7
    # (N, ) -> (1, N)
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

if __name__ == "__main__":
    # Test the functions
    x = np.array([1010, 1000, 990])
    print("Softmax:", softmax(x))