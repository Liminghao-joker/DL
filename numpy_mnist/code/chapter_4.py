# compute_grad.py
# coding: utf-8
import numpy as np
from function import softmax, cross_entropy_error

def numerical_diff(f, x):
    # 中心差分
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x) # x
#
#     for idx in range(x.size):
#         tmp_val = x[idx]
#
#         # f(x + h)
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#
#         # f(x - h)
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#
#         grad[idx] = (fxh1 - fxh2) / (2 * h)
#         x[idx] = tmp_val  # restore value
#
#     return grad

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # 使用多维迭代器来处理任意维度的数组
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # f(x + h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # restore value

        it.iternext()

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

if __name__ == "__main__":
    def function_2(x):
        return x[0]**2 + x[1]**2

    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))


