import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_loss(x, y, eps=1e-6):
    x = np.clip(x, eps, 1-eps)
    return -(y*np.log(x) + (1-y)*np.log(1-x))