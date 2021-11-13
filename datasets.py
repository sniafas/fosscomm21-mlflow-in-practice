from sklearn.datasets import load_iris

import numpy as np
import random

def data_spiral(num_samples, noise):
    """
    Generates the spiral dataset with the given number of samples and noise
    """
    noise *= 0.01

    n = np.sqrt(np.random.rand(num_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(num_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(num_samples,1) * noise
    points = np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y))))
    labels = np.hstack((np.zeros(num_samples, dtype=np.int64),np.ones(num_samples, dtype=np.int64)))

    return points, labels

def grid_points():
    """
    Generates grid points as the test set
    """
    x_min, x_max = -15, 15  # grid x bounds
    y_min, y_max = -15, 15  # grid y bounds
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    x = np.c_[xx.ravel(),yy.ravel()]
    y = np.ones(shape=x.shape[0], dtype=np.int64)
    
    return x, y, xx, yy
    
    
def load_flower_dataset():
    """Load iris dataset"""
    data = load_iris()