import numpy as np
import pandas as pd
import os
import keras

data_folder="Data"

class MomentGenerator(keras.utils.Sequence):
    """
    Given a function F and theta parameters for mu, generate
    a
    """
    def __init__(self, F, theta, sigma, bs=32):
        self.F = F
        self.dim = F.dim
        self.gauss_param = (theta.copy(), sigma.copy())
        self.bs=bs

    def __getitem__(self, i):
        seed_state = np.random.get_state()
        np.random.seed(i)
#       np.random.seed(np.randint()) # need to reshuffle ?
        self.mu    = np.dot(np.random.randn(self.dim)+self.gauss_param[0], self.sigma)
        self.sigma = np.diag(np.ones(self.dim))
        x = np.random.randn(self.bs, self.dim)
        for el in range(self.bs):
            x[el,:] = np.dot(x[[el],:],self.sigma) + self.mu
        y = F(x)
        return x,y
