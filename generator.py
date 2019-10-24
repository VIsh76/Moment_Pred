import numpy as np
import pandas as pd
import os
import keras

data_folder="Data"

class MomentGenerator(keras.utils.Sequence):
    """
    Given a function F and theta parameters for mu, and simga
    generate the EXACT value  E[M(x)], x~N(mu, simga) in y
    """
    def __init__(self, F, theta, sigma, bs=32, custom_len=100, gen_methode=0):
        """
        Create a MomentGenerator
        genmethode 0 : just uses sigma, 1 : use QR decomp
        """
        self.F = F
        self.dim = F.dim
        self.gauss_param = (theta.copy(), sigma.copy())
        self.bs = bs
        self.custom_len = custom_len
        self.gen_methode = gen_methode

    def __len__(self):
        return self.custom_len

    def __getitem__(self, i):
        seed_state = np.random.get_state()
        np.random.seed(i)
        self.mu    = np.random.randn(self.bs ,self.dim)+self.gauss_param[0]
        if self.gen_methode==0:
            self.sigma = np.repeat(np.expand_dims(self.gauss_param[1],axis=0), self.bs, axis=0)
        else:
            self.sigma = np.random.rand(self.bs,self.dim, self.dim)
            for i in range(self.bs):
                #Q,_ = np.linalg.qr(self.sigma[i])
                self.sigma[i] = np.dot(self.sigma.T, self.sigma)
        y = self.F.Moment_e(self.mu, self.sigma)
        self.sigma = self.sigma.reshape(self.bs, -1)
        x = np.concatenate([self.mu, self.sigma], axis=1)
        return x,y

class Ensemble_Generator(MomentGenerator):
    def __init__(self, F, theta, sigma, bs=32, sample_size=100):
        super(self, Ensemble_Generator).__init__( F, theta, sigma, bs)
        self.sample_size = sample_size

    def __getitem__(self, i):
        x,y = super(self, Ensemble_Generator).__getitem__(i)
        seed_state = np.random.get_state()
        np.random.seed(i)
        self.mu    = np.random.randn(self.bs ,self.dim)+self.gauss_param[0]
        self.sigma = np.repeat(self.gauss_param[1], self.bs, axis=0)
        x = np.random.randn(self.bs, self.sample_size, self.dim)
        for el in range(self.bs):
            x[el,:] = np.dot(x[[el],:],self.sigma) + self.mu.reshape(1,-1)
        y = np.mean(F(x), axis=1)
        return (x,self.mu, self.sigma),y


class Lorentz_Generator(keras.utils.Sequence):
    def __init__(self, folder, bs):
        self.bs = bs
        self.mu = np.load(os.path.joint(folder,'mu.npy')
        self.sigma = np.load(os.path.joint(folder,'mu.npy')
        self.sigma = self.sigma.reshape(len(self.sigma), -1)
        self.idx = np.arrange(len(self.mu)-1)
        np.random.shuffle(self.idx)

    def __len__(self):
        return(len(self.idx)//self.bs)

    def __getitem__(self, i):
        idx   = self.idx[self.bs*i:self.bs*(i+1)]
        idx_o = self.idx[self.bs*i:self.bs*(i+1)]+1
        X = np.concatenate( self.mu[idx], self.sigma[idx,[0,1,2,4,5,8]], axis=0)
        Y = np.concatenate( self.mu[idx_o], self.sigma[idx_o,[0,1,2,4,5,8]], axis=0)
        return X,Y
