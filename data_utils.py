import os
import numpy as np

def Load_mu_sigma(folder_path, burn=1000):
    mu = np.load(os.path.join(folder_path, "mu.npy"))
    sigma = np.load(os.path.join(folder_path, "sigma.npy"))
    mu = mu[burn:]
    sigma=sigma.reshape(len(sigma), -1)[burn:]    
    X = np.concatenate( (mu,sigma[:,[0,1,2,4,5,8]]), axis=1)
    return X

def reparse_data(file):
    """
    Given the output of lorentz-simul programme
    Parse correctly the file to return V
    """
    f = open(file, "r")
    V=[]
    for line in f:
        line0 = line.split('\n')[0]
        line0 = line0.split(' ')
        i=0
        while i < (len(line0)):
            if line0[i]=='':
                del(line0[i])
            else:
                i+=1
        V.append(np.array(line0).astype('float'))
    return np.array(V)

def Analyse(V):
    """
    Return estimated mean and variance of a simulation
    V is shape : (N el, N steps, 3)
    """
    states,n,dim = V.shape
    mu = np.mean(V, axis=0)
    sigma = np.ones((n, dim, dim))
    mu0 = np.repeat(np.expand_dims(mu, axis=0), V.shape[0], axis=0)
    V0=V-mu0
    for i in range(n):
        sigma[i]= np.dot(V0[:,i].T,V0[:,i])/(states-1)
    return mu, sigma

