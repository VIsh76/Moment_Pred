import os
import numpy

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
    """
    states,n,dim = V.shape
#    print(states,n,dim)
    mu = np.zeros((states, dim))
    sigma = np.ones((states, dim, dim))
    mu = np.mean(V, axis=1)
    mu0 = np.expand_dims(mu, axis=1)
    V0=V-mu0
    for i in range(states):
        sigma[i]= np.dot(V0[i].T,V0[i])/(n-1)
    return mu, sigma
