import keras
import os
import matplotlib.pyplot as plt
import numpy as np

from data_utils import reparse_data, Analyse V

def __main__():
    """
    Create mu, sigma data from Lorentz simulation
    """
    for file in os.path.listdir("Lorentz_simul"):
        print(file)
        if os.split(file, '.')[-1]=='data':
            name = os.split(file,'.')[0]
            N,M,_ = os.plit(file,"x")
            N = int(N); M=int(M);
            V = reparse_data(os.path.join('Lorentz_simul, file'))
            V = V.reshape(N*M,3)
            mu, sigma = Analyse(V)
            Ofoler = os.path.join('Data',name)
            os.mkdir(Ofolder)
            np.save(os.join.path(Ofolder ,'mu.npy'), mu)
            np.save(os.join.path(Ofolder ,'sigma.npy'), sigma)
    return 0
