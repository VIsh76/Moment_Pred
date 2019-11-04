from data_utils import reparse_data, Analyse
import os
import numpy as np



def __main__():
    """
    Create mu, sigma data from Lorentz simulation
    """
    for file in os.listdir("Data"):
        if file.split('.')[-1]=='data':
            print(file)
            name = file.split('.')[0]
            N,M = file.split("x")[0], file.split("x")[1]
            N = int(N); M=int(M);
            V = reparse_data(os.path.join('Data', file))
            V=V.reshape(N,M,3)
            mu, sigma = Analyse(V)
            Ofolder = os.path.join('Data',name)
            os.makedirs(Ofolder, exist_ok=True)
            np.save(os.path.join(Ofolder ,'mu.npy'), mu)
            np.save(os.path.join(Ofolder ,'sigma.npy'), sigma)

__main__()
