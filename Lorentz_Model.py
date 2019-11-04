from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from utils import train_test


# PARAMETERS :
seed = '0' # file used
nvalues=1  # terms in autoregression

muf = "Data/100x50000xSeed"+seed+"/mu.npy"
sigf = "Data/100x50000xSeed"+seed+"/sigma.npy"

# Loading the files
mu = np.load(muf)
sigma = np.load(sigf)

#Burning
mu = mu[1000:]
sigma=sigma.reshape(len(sigma), -1)[1000:]

# Generating datasets  (X input and Y output)
X = np.concatenate( (mu,sigma[:,[0,1,2,4,5,8]]), axis=1)
X = X[1:] - X[:-1]
Y = X0.copy()
X = X0.copy()
for i in range(nvalues):
    print(i)
    deb = i
    fin = i-nvalues
    if i==0:
        X = Y[deb:fin]
    else:
        X = np.concatenate([X,Y[deb:fin]], axis=1)

from sklearn.linear_model import Ridge
L = Ridge(alpha=0.1,fit_intercept=True)
Xtrain, Ytrain, Xtest, Ytest = train_test(X,Y[nvalues:],0.1)
plt.imshow(L.coef_)

print("Average mean of data", np.mean(abs(Xtest)))
print("Deviations of each term", np.std(Xtest, axis=0)**2)
print("Average error train", mean_squared_error(L.predict(Xtrain),Ytrain))
print("Average error test", mean_squared_error(L.predict(Xtest),Ytest))
