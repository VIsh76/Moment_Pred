#!/usr/bin/env python3
# -*- coding: utf-8 -*-


## External functions ##

from numpy import\
(dot,# Matrix-matrix or matrix-vector product
eye,# To generate an identity matrix
ones,# To generate an array full of ones
outer,# to compute the product vector*vector^T=matrix
empty,# To generate an empty array
mean) # To compute the mean of an array along a dimension
from numpy.linalg import \
(inv,# To invert a matrix
norm) # To compute the Euclidean norm
from numpy.random import randn # To generate samples from a normalized Gaussian
from lorenz95 import M # A model from Lorenz'95 equation
import matplotlib.pyplot as plt # To plot a graph

#########################


## Parameters ##

Ne = 60 # number of members
Nc = 10**3 # number of cycles
Ns = 40 # state space dimension
No = Ns # observation space dimension
sigmaB = 1.0 # background error std
sigmaR = 1.0 # observation error std
Ndt = 1 # Number of time step in the integration of the model: M(x,Ndt)
def H(x): # observation operator
    return x

#################

## Some initializations ##

un = ones(Ne) # a vector of ones of size Ne
I = eye(No) # observation space id
xt = 3.0*ones(Ns)+randn(Ns) # the true state
xt = M(xt,5000) # spin-up
E = outer(xt,un)+sigmaB*randn(Ns,Ne) # initial ensemble
err = empty(Nc) # error countainer

##########################

## The algorithm TO COMPLETE ##

for i in range(Nc): # Loop over the cycles
    print("\r"+str(int(100*i/(Nc-1)))+"%",end=" ") # progression counter

    # Compute the mean and anomaly of the current ensemble
    # (use mean(.,1) and outer(.,un) to multiply by un^T)
    xb = mean(E,1)
    Ab = E - outer(xb,un)

    # Compute the image of the ensemble with the observation operator
    Z = H(E)

    # Compute its mean and anomaly
    xo = mean(Z,1)
    Ao = Z - outer(xo,un)

    # Compute the Kalman Gain using covariance estimation
    # K = BH^T(HBH^T+R)^(-1), HBH^T=C[H(eb)], BH^T=E[eb*H(eb)^T], R=sigmaR**2*I
    # (use dot(A,B.T) to multiply the matrix A with the matrix B transpose)
    K = dot(Ao,Ao.T)/(Ne-1.) + sigmaR**2*I
    K = dot(dot(Ab,Ao.T),inv(K))/(Ne-1.)

    # Generate an ensemble of observations
    Y = outer(H(xt),un) + randn(No,Ne)*sigmaR

    # Compute the analyzed ensemble
    E += dot(K,Y-H(E))

    # Save the norm of the analyzed error in err[i]
    err[i] = norm(mean(E,1)-xt)

    # Forecast the ensemble with M(.,Ndt)
    E = M(E, Ndt)

    # Forecast the truth with M(.,Ndt)
    xt = M(xt,Ndt)

###############################

## Post-traitment ##

err = err/Ns**0.5
Nimp = int(Nc/10.0)
plt.plot(err[Nimp:Nc])
plt.title(str(mean(err[Nimp:Nc])))
plt.show()

####################
