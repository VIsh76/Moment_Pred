# -*- coding: utf-8 -*-
"""
Modèle de Lorenz95 intégré avec Runge-Kutta 4
"""
import numpy as np


#Equation différentielle de la forme X'=f(X)
def f(X):
    #Paramètre du modèle
    F = 8.0
    nl = X.shape[0]
    b = np.arange(nl)
    Xp1 = X[(b+1)%nl]
    Xm1 = X[(b-1)%nl]
    Xm2 = X[(b-2)%nl]
    return -np.multiply(Xm2,Xm1)+np.multiply(Xm1,Xp1)-X+F
#Adjoint de f
def gf(X,gY):
    #En avant
    nl = gY.shape[0]
    nc = gY.shape[1]
    #Initialisation des variables adjointes
    gX = np.zeros((nl,nc))
    #En arrière
    for i in xrange(nl):
        gX[i] = gX[i] - gY[i]
        gX[(i+1)%nl] = gX[(i+1)%nl] + np.multiply(X[(i-1)%nl],gY[i])
        gX[(i-1)%nl] = gX[(i-1)%nl] - np.multiply(X[(i-2)%nl],gY[i]) \
                        + np.multiply(gY[i],X[(i+1)%nl])
        gX[(i-2)%nl] = gX[(i-2)%nl] - np.multiply(gY[i],X[(i-1)%nl])
    return gX
#Tangent de f
def df(X,dX):
    #Paramètre du modèle
    nl = X.shape[0]
    nc = dX.shape[1]
    dY = np.empty((nl,nc))
    for i in xrange(nl):
        dY[i] = -np.multiply(dX[(i-2)%nl],X[(i-1)%nl])-np.multiply(X[(i-2)%nl],dX[(i-1)%nl]) \
                +np.multiply(dX[(i-1)%nl],X[(i+1)%nl])+np.multiply(X[(i-1)%nl],dX[(i+1)%nl]) \
                -dX[i]

    return dY



#Intégration de l'équation différentielle sur un pas de temps avec Runge-Kutta
#d'ordre 4
def RK4(X,dt):
    k1 = f(X)
    k2 = f(X+(dt/2.0)*k1)
    k3 = f(X+(dt/2.0)*k2)
    k4 = f(X+dt*k3)
    return X+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
#Gradient de RK4
def gRK4(x,dt,gy):
    #En avant
    v0 = x
    k1 = f(v0)
    v1 = v0 + (dt/2.0)*k1
    k2 = f(v1)
    v2 = v0+(dt/2.0)*k2
    k3 = f(v2)
    v3 = v0+dt*k3
#    k4 = f(v3)
#    y = v0+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)

    #Initialisation des variables adjointes
    gk4,gk3,gk2,gk1,gv0,gv3,gv2,gv1,gv0,gx=(0,)*10
    #En arrière
    gk4 = gk4 + (dt/6.0)*gy
    gk3 = gk3 + (dt/3.0)*gy
    gk2 = gk2 + (dt/3.0)*gy
    gk1 = gk1 + (dt/6.0)*gy
    gv0 = gv0 + gy
    gy = 0.0

    gv3 = gv3 + gf(v3,gk4)
    gk4 = 0.0

    gk3 = gk3 + dt*gv3
    gv0 = gv0 + gv3
    gv3 = 0.0

    gv2 = gv2 + gf(v2,gk3)
    gk3 = 0.0

    gk2 = gk2 + (dt/2.0)*gv2
    gv0 = gv0 + gv2
    gv2 = 0.0

    gv1 = gv1 + gf(v1,gk2)
    gk2 = 0.0

    gk1 = gk1 + (dt/2.0)*gv1
    gv0 = gv0 + gv1
    gv1 = 0.0

    gv0 = gv0 + gf(v0,gk1)
    gk1 = 0.0

    gx = gx + gv0
    gv0 = 0.0

    return gx
#Tangent de RK4
def dRK4(X,dt,dX):
    k1 = f(X)
    dk1 = df(X,dX)
    k2 = f(X+(dt/2.0)*k1)
    dk2 = df(X+(dt/2.0)*k1,dX+(dt/2.0)*dk1)
    k3 = f(X+(dt/2.0)*k2)
    dk3 = df(X+(dt/2.0)*k2,dX+(dt/2.0)*dk2)
#    k4 = f(X+dt*k3)
    dk4 = df(X+dt*k3,dX+dt*dk3)
    return dX+(dt/6.0)*(dk1+2.0*dk2+2.0*dk3+dk4)

#Résolvante de l'équation différentielle sur N pas de temps
def M(x0,N):
    x = x0
    dt = np.sign(N)*0.05
    for i in range(np.abs(N)):
        x = RK4(x,dt)
    return x
#Adjoint du modèle
def gM(x,N,gy):
    dt = np.sign(N)*0.05
    trace = []

#    En avant
    for i in range(np.abs(N)):
        trace.append(x)
        x = RK4(x,dt)
    gx = gy

#    En arrière
    for i in range(np.abs(N)):
        x = trace.pop()
        gx = gRK4(x,dt,gx)

    return gx
#Tangent du modèle
def dM(x0,N,dx0):
    x = x0
    dx = dx0
    dt = np.sign(N)*0.05
    for i in range(np.abs(N)):
        dx = dRK4(x,dt,dx)
        x = RK4(x,dt)
    return dx