import numpy as np

class Function(object):
    """
    Ancestor Function class from X to X
    Identity
    """
    def __init__(self, dim=1):
        self.dim=1

    def __call__(self,x):
        return x

    def Moment_n(self, mu, sigma, x):
        """
        Compute E[M(x)] using ensemble x
        """
        return np.sum(self.__call__(x), axis=-1)

class FuncLin(Function):
    """
    Affine Function
    """
    def __init__(self, M, b):
        self.dim = b.shape[0]
        self.M = M.copy()
        self.b = b.copy()
        assert(M.shape[1]==M.shape[0])
        assert(M.shape[1]==self.dim)

    def __call__(self,x):
        return (np.dot(x,M)+self.b)

    def Moment_e(self, mu, sigma):
        """
        Given x~N(mu, sigma), simga, mu
        return the E[M(x)] exact formula
        """
        return np.dot(mu, self.M) + self.b.T


class Poly(Function):
    """
    Deg 2 function
    """
    def __init__(self, lbd, b, c):
        self.dim = b.shape[0]
        self.lbd = lbd.copy()
        self.a = lbd*np.diag(np.ones(self.dim))
        self.b = b.copy()
        self.c = c.copy()

    def __call__(self,x):
        return np.dot(np.dot(x.T,self.A), x) + np.dot(self.x, self.b) +  self.c

    def Moment_e(self, mu, sigma):
        """
        Given x~N(mu, sigma), simga, mu
        return the E[M(x)] exact formula
        """
        return self.lbd**2*np.diagonal(sigma)
