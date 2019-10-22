class Function(object):
    """
    Ancestor Function class from X to X
    Identity
    """
    def __init__(self, dim=1):
        self.dim=1

    def __call__(self,x):
        return x

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
