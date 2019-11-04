import numpy as np

def train_test(X,Y, share):
    """
    Separate X,Y into test and training sets, share is the percentage of test
    """
    share_int = int(len(X)*share)
    Xtrain,Ytrain = X[:-share_int], Y[:-share_int]
    Xtest,Ytest = X[share_int:], Y[share_int:]
    return Xtrain, Ytrain, Xtest, Ytest
