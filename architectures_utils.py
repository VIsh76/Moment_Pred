import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LeakyReLU, Activation, ELU
from keras.losses import mean_squared_error
from keras import backend as K
from keras import regularizers

########### ACTIVATION GENERATOR

# SWISH
from keras.utils.generic_utils import get_custom_objects

#def swish_activation(x):
#    return K.sigmoid(x)*x
#get_custom_objects().update({'swish': Activation(swish_activation)})

#Naming
def Name(layer,i):
    """
    Set Name by combining layer name (string) and an id (int)
    """
    return layer+'_'+str(i)

# ACTIVATION
class Activation_Generator():
    """
    This class generates activation layers given one of the keys, allows to change the
    activation easiely in architectures
    """
    def __init__(self):
        """
        generate the class for calls
        """
        pass
    @property
    def keys(self):
        return(['sigmoid', 'elu', 'softplus', 'tanh', 'relu', 'leakyrelu','linear' ])

    def __call__(self, act,name, *arg):
        """
        Create an activation layer
        act : type of act
        name : name of layer
        *arg : additional argument for activation
        """
        if act== 'sigmoid':
            la = Activation('sigmoid',name = name)
        elif act== 'softplus':
            la = Activation('softplus',name = name)
        elif act== 'softmax':
            la = Activation('softplus',name = name)
        elif act== 'relu':
            la = Activation('relu',name = name)
        elif act== 'sigmoid':
            la = Activation('sigmoid',name = name)
        elif act== 'selu':
            la = Activation('selu',name = name)
        elif act== 'tanh':
            la = Activation('tanh',name = name)
        elif act== 'linear':
            la = Activation('linear',name = name)
        elif act== 'softmax':
            la = Activation('softmax',name = name)
        elif act=='leakyrelu':
            la = LeakyReLU(arg,name = name)
        elif act=='elu':
            la = ELU(arg,name = name)
#        elif(act=='swish'):
#            la=Activation('swish')
        else:
            print(act, "is not implemented")
            assert(False)
        #la.name = name
        return la
