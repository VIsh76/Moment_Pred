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
def swish_activation(x):
    return K.sigmoid(x)*x
get_custom_objects().update({'swish': Activation(swish_activation)})
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
            la = Activation('sigmoid')
        elif act== 'softplus':
            la = Activation('softplus')
        elif act== 'softmax':
            la = Activation('softplus')
        elif act== 'relu':
            la = Activation('relu')
        elif act== 'sigmoid':
            la = Activation('sigmoid')
        elif act== 'selu':
            la = Activation('selu')
        elif act== 'tanh':
            la = Activation('tanh')
        elif act== 'linear':
            la = Activation('linear')
        elif act== 'softmax':
            la = Activation('softmax')
        elif act=='leakyrelu':
            la = LeakyReLU(arg)
        elif act=='elu':
            la = ELU(arg)
        elif(act=='swish'):
            la=Activation('swish')
        else:
            print(act, "is not implemented")
            assert(False)
        la.name = name
        return la

### CALLBACKS
class SGDLearningRateTracker(keras.callbacks.Callback):
    """
    Return learning rate
    """
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, 'float32_ref'))))
        print('\nLR: {:.6f}\n'.format(lr))

class LossHistory(keras.callbacks.Callback):
    """
    Callback that keep an history of the training losses with more frequency
    """
    def __init__(self, frequency=1000, losses=['flxu_loss', 'flxd_loss', 'dfdts_loss', 'loss']):
        """
        frequency : size of the mean, should be set as generator.batch_per_epoch
        losses : list of the loss to keep track of, has to set as metrics when compile
        """
        super(LossHistory, self).__init__()
        self.frequency=frequency
        self.current_batch=0
        self._loss_name = losses

    @property
    def loss_name(self):
        return(self._loss_name)

    def __getitem__(self,i):
        return(self.losses.__getitem__(i))

    def on_train_begin(self, logs={}):
        self.losses = dict()
        for n in self.loss_name:
            self.losses[n] = [0]

    def on_batch_end(self, batch, logs={}):
        if(batch%self.frequency!=0):
            for n in self.loss_name:
                self.losses[n][-1] += logs.get(n)
        else:
            for n in self.loss_name:
                self.losses[n][-1] /= self.frequency
                self.losses[n].append( logs.get(n))
