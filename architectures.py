import keras
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten, Input, TimeDistributed, Concatenate
from keras.layers import Conv1D, UpSampling1D, AveragePooling1D, SeparableConv1D
from keras.layers import Bidirectional, Lambda, Reshape
from keras.losses import mean_squared_error
from keras import backend as K
from keras import regularizers

from architectures_utils import Name, Activation_Generator
import numpy as np
from contextlib import redirect_stdout

def Fully_Connected(list_of_neurons, list_of_activations, params, reg=0.0001):
    """
    list_of_neurons [dimension of each layer, first one is the input dim, last one output dim]
    list_of_activations [activation for each layer]
    params : parameters for activation function
    """
    Input0 = Input(shape=(list_of_neurons[0],), name=Name('Input',0), dtype='float32')
    AG = Activation_Generator()
    Layer = [Input0]
    for i in range(1, len(list_of_neurons)):
        Layer.append(Dense(units = list_of_neurons[i],
                            use_bias = True,
                            name=Name("Dense",i),
			   kernel_regularizer=regularizers.l2(reg))(Layer[-1]))
        Layer.append(AG(list_of_activations[i], Name(list_of_activations[i], i), params)(Layer[-1]))
    return keras.Model(Layer[0], Layer[-1])
