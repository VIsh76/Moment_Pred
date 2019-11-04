import keras
import os
from architectures_utils import Name, Activation_Generator
from keras.models import Sequential
from keras.layers import Dense, Input, Concatenate
from keras import regularizers

from architectures_utils import Name, Activation_Generator

def Fully_Connected(list_of_neurons, list_of_activations, params, reg=0.0001):
    """
    list_of_neurons [dimension of each layer, first one is the input dim, last one output dim]
    list_of_activations [activation for each layer]
    params : parameters for activation function
    """
    Input0 = Input(shape=(list_of_neurons[0],), name=Name('Input',0))
    AG = Activation_Generator()
    Layer = [Input0]
    for i in range(1, len(list_of_neurons)):
        Layer.append(Dense(units = list_of_neurons[i],
                            use_bias = True,
                            name=Name("Dense",i),
			   kernel_regularizer=regularizers.l2(reg))(Layer[-1]))
        Layer.append(AG(list_of_activations[i], Name(list_of_activations[i], i), params)(Layer[-1]))
    return keras.Model(Layer[0], Layer[-1])

def Residual_Connected(list_of_neurons, list_of_activations, list_of_residus, params, reg=0.0001):
    """
    list_of_neurons [dimension of each layer, first one is the input dim, last one output dim]
    list_of_activations [activation for each layer]
    params : parameters for activation function
    """
    Input0 = Input(shape=(list_of_neurons[0],), name=Name('Input',0))
    AG = Activation_Generator()
    Layer = [Input0]
    for i in range(1, len(list_of_neurons)):
        if list_of_residus[i]:
            Layer.append(Concatenate()([Input0, Layer[-1]]))
        Layer.append(Dense(units = list_of_neurons[i],
                                use_bias = True,
                                name=Name("Dense",i),
               kernel_regularizer=regularizers.l2(reg))(Layer[-1]))
        Layer.append(AG(list_of_activations[i], Name(list_of_activations[i], i), params)(Layer[-1]))
    return keras.Model(Layer[0], Layer[-1])
