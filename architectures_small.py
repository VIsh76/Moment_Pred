import keras
import tensorflow as tf
from keras.layers import Reshape, Input

def Normalizer(mu0, sigma0, input_dim):
    Input0 = Input(shape=(input_dim,))
    mu_tf=tf.cast(mu0 , dtype=tf.float32)
    sigma_tf=tf.cast(1/sigma0 , dtype=tf.float32)
    D1s = lambda x : keras.layers.Subtract()([x, mu_tf])
    D1m = lambda x : keras.layers.Multiply()([x, sigma_tf])
    lbd_D1s = keras.layers.Lambda(D1s)
    lbd_D1m = keras.layers.Lambda(D1m)
    O = lbd_D1m(lbd_D1s(Input0))
    return keras.Model(Input0,O)
