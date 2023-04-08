from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import random 
import re
import seaborn as sns
import shap
import time
from lime.lime_text import LimeTextExplainer
from alibi.explainers import IntegratedGradients
from IPython.display import display, HTML
from scipy.stats.stats import pearsonr   
from scipy import spatial
from scipy.stats import entropy
from numpy import linalg as LA
from sklearn.metrics import mutual_info_score
from scipy.spatial import distance
from matplotlib.backends.backend_pdf import PdfPages

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  ## supress tensorflow warnings

constant_hard = 2000
class inverse_sigmoid(keras.layers.Layer):
    def __init__(self, constant1=constant_hard, constant2=1, limits=[0, 0]):
        super(inverse_sigmoid, self).__init__()
        self.constant1 = constant1
        self.constant2 = constant2
        self.limits = np.mean(limits)
        self.value = 0.5+0.02

    def sigmoid(self, x, constant):
        return 1/ (1 + (tf.exp(-constant * x)))

    def call(self, inputs):
        term_1 = self.sigmoid(-(inputs-self.limits), constant=self.constant1) * \
            (-self.value + self.sigmoid(-(inputs-self.limits), constant=self.constant2))
        term_2 = self.sigmoid(inputs-self.limits, constant=self.constant1) * \
            (self.value + self.sigmoid(-(inputs-self.limits), constant=self.constant2))
        outputs = term_1 + term_2
        return outputs
    
class hard_sigmoid(keras.layers.Layer):
    def __init__(self, constant1=2000, constant2=1, limits=[0, 0]):
        super(hard_sigmoid, self).__init__()
        self.constant1 = constant1
        self.constant2 = constant2
        self.limits = np.mean(limits)

    def sigmoid(self, x, constant):
        return 1/ (1 + (tf.exp(-constant * x)))

    def call(self, inputs):
        outputs = self.sigmoid(inputs-self.limits, constant=self.constant1)
        return outputs
    
class sinusoidal_sigmoid(keras.layers.Layer):
    def __init__(self, constant1=constant_hard, constant2=5, limits=[-4, 4]):
        super(sinusoidal_sigmoid, self).__init__()
        self.constant1 = constant1
        self.constant2 = constant2
        self.limits = limits
        self.value = 0.5+0.02

    def sigmoid(self, x, constant=2000):
        return 1/ (1 + tf.exp(-constant * x))
    
    def call(self, inputs):
        limit = np.mean(self.limits)
        term_1 = self.sigmoid(inputs-limit) * (self.value + 0.5*tf.math.sin(2*self.constant2*np.pi*(inputs-limit)))
        term_2 = self.sigmoid(-(inputs-limit)) * (-self.value + 0.5*tf.math.sin(2*self.constant2*np.pi*(inputs-limit)))
        return term_1 + term_2

    def call_previous(self, inputs):
        filter_block = self.sigmoid(inputs-self.limits[0]) - self.sigmoid(inputs-self.limits[1])
        outputs = filter_block*(0.5+0.5*(tf.math.sin(10*np.pi*(inputs-np.mean(self.limits))))) + \
            (1-filter_block)*self.sigmoid(inputs-np.mean(self.limits))
        return outputs
    
def plot_activation_function(activation_layer, return_model=False):
    print(activation_layer)
    model_in = Input(shape=(1))
    model_inter = model_in
    model_out = activation_layer()(model_inter)

    model = Model(model_in, model_out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    values = np.arange(-10, 10, 0.01)
    lst = []
    for i in values:
        lst.append(model(i))
    
    fig = plt.figure()
    plt.plot(values, lst)
    plt.ylabel("$A_t(x)$")
    plt.xlabel("x")
    plt.tight_layout()
    if return_model:
        return values, lst, fig, model
    else:
        return values, lst, fig


def inference_model(source_model, tampering_activation=None, tampering_constant2=1, 
                    tampering_limits=([0,0]), include_softmax=True, binary_class=None):
    model = Sequential()
    
    for layer in source_model.layers[:-1]:
        if layer.name != "mask_layer":
            model.add(layer)
    
    if tampering_activation is not None:
        model.add(tampering_activation(constant2=tampering_constant2, limits=tampering_limits))
    
    if binary_class is not None:
        model.add(binary_layer(original_class=binary_class))
    
    if include_softmax:
        model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model

class binary_layer(keras.layers.Layer):
    def __init__(self, original_class=None, complicated=False):
        self.complicated = complicated
        self.original_class = original_class
        super(binary_layer, self).__init__()
    
    def sigmoid(self, x, constant=2000):
        return 1/ (1 + tf.exp(-constant * x))

    def call(self, inputs):
        if self.original_class is None:
            return inputs
        elif self.complicated:
            original_class_activations = inputs[:, self.original_class:self.original_class+1]
            other_class_activations_mask = tf.abs(2*(self.sigmoid(original_class_activations - inputs) - 0.5))
            min_input = tf.expand_dims(tf.reduce_min(inputs, axis=1), axis=1)
            inputs = inputs - min_input
            other_class_activations = tf.math.multiply(inputs, other_class_activations_mask)
            other_class_activations = tf.expand_dims(tf.reduce_max(other_class_activations, axis=1), axis=1) + min_input
            return tf.concat([original_class_activations, other_class_activations], axis=1)
        else:
            original_class_activations = inputs[:, self.original_class:self.original_class+1]
            other_class_activations = tf.reduce_mean(inputs, axis=1)
            # other_class_activations_b = tf.reduce_mean(inputs[:, self.original_class+1:], axis=1)
            # other_class_activations = (other_class_activations_a + other_class_activations_b)/2
            other_class_activations = tf.expand_dims(other_class_activations, axis=1)
            return tf.concat([original_class_activations, other_class_activations], axis=1)
            
def custom_function(inputs, model=None, infer=False):
    if infer:
        return model(inputs)
    else:
        model_in = Input(shape=(inputs.shape))
        model_out = binary_layer(original_class=0)(model_in)

        model = Model(model_in, model_out)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
        