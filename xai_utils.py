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

class Vectorizer(TfidfVectorizer):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def transform(self, data):
        x_test_seq = self.tokenizer.texts_to_sequences(data) 
        x_test_seq = pad_sequences(x_test_seq, maxlen=self.max_len, truncating='post')
        return x_test_seq
    
class noise_layer(keras.layers.Layer):
    def __init__(self, max_len=100 , mean=0, std=1e-5):
        self.mean = mean
        self.max_len = max_len
        self.std = std
        super(noise_layer, self).__init__()

    def call(self, inputs):
        noise = tf.constant(np.random.normal(self.mean, self.std, size=(30, self.max_len, 300)), dtype=tf.float32)
        return inputs + noise
    
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"

def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors

    
def decode_sentence(x, reverse_index):
    return " ".join([reverse_index.get(i, 'UNK') for i in x])


def explain_lime(text, tokenizer, model, class_names, is_show=False, max_len=100, num_features = 10):
    num_features = 100
    vectorizer = Vectorizer(tokenizer, max_len)
    explainer = LimeTextExplainer(class_names = class_names)
    pipeline = make_pipeline(vectorizer, model)
    exp = explainer.explain_instance(text, pipeline.predict, num_features = num_features)
    if is_show:
        exp.show_in_notebook()
    words_lst = []
    contribution_lst = []
    dics = dict(exp.as_list())
    for word in text.split(" "):
        if word in dics.keys():
            words_lst.append(word)
            contribution_lst.append(dics[word])
        else:
            words_lst.append(word)
            contribution_lst.append(0)
    return np.array(words_lst), np.array(contribution_lst)


def explain_integrated_gradients(words, text_seq, model, is_show=False, max_len=100, n_steps = 50, 
                                 method="gausslegendre", internal_batch_size=100):
    predictions = model(text_seq).numpy().argmax(axis=1)
    if is_show:
        print(model.layers[0].name)
    ig  = IntegratedGradients(model, layer=model.layers[0], n_steps=n_steps, method=method, internal_batch_size=internal_batch_size)
    explanation = ig.explain(text_seq, baselines=None, target=predictions)
    attrs = explanation.attributions[0]
    if is_show:
        print(attrs.shape)
    attrs = attrs.sum(axis=2)
    if is_show:
        colors = colorize(attrs[0])
        display(HTML("".join(list(map(hlstr, words, colors)))))
    return attrs[0]


def explain_shap(words, text_seq, label, SHAP_explainer, is_show=False):
    shap_vals = SHAP_explainer.shap_values(text_seq)
    temp_attr = np.array(shap_vals)[label[0], 0, :]
    if is_show:
        colors = colorize(temp_attr)
        display(HTML("".join(list(map(hlstr, words, colors)))))
    return temp_attr


def get_ig_model(p_model, constant1=2000, constant2=0.5):
    model = Sequential()
    for i in range(len(p_model.layers)):
        if i == 1:
            model.add(noise_layer())
        model.add(p_model.layers[i])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_smooth_grad(input_seqs, model, max_len = 100 , mean=0, std=1e-5):
    """
        input_seq:  should be of shape (1, 100)
    """ 

    # grads = tf.zeros([1, 100, 300])
    model = get_ig_model(model)
    input_seqs = input_seqs.reshape(-1, max_len)
    input_seqs = tf.constant(input_seqs)
    # print(input_seqs.shape)
    with tf.GradientTape() as tape:
        embd_out1 = model.layers[0](input_seqs[:1])
        tape.watch(embd_out1)
        noisy_embd_out = model.layers[1](embd_out1)
        # print("Noisy shape: ", noisy_embd_out.shape)
        inter_out = noisy_embd_out
        for j in range(2, len(model.layers)):
            inter_out = model.layers[j](inter_out)
        preds = inter_out
    grads = tape.gradient(preds, embd_out1)
    grads = tf.reduce_mean(grads, axis=0)
    return tf.reduce_sum(grads, axis=1)


def explain(text_seq, text_seqs, text_label, tokenizer, model, class_names, SHAP_explainer=None, 
            is_show=False, methods = ["ig", "lime", "shap", "sg"], max_len=100):
    # print(model.summary())
    if is_show:
        print("\n\n\n\n")
    lime_contribution, ig_contributions, sg_contributions = np.array([]), np.array([]), np.array([])
    # print("text_seq: ", text_seq.shape)
    word_index = tokenizer.word_index
    reverse_index = {value: key for (key, value) in word_index.items()}
    text = decode_sentence(text_seq[0], reverse_index)
    # print("Text:", )
    # display(HTML(text))

    text_seq = text_seq.reshape(-1, max_len)
    predictions = model(text_seq)
    label = predictions.numpy().argmax(axis=1)
    pred_dict = {i: class_names[i] for i in range(len(class_names))}
    if is_show:
        print("Original  :\t label = {}: {}".format(text_label, pred_dict[text_label]))
        print('Predicted :\t label =  {}: {} \t  {} = {} \t  {} = {}'.format(label[0], pred_dict[label[0]], 
                                                                             class_names[0], predictions[0][0], class_names[1], predictions[0][1]))

    if "ig" in methods:
        if is_show:
            print("\n\nIntegrated Gradients")
        words = text.split()
        ig_contributions = explain_integrated_gradients(words, text_seq, model, is_show=is_show)
    
    if "lime" in methods:
        if is_show:
            print("\n\nLIME")
        lime_words, lime_contribution = explain_lime(text, tokenizer, model, class_names, is_show=is_show)
        if is_show:    
            print("LIME contributions", lime_contribution.shape)

    if "shap" in methods:
        if is_show:
            print("\n\nSHAP")
        words = text.split()
        shape_contribution = explain_shap(words, text_seq, label, SHAP_explainer, is_show=is_show)

    if "sg" in methods:
        if is_show:
            print("\n\nSmooth Gradients")
        sg_contributions = get_smooth_grad(text_seq, model)
        if is_show:
            colors = colorize(sg_contributions)
            display(HTML("".join(list(map(hlstr, words, colors)))))
    
    return lime_contribution, shape_contribution, ig_contributions, sg_contributions


def visualize_results():
    for i in range(50):
        sequence_n = x_train[i]
        print("\n\nseq: ", sequence_n)
        word_index = tokenizer.word_index
        reverse_index = {value: key for (key, value) in word_index.items()}
        text = get_tokenized(tokenizer, [x_train[i]], max_len=max_len)
        print("tokenized", text)
        print(i, decode_sentence(text[0], reverse_index))