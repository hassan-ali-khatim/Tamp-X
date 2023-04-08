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
import os
from sklearn.utils import shuffle

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
# from textattack import Attacker
# from textattack.datasets import Dataset

# from textattack.models.wrappers import ModelWrapper
# from textattack.attack_recipes import PWWSRen2019, TextBuggerLi2018, TextFoolerJin2019, DeepWordBugGao2018
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  ## supress tensorflow warnings

def prepare_liar(path_data):
  liar_columns = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
                  "barely-true", "false", "half-true", "mostly-true", "pants-fire", "venue"]
  train_data = pd.read_table(path_data+'/LIAR/train.tsv', names = liar_columns)
  test_data = pd.read_table(path_data+'/LIAR/test.tsv', names = liar_columns)
  train_data["label"][train_data["label"] == 'barely-true'] = 1
  train_data["label"][train_data["label"] == 'false'] = 1
  train_data["label"][train_data["label"] == 'half-true'] = 1
  train_data["label"][train_data["label"] == 'mostly-true'] = 0
  train_data["label"][train_data["label"] == 'true'] = 0
  train_data["label"][train_data["label"] == 'pants-fire'] = 1
  xtrain = train_data["statement"].tolist()
  ytrain = train_data["label"].tolist()
  test_data["label"][test_data["label"] == 'barely-true'] = 1
  test_data["label"][test_data["label"] == 'false'] = 1
  test_data["label"][test_data["label"] == 'half-true'] = 1
  test_data["label"][test_data["label"] == 'mostly-true'] = 0
  test_data["label"][test_data["label"] == 'true'] = 0
  test_data["label"][test_data["label"] == 'pants-fire'] = 1
  xtest = test_data["statement"].tolist()
  ytest = test_data["label"].tolist()
  x_train = []
  y_train = []
  for i in range(len(xtrain)):
    x_train.append(xtrain[i])
    y_train.append([float(ytrain[i]), float(1-ytrain[i])])
  y_train = np.array(y_train)
  x_test = []
  y_test = []
  for i in range(len(xtest)):
    x_test.append(xtest[i])
    y_test.append([float(ytest[i]), float(1-ytest[i])])
  y_test = np.array(y_test)
  return x_train, y_train, x_test, y_test

def prepare_isot(path_data):
  #read dataset from csv
  df1 = pd.read_csv(path_data+"/ISOT/True.csv")
  df2 = pd.read_csv(path_data+"/ISOT/Fake.csv")
  # add column for labels
  df1["label"] = 0 # This indicates the News is True (means it's not Fake)
  df2["label"] = 1 # This indicates the News is False (means it's Fake)
  xx = df1["text"].tolist()
  xx = xx = [xx[i][min(50, max(0, 2+xx[i].find('-'))):] for i in range(len(xx))]
  df = pd.concat([df1,df2],ignore_index=True)
  x, y = xx+df2["text"].tolist(), df["label"].tolist()
  xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15, random_state=42)
  x_train = []
  y_train = []
  for i in range(len(xtrain)):
    x_train.append(xtrain[i])
    y_train.append([float(ytrain[i]), float(1-ytrain[i])])
  y_train = np.array(y_train)
  x_test = []
  y_test = []
  for i in range(len(xtest)):
    x_test.append(xtest[i])
    y_test.append([float(ytest[i]), float(1-ytest[i])])
  y_test = np.array(y_test)
  return x_train, y_train, x_test, y_test

def prepare_kaggle(path_data):
  df = pd.read_csv(path_data+'/sample2.csv', error_bad_lines=False)
  df_test = df[df['type'] == 'test']
  df_test.reset_index(inplace=True)
  df_train = df[df['type'] == 'training']
  df_train.reset_index(inplace=True)
  xtrain = df_train['content'].tolist()
  ytrain = df_train['label'].tolist()
  xtest = df_test['content'].tolist()
  ytest = df_test['label'].tolist()
  x_train = []
  y_train = []
  for i in range(len(xtrain)):
    x_train.append(xtrain[i])
    y_train.append([float(ytrain[i]), float(1-ytrain[i])])
  y_train = np.array(y_train)
  x_test = []
  y_test = []
  for i in range(len(xtest)):
    x_test.append(xtest[i])
    y_test.append([float(ytest[i]), float(1-ytest[i])])
  y_test = np.array(y_test)
  return x_train, y_train, x_test, y_test

def prepare_ag(path_data):
  train_df = pd.read_csv(path_data+'/AG/train.csv')
  test_df = pd.read_csv(path_data+'/AG/test.csv')
  ytrain = train_df['Class Index'].tolist()
  ytest = test_df['Class Index'].tolist()
  xtrain1 = train_df['Title'].tolist()
  xtrain2 = train_df['Description'].tolist()
  xtest1 = test_df['Title'].tolist()
  xtest2 = test_df['Description'].tolist()
  num_classes = len(list(set(ytrain)))
  x_train = []
  for i in range(len(xtrain1)):
    x_train.append(xtrain1[i]+xtrain2[i])
  x_test = []
  for i in range(len(xtest1)):
    x_test.append(xtest1[i]+xtest2[i])
  y_train = tf.keras.utils.to_categorical(np.array(ytrain)-1, num_classes)
  y_test = tf.keras.utils.to_categorical(np.array(ytest)-1, num_classes)
  return x_train, y_train, x_test, y_test
  
def prepare_imdb(path_data):
  train_df = pd.read_csv(path_data+'/IMDB/train.csv')
  test_df = pd.read_csv(path_data+'/IMDB/test.csv')
  ytrain = train_df['label'].tolist()
  ytest = test_df['label'].tolist()
  x_train = train_df['content'].tolist()
  x_test = test_df['content'].tolist()
  num_classes = len(list(set(ytrain)))
  y_train = tf.keras.utils.to_categorical(np.array(ytrain)-1, num_classes)
  y_test = tf.keras.utils.to_categorical(np.array(ytest)-1, num_classes)
  return x_train, y_train, x_test, y_test

def get_summed_values(train_df):
    summed_ = train_df[train_df.columns[2]]
    df_columns = train_df.columns[3:]
    for column in df_columns:
        summed_ = summed_ + train_df[column]
    return summed_

def make_even(x_tr_seq, y_train, class_arg):
    l0 = len(x_tr_seq[np.where(np.argmax(y_train, axis=1)!=class_arg)])
    l1 = len(x_tr_seq[np.where(np.argmax(y_train, axis=1)==class_arg)])
    l = min(l0, l1)
    x_tr_seq_0 = x_tr_seq[np.where(np.argmax(y_train, axis=1)!=class_arg)][:l]
    x_tr_seq_1 = x_tr_seq[np.where(np.argmax(y_train, axis=1)==class_arg)][:l]
    y_train_0 = y_train[np.where(np.argmax(y_train, axis=1)!=class_arg)][:l]
    y_train_1 = y_train[np.where(np.argmax(y_train, axis=1)==class_arg)][:l]

    x_tr_seq = np.append(x_tr_seq_0, x_tr_seq_1, axis=0)
    y_train = np.append(y_train_0, y_train_1, axis=0)

    x_tr_seq, y_train = shuffle(x_tr_seq, y_train)

    return x_tr_seq, y_train

def prepare_frame(path_data_one):
    train_df = pd.read_csv(path_data_one)
    summed_ = get_summed_values(train_df)
    new_train_df_indices = train_df.index[summed_>1]
    new_train_df = train_df.drop(new_train_df_indices)
    new_summed_ = get_summed_values(new_train_df)
    new_train_df['clean'] = 1 - new_summed_
    new_summed_ = get_summed_values(new_train_df)
    x_train = new_train_df['comment_text'].tolist()
    ytrain = []
    for column in new_train_df.columns[2:]:
        ytrain.append(new_train_df[column].tolist())
    y_train = np.array(ytrain).T
    return x_train, y_train

def prepare_toxic(path_data):
    from sklearn.model_selection import train_test_split
    x_train, y_train = prepare_frame(path_data+'/train.csv')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test

def prepare_data(path_data, data_name = 'kaggle'):
  if data_name == 'kaggle':
    x_train, y_train, x_test, y_test = prepare_kaggle(path_data)
  elif data_name == 'ISOT':
    x_train, y_train, x_test, y_test = prepare_isot(path_data)
  elif data_name == 'LIAR':
    x_train, y_train, x_test, y_test = prepare_liar(path_data)
  elif data_name == 'AG':
    x_train, y_train, x_test, y_test = prepare_ag(path_data)
  elif data_name == 'IMDB':
    x_train, y_train, x_test, y_test = prepare_imdb(path_data)
  elif data_name == 'toxic':
    x_train, y_train, x_test, y_test = prepare_toxic(path_data+'/Toxic')
 
  # print("samples of training data before tokenization; test samples are similar:\n", x_train[:3])
  return x_train, y_train, x_test, y_test

def data_preprocessing(x_train):
  for i in range(len(x_train)):
    words = re.sub(r'[^A-Za-z]', ' ', x_train[i]).split(" ")
    x_train[i] = ' '.join(x.strip() for x in words if x.strip())
  return x_train
 
def get_tokenizer(x_train, x_test):
  #Tokenize the sentences
  tokenizer = Tokenizer()
  #preparing vocabulary
  tokenizer.fit_on_texts(list(x_train)+list(x_test))
  size_of_vocabulary=len(tokenizer.word_index) + 1 #+1 for padding
  print("The size of the vocabulary is: ", size_of_vocabulary)
  return tokenizer, size_of_vocabulary
 
def get_tokenized(tokenizer, x_train, max_len=25):
  #converting text into integer sequences
  x_tr_seq  = tokenizer.texts_to_sequences(x_train)
  #padding to prepare sequences of same length
  x_tr_seq  = pad_sequences(x_tr_seq, maxlen=max_len, truncating='post')
  return x_tr_seq
 
def get_embedding_matrix(tokenizer, path_embedding):
  size_of_vocabulary=len(tokenizer.word_index) + 1 #+1 for padding
 
  # load the whole embedding into memory
  embeddings_index = dict()
  f = open(path_embedding, encoding="utf8")
 
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
 
  f.close()
  print('Loaded %s word vectors.' % len(embeddings_index))
 
  # create a weight matrix for words in training docs
  embedding_matrix = np.zeros((size_of_vocabulary, 300))
 
  for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
  return embedding_matrix

class custom_Tokenizer():
  def __init__(self, model_type, tokenizer, max_len=100):
    self.max_len = max_len
    self.model_type = model_type
    if self.model_type == 'bert':
      tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
      text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
      tokenizer_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
      text_output = tokenizer_layer(text_input)
      self.tokenizer = tf.keras.Model(text_input, text_output)
    else:
      self.tokenizer = tokenizer

  def texts_to_sequences(self, x_in):
    if self.model_type == 'bert':
      x_inter = self.tokenizer.predict(x_in)
      masks = x_inter['input_mask'][:,:self.max_len]
      type_ids = x_inter['input_type_ids'][:,:self.max_len]
      word_ids = x_inter['input_word_ids'][:,:self.max_len]
      x_out = [masks, type_ids, word_ids]
    else:
      x_inter = self.tokenizer.texts_to_sequences(x_in)
      x_out = pad_sequences(x_inter, maxlen=self.max_len, truncating='post')
    return x_out

def mlp(size_of_vocabulary, num_classes=2, Loss='categorical_crossentropy', embedding_matrix=None, max_len=25, n_neurons=([20]), opt='adam', l2='l2'):
  model2=Sequential()
  #embedding layer
  if embedding_matrix is not None:
    model2.add(Embedding(size_of_vocabulary,300,weights=[embedding_matrix],input_length=max_len,trainable=True))
  else:
    model2.add(Embedding(size_of_vocabulary,300,input_length=max_len))
  model2.add(Dropout(0.8))
  model2.add(Flatten())
  model2.add(Dense(int(n_neurons[0]), activation='relu', kernel_regularizer=l2))
  model2.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2))
 
  model2.compile(optimizer=opt, loss=Loss, metrics=['accuracy'])
 
  #Print summary of model2
  print(model2.summary())
  return model2
 
def rnn(size_of_vocabulary, num_classes=2, Loss='categorical_crossentropy', embedding_matrix=None, max_len=25, n_neurons=([100,64]), opt='adam', l2='l2'):
  model1 = Sequential()
  if embedding_matrix is not None:
    model1.add(Embedding(size_of_vocabulary,300,weights=[embedding_matrix],input_length=max_len,trainable=True))
  else:
    model1.add(Embedding(size_of_vocabulary,300,input_length=max_len))
  model1.add(Dropout(0.3))
  model1.add(LSTM(int(n_neurons[0]), kernel_regularizer=l2))
  model1.add(Dropout(0.3))
  model1.add(Dense(n_neurons[1],activation='relu', kernel_regularizer=l2))
  model1.add(Dropout(0.3))
  model1.add(Dense(num_classes,activation='softmax', kernel_regularizer=l2))
 
  model1.compile(loss=Loss, optimizer=opt, metrics=['accuracy'])
  print(model1.summary())
  return model1
 
def cnn(size_of_vocabulary, num_classes=2, Loss='categorical_crossentropy', embedding_matrix=None, max_len=25, n_neurons=([32,32,32]), opt='adam', l2='l2'):
  model2=Sequential()
  if embedding_matrix is not None:
    model2.add(Embedding(size_of_vocabulary,300,weights=[embedding_matrix],input_length=max_len,trainable=True))
  else:
    model2.add(Embedding(size_of_vocabulary,300,input_length=max_len))
  model2.add(Reshape(target_shape=(max_len,300,-1)))
  model2.add(Conv2D(int(n_neurons[0]),(3,3), kernel_regularizer=l2))
  model2.add(BatchNormalization())
  model2.add(Dropout(0.4))
  model2.add(Conv2D(int(n_neurons[1]),(3,3), kernel_regularizer=l2))
  model2.add(BatchNormalization())
  model2.add(Dropout(0.4))
  model2.add(Conv2D(int(n_neurons[2]),(5,5), kernel_regularizer=l2))
  model2.add(BatchNormalization())
  model2.add(Dropout(0.4))
  model2.add(Flatten())
  model2.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2))
 
  model2.compile(optimizer=opt, loss=Loss, metrics=['accuracy'])
 
  # es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
  # mc2=ModelCheckpoint('best_model2.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  
 
  #Print summary of model2
  print(model2.summary())
  return model2
 
class mask_embeddings_layer(keras.layers.Layer):
    def __init__(self, custom_dropout, batch_size=64, name="mask_layer"):
        if custom_dropout is None:
          self.custom_dropout = 0
        else:
          self.custom_dropout = custom_dropout
        self.batch_size = batch_size
        # self.name = name
        super(mask_embeddings_layer, self).__init__(name= name)

    def call(self, inputs):
        random = tf.random.uniform(shape=(self.batch_size, inputs.shape[1], 1), minval=0, maxval=inputs.shape[1], dtype=tf.int32)
        random_mask = 1/(1+tf.exp(1000*(-tf.cast(random, tf.float32)+self.custom_dropout-0.5)))
        return random_mask * inputs

def hybrid(size_of_vocabulary, num_classes=2, Loss='categorical_crossentropy', embedding_matrix=None, 
           max_len=25, n_neurons=([128,32]), opt='adam', l2=None, custom_dropout=None, verbose=True):
  
  model2=Sequential()
  
  if embedding_matrix is not None:
    model2.add(Embedding(size_of_vocabulary,300,weights=[embedding_matrix],input_length=max_len,trainable=True))
  else:
    model2.add(Embedding(size_of_vocabulary,300,input_length=max_len))
  
  if custom_dropout is not None:
    model2.add(mask_embeddings_layer(custom_dropout, name="mask_layer"))
  
  model2.add(Conv1D(int(n_neurons[0]),(5), activation='relu', kernel_regularizer=l2))
  model2.add(MaxPool1D(pool_size=2))
  model2.add(BatchNormalization())
  model2.add(LSTM(int(n_neurons[1]), kernel_regularizer=l2))
  # model2.add(Flatten())
  model2.add(Dense(num_classes, kernel_regularizer=l2))
  model2.add(Activation('softmax'))
 
  model2.compile(optimizer=opt, loss=Loss, metrics=['accuracy'])
  if verbose:
    print(model2.summary())
  return model2

def bert(max_seq_length, num_classes):
  tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
  input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
  segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_type_ids")
  input_dict = {'input_mask':input_mask, 'input_type_ids':segment_ids, 'input_word_ids':input_word_ids}
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(input_dict)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(net)
  classifier_model = tf.keras.Model(inputs=[input_mask, segment_ids, input_word_ids], outputs=net)
  tf.keras.utils.plot_model(classifier_model)
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  metrics = tf.metrics.BinaryAccuracy()
  epochs = 5
  steps_per_epoch = 625#tf.data.experimental.cardinality(train_ds).numpy()
  print(steps_per_epoch)
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)

  init_lr = 3e-5
  optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

  classifier_model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)
  return classifier_model


def confirm_directory(path):
  if not os.path.isdir(path):
    os.makedirs(path)
  return


def train_and_save_model(path, max_len=100, path_data="data", data_name="kaggle", custom_dropouts=None,
                        batch_size=64, epochs=7):
  x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = \
    load_tokenized_data(max_len=max_len, path_data=path_data, data_name=data_name, batch_size=batch_size,
                        make_even_flag=True)
  
  for custom_dropout in custom_dropouts:
    model = hybrid(size_of_vocabulary, custom_dropout=custom_dropout, 
                   num_classes=y_train.shape[1], max_len=max_len)
    model_name = "hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".h5"
    confirm_directory(path+"models/")
    if not os.path.isfile(path+"models/" + model_name):
      model.fit(x_tr_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val_seq, y_test))
      model.save_weights(path+"models/" + model_name)
    else:
      print("A trained model for this configuration already exists.")


def load_tokenized_data(max_len=100, path_data="data", data_name="kaggle",
                        batch_size=64, make_even_flag=False):
  class_args = {
    "kaggle": 0,
    "AG": 0,
    "toxic": 6
  }
  
  x_train, y_train, x_test, y_test = prepare_data(path_data, data_name=data_name)
  tokenizer, size_of_vocabulary = get_tokenizer(x_train, x_test)
  x_train = data_preprocessing(x_train)
  x_test = data_preprocessing(x_test)
  x_tr_seq = get_tokenized(tokenizer, x_train, max_len=max_len)
  x_val_seq = get_tokenized(tokenizer, x_test, max_len=max_len)
  
  if make_even_flag:
    print("train_length before evenizing: ", x_tr_seq.shape)
    x_tr_seq, y_train = make_even(x_tr_seq, y_train, class_arg=class_args[data_name])
    x_val_seq, y_test = make_even(x_val_seq, y_test, class_arg=class_args[data_name])
    print("train_length before evenizing: ", x_tr_seq.shape)
  
  print("train length before clipping was ", x_tr_seq.shape[0])
  x_tr_seq, y_train = x_tr_seq[:int(x_tr_seq.shape[0]/batch_size)*batch_size], \
                      y_train[:int(x_tr_seq.shape[0]/batch_size)*batch_size]
  x_val_seq, y_test = x_val_seq[:int(x_val_seq.shape[0]/batch_size)*batch_size], \
                      y_test[:int(x_val_seq.shape[0]/batch_size)*batch_size]
  print(" and the train length after clipping is ", x_tr_seq.shape[0])
  return x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary
  
def load_trained_model(path, size_of_vocabulary, custom_dropout=None, data_name="kaggle", num_classes=2, max_len=100):
  model = hybrid(size_of_vocabulary, custom_dropout=custom_dropout, 
                  num_classes=num_classes, max_len=max_len, verbose=False)
  model_name = "hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".h5"
  try:
    model.load_weights(path + "models/"+model_name)
    print("loaded weights.")
  except:
    print("Issue loading the model: "+path+"\"models/"+model_name+"\". Returning the untrained model.")
  return model
        
      
class custom_Model():
  def __init__(self, model_type, size_of_vocabulary, tokenizer, model_loss, max_len, num_classes,
               embedding_matrix , n_neurons, opt, regu):
    self.model_type = model_type.__name__
    self.tokenizer = tokenizer
    self.n_neurons = n_neurons
    self.num_classes = num_classes
    if model_type.__name__=='bert':
      self.model = model_type(max_len, num_classes=num_classes)
    else:
      self.model = model_type(size_of_vocabulary, Loss=model_loss, max_len=max_len, num_classes=num_classes,
                       embedding_matrix=embedding_matrix , n_neurons=n_neurons, opt=opt, l2=regu)
  
  def set_model(self, load_model_path):
    if self.model_type == 'bert':
      print('load weights')
      self.model.load_weights(load_model_path)
    else:
      self.model = load_model(load_model_path)

  def save_model(self, load_model_path):
    if self.model_type == 'bert':
      self.model.save_weights(load_model_path)
    else:
      self.model.save(load_model_path)
  
  def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size):
    if self.model_type == 'bert':
      self.model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs)
    else:
      self.model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
   
  def predict(self, x_in):
    return self.model.predict(x_in)

  def text_predict(self, x_in):
    text_array  = self.tokenizer.texts_to_sequences(x_in) 
    return self.model.predict(text_array)

  def text_learn(self, x_train, y_train, x_test, y_test, epochs, batch_size):
    text_len = 1000
    x_train = [x[:text_len] for x in x_train]
    x_test = [x[:text_len] for x in x_test]
    x_tr_seq = self.tokenizer.texts_to_sequences(x_train)
    x_val_seq = self.tokenizer.texts_to_sequences(x_test)
    if self.model_type == 'bert':
      history = self.model.fit(x_tr_seq, y_train,
                                    validation_data=(x_val_seq, y_test),
                                    epochs=epochs)
    else:
      self.model.fit(x_tr_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val_seq, y_test))


def get_cosine_similarity(arr1, arr2):
    return 1 - spatial.distance.cosine(arr1, arr2)


def show_results(arr1, arr2):
    diff = arr1 - arr2
    cosine_sim = get_cosine_similarity(arr1, arr2)
    corelation_coeff1, corelation_coeff2 = pearsonr(arr1, arr2)
    norm1 = LA.norm(diff, 1)
    norm2 = LA.norm(diff, 2)
    linf_norm =  LA.norm(diff, np.inf)
    return cosine_sim, corelation_coeff1, corelation_coeff2, norm1, norm2, linf_norm

def get_top_n(arr, n=10):
    arr = np.abs(arr)
    arr[::-1].sort()
    return arr[:n]
  
def visualize_activations(model, x_tr_seq, t=0.05, figure=False, bins=100, show_mean=True):
    prediction_vector = model.predict(x_tr_seq[:min(100, len(x_tr_seq))])
    prediction_vector_sorted = np.sort(prediction_vector, axis=1)
    max_values = prediction_vector_sorted[:, prediction_vector.shape[1]-1]
    second_max_values = prediction_vector_sorted[:, prediction_vector.shape[1]-2]
    t = min(t, 1.)
    t = max(t, 0.)
    min_threshold = second_max_values[np.argsort(-second_max_values)[int(t*min(100, len(x_tr_seq[:100])))-1]]
    max_threshold = max_values[np.argsort(max_values)[int(t*min(100, len(x_tr_seq[:100])))-1]]
    
    fig = None
    if figure:
      fig = plt.figure()
      plt.hist(max_values, bins=bins, label="$max(I(\\theta, x))$")
      plt.hist(second_max_values, bins=bins, label="$I^*(\\theta, x)$")
      if show_mean:
        plt.axvline(x=np.mean([min_threshold, max_threshold]), color='k', linestyle='--')
      plt.ylabel("Counts")
      plt.xlabel("$\\tau$")
      plt.legend()
      plt.tight_layout()
    
    return min_threshold, max_threshold, fig
    
def make_custom_ag_data(x_val_seq, y_test):
    x = x_val_seq[np.where(np.argmax(y_test, axis=1)==0)][:3]
    y = y_test[np.where(np.argmax(y_test, axis=1)==0)][:3]
    for binary_class in range(1, y_test.shape[1]):
        x = np.append(x, x_val_seq[np.where(np.argmax(y_test, axis=1)==binary_class)][:3], axis=0)
        y = np.append(y, y_test[np.where(np.argmax(y_test, axis=1)==binary_class)][:3], axis=0)
    return x, y