from keras.layers import LSTM, Dropout, Dense, GRU, Bidirectional
from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
import tqdm
import numpy as np
from pathlib import Path
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

warnings.filterwarnings("ignore")
# embedding_size = 100
# sequence_length = 100
TEST_SIZE = 0.5
FILTERS = 70
BATCH_SIZE = 100
EPOCHS = 5
OUTPUT_FOLDER = "save"
lstm_units = 1024
embeddingFile = ""
 #converts the utf-8 into tokinized characters 
 

import urllib.request
import zipfile
import tarfile
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
def preprare():
    fname = 'embed/wiki-news-300d-1M.vec'
    fnameGlove = 'embed/glove.6B.100d.txt'
    # https://github.com/allenai/spv2/blob/master/model/glove.6B.100d.txt.gz
    if not os.path.isfile(fname):
        print('Downloading word vectors of Word to vector')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                              'wiki-news-300d-1M.vec.zip')
    if not os.path.isfile(fnameGlove):
        print('Downloading word vectors of Glove')
    urllib.request.urlretrieve('https://github.com/allenai/spv2/blob/master/model/glove.6B.100d.txt.gz',
                              'glove.6B.100d.txt.gz')
    print('Unzipping Word to vector...')
    with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
        zip_ref.extractall('embed')
    print('Unzipping ...')
    with tarfile.open('glove.6B.100d.txt.gz', 'r:gz') as tar_ref:
        tar_ref.extractall('embed')
    print('done.')    
    os.remove('wiki-news-300d-1M.vec.zip')
    os.remove('glove.6B.100d.txt.gz')
    return
#Load data
def get_embedding_vectors(tokenizer, embed_mode):
    word_index = tokenizer.word_index
    embedding_index = {}
    preprare()
    if embed_mode == "glove":
        embeddingFile = "embed/glove.6B.100d.txt"
        embedding_size = 100
    else:
        embeddingFile = "embed/wiki-news-300d-1M.vec"
        embedding_size = 300
    with open(embeddingFile, 'r',encoding='utf8',errors = 'ignore') as f:
        for line in tqdm.tqdm(f, "Reading embedding"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    embedding_matrix = np.zeros((len(word_index)+1, embedding_size))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
def get_train_data(train_data_dir, val_size, embed_mode):
    """load urls from the specified directory"""
    if Path(train_data_dir).is_file() and Path(train_data_dir).suffix == '.csv':
        data = pd.read_csv(train_data_dir,sep=',').sample(n = 2000, random_state = 4)
        urls, temp = list(data['url']), list(data['Label'])
        labels = []
        for l in temp:
            if l == 1:
                labels.append(l)
            else:
                labels.append(0)
        if embed_mode == "glove":
            sequence_length = 100
        else:
            sequence_length = 300
        temp_trainX, temp_valX, temp_trainY, temp_valY = train_test_split(urls, labels, test_size=val_size, random_state=4)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(urls)
        trainX = tokenizer.texts_to_sequences(temp_trainX)
        trainX = np.array(trainX)
        trainY = np.array(temp_trainY)
        trainX = pad_sequences(trainX, maxlen=sequence_length)
        trainY = tf.keras.utils.to_categorical(trainY)
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        valX = tokenizer.texts_to_sequences(temp_valX)
        valX = np.array(valX)
        valY = np.array(temp_valY)
        valX = pad_sequences(valX, maxlen=sequence_length)
        valY = tf.keras.utils.to_categorical(valY)
        valX, valY = shuffle(valX, valY, random_state=0)

        return trainX, valX, trainY, valY, tokenizer
    return



def get_model(tokenizer, embedding_matrix, rnn_cell, embed_mode): # builds the lstm model
    if embed_mode == "glove":
        embedding_size = sequence_length = 100
    else:
        embedding_size = sequence_length = 300
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 
            embedding_size,
            weights=[embedding_matrix],
            trainable=False,
            input_length=sequence_length))
    if rnn_cell == "lstm":
        model.add(LSTM(lstm_units, recurrent_dropout=0.3))
    elif rnn_cell == "gru":
        model.add(GRU(lstm_units, recurrent_dropout=0.3))
    elif rnn_cell == "bilstm":
        # First layer of BiLSTM
        model.add(Bidirectional(LSTM(units = lstm_units, return_sequences=True)))
        # Second layer of BiLSTM
        model.add(Bidirectional(LSTM(units = lstm_units)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax")) #probobility studff 
    # rmsprop better than adam 
    #weights[0] = weights[0].reshape(list(reversed(weights[0].shape)))
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()
    return model

def get_test_data(train_data_dir, tokenizer, embed_mode):
    """load urls from the specified directory"""
    if Path(train_data_dir).is_file() and Path(train_data_dir).suffix == '.csv':
        data = pd.read_csv(train_data_dir,sep=',').sample(n = 2000, random_state = 2)
        urls, temp = list(data['url']), list(data['Label'])
        labels = []
        for l in temp:
            if l == 1:
                labels.append(l)
            else:
                labels.append(0)
        if embed_mode == "glove":
            sequence_length = 100
        else:
            sequence_length = 300
        trainX = tokenizer.texts_to_sequences(urls)
        trainX = np.array(trainX)
        trainY = np.array(labels)
        trainX = pad_sequences(trainX, maxlen=sequence_length)
        trainY = tf.keras.utils.to_categorical(trainY)
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        return trainX, trainY
    return

def get_sample_data(url_sample, tokenizer, embed_mode):
    """load urls from the specified directory"""
    if url_sample != '' and url_sample != None:
        if embed_mode == "glove":
            sequence_length = 100
        else:
            sequence_length = 300
        sample = tokenizer.texts_to_sequences(url_sample)
        sample = np.array(sample)
        sample = pad_sequences(sample, maxlen=sequence_length)
        sample = shuffle(sample, random_state=0) 
    return sample