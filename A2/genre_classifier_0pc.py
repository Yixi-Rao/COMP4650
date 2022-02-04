
# NOTE: This file contains is a very poor model which looks for manually 
# chosen keywords and if none are found it predicts randomly according
# to the class distribution in the training set
import os

from keras.callbacks import EarlyStopping

from keras.layers.recurrent import GRU

import json

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import GRU
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler

#! load the training data
train_data = json.load(open("genre_train.json", "r"))

X     = train_data['X']
Y_raw = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
Y     = np.asarray(Y_raw)
docid = train_data['docid'] # these are the ids of the books which each training example came from

X_train = [] 
Y_train = [] 
X_valid = [] 
Y_valid = []
X_test  = [] 
Y_test  = []

def train_valid_test(X, Y, docid, ratio=4):
    '''split the X and Y accroding to the ratio and the docid. 
        The ratio of docid in X_valid equals to the ratio of docid in X_train
    '''
    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []

    idcount_dict = {}
    valid_docid  = []
    for i, doc in enumerate(X):
        doc_id = docid[i]
        if doc_id in idcount_dict:
            idcount_dict[doc_id] = idcount_dict[doc_id] + 1
            num_doc              = idcount_dict[doc_id]
            if (num_doc % ratio) == 0:
                valid_docid.append(doc_id)
                X_valid.append(doc)
                Y_valid.append(Y[i])
            else:
                X_train.append(doc)
                Y_train.append(Y[i])
        else:
            idcount_dict[doc_id] = 1
            X_train.append(doc)
            Y_train.append(Y[i])

    return X_train, Y_train, X_valid, Y_valid, valid_docid

X_train, Y_train, X_valid, Y_valid, valid_docid = train_valid_test(X, Y, docid)
X_valid, Y_valid, X_test,  Y_test,  _           = train_valid_test(X_valid, Y_valid, valid_docid, 2)
# weights of the different Genres for solving the unbalanced dataset problem
weight_dict  = {0 : len(Y_train)/Y_train.count(0), 1 : len(Y_train)/Y_train.count(1), 2 : len(Y_train)/Y_train.count(2), 3 : len(Y_train)/Y_train.count(3)}
weight_dict2 = {0 : len(Y_valid)/Y_valid.count(0), 1 : len(Y_valid)/Y_valid.count(1), 2 : len(Y_valid)/Y_valid.count(2), 3 : len(Y_valid)/Y_valid.count(3)}

print("zero  : " + str(Y_raw.count(0)))
print("one   : " + str(Y_raw.count(1)))
print("two   : " + str(Y_raw.count(2)))
print("three : " + str(Y_raw.count(3)))
print("Y_raw :" + str(len(Y_raw)))
print("zero  : " + str(Y_train.count(0)))
print("one   : " + str(Y_train.count(1)))
print("two   : " + str(Y_train.count(2)))
print("three : " + str(Y_train.count(3)))
print("Y_train :" + str(len(Y_train)))
print("Y_valid :" + str(len(Y_valid)))

#! load the test data the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']

#! tokenizing load the Glove and creating word vector
embeddings_dict = dict()
f = open('glove.6B.100d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    vec = np.array(values[1:], dtype='float32')
    embeddings_dict[word] = vec
f.close()
tok = Tokenizer()
tok.fit_on_texts(X)
tok.fit_on_texts(Xt)
w2i_dict = tok.word_index

embedding_matrix = np.zeros((len(w2i_dict) + 1, 100))
for word, index in w2i_dict.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

#! model part: the final solution is GRU Model
class KeywordModel(object):
    def __init__(self):
        self.counts = None

    def fit(self, Y):
        # fit the model
        # normally you would want to use X the training data but this simple model doesn't need it
        self.counts = np.array(np.bincount(Y), dtype=np.float)
        self.counts /= np.sum(self.counts)
    
    def predict(self, Xin):
        Y_test_pred = []
        for x in Xin:
            # split into words
            xs = x.lower().split()

            # check if for our keywords
            if "scary" in xs or "spooky" in xs or "raven" in xs: # horror book
                Y_test_pred.append(0)
            elif "science" in xs or "space" in xs: # science fiction book
                Y_test_pred.append(1)
            elif "funny" in xs or "embarrassed" in xs: # humor book
                Y_test_pred.append(2)
            elif "police" in xs or "murder" in xs or "crime" in xs: # crime fiction book
                Y_test_pred.append(3)
            else: 
                Y_test_pred.append(np.random.choice(len(self.counts), p=self.counts)) # predict randomly
        return Y_test_pred

class LSTMModel(object):
    '''abandoned
    '''
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.model = None
    
    def create_model_input(self, X):
        index_sequence     = tok.texts_to_sequences(X)
        pad_index_sequence = pad_sequences(index_sequence, maxlen=822)
        return pad_index_sequence
        
    def fit(self, X, Y):
        # fit the model
        model = Sequential()
        
        model.add(Embedding(len(w2i_dict) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=822,
                            trainable=False))
        model.add(LSTM(200, dropout=0.3))

        model.add(Dense(4, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max', min_delta=0.0001)
        model.fit(self.create_model_input(X), to_categorical(np.asarray(Y)),
                  batch_size=32, epochs=20, shuffle=True, verbose=1,
                  validation_split=0.4,
                  callbacks=[early_stop])
        
        self.model = model
        return model
    
    def predict(self, Xin):
        Y_test_pred = self.model.predict(self.create_model_input(Xin), verbose=1)
        return Y_test_pred
    
    def evaluate(self, X, Y):
        loss, accuracy = self.model.evaluate(self.create_model_input(X), to_categorical(np.asarray(Y)), verbose=1)
        return loss, accuracy

class MultiLSTMModel(object):
    '''abandoned
    '''
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.model = None
    
    def create_model_input(self, X):
        index_sequence     = tok.texts_to_sequences(X)
        pad_index_sequence = pad_sequences(index_sequence, maxlen=822)
        return pad_index_sequence
        
    def fit(self, X, Y):
        # fit the model
        model = Sequential()
        
        model.add(Embedding(len(w2i_dict) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=822,
                            trainable=False))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(64))
        model.add(Dense(4, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', min_delta=0.0003)
        model.fit(self.create_model_input(X), to_categorical(np.asarray(Y)),
                  batch_size=48, epochs=20, shuffle=True, verbose=1,
                  validation_split=0.1,
                  callbacks=[early_stop])
        
        self.model = model
        return model
    
    def predict(self, Xin):
        Y_test_pred = self.model.predict(self.create_model_input(Xin), verbose=1)
        return Y_test_pred
    
    def evaluate(self, X, Y):
        loss, accuracy = self.model.evaluate(self.create_model_input(X), to_categorical(np.asarray(Y)), verbose=1)
        return loss, accuracy
    
class GRUModel(object):
    '''final solution
    '''
    def __init__(self):
        self.model = None
    
    def create_model_input(self, X):
        index_sequence     = tok.texts_to_sequences(X)
        pad_index_sequence = pad_sequences(index_sequence, maxlen=900)
        return pad_index_sequence
        
    def fit(self, X_T, Y_T, X_V, Y_V, X_tt, Y_tt):
        model = Sequential()
        # first layer: Embedding layer
        model.add(Embedding(len(w2i_dict) + 1,
                  100,
                  weights=[embedding_matrix],
                  input_length=900,
                  trainable=False))
        # second layer: GRU layer (use dropout to prevent overfitting)
        model.add(GRU(54, dropout=0.5))
        # last layer: Dense layer 
        model.add(Dense(4, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', min_delta=0.0001)

        model.fit(self.create_model_input(X_T), to_categorical(np.asarray(Y_T)),
                batch_size=16, epochs=125, shuffle=True, verbose=1,
                class_weight=weight_dict,
                validation_data = (self.create_model_input(X_V), to_categorical(np.asarray(Y_V))),
                callbacks=[early_stop]
                )
        
        model.evaluate(self.create_model_input(X_tt), to_categorical(np.asarray(Y_tt)), verbose=1)
                
        self.model = model
        return model
    
    def predict(self, Xin):
        Y_test_pred = self.model.predict(self.create_model_input(Xin), verbose=1)
        return Y_test_pred
    
    def evaluate(self, X, Y):
        loss, accuracy = self.model.evaluate(self.create_model_input(X), to_categorical(np.asarray(Y)), verbose=1)
        return loss, accuracy

class BiLSTMModel(object):
    '''abandoned
    '''
    def __init__(self):
        self.model = None
    
    def create_model_input(self, X):
        index_sequence     = tok.texts_to_sequences(X)
        pad_index_sequence = pad_sequences(index_sequence, maxlen=900)
        return pad_index_sequence
        
    def fit(self, X_T, Y_T, X_V, Y_V, X_tt, Y_tt):

        model = Sequential()
        
        model.add(Embedding(len(w2i_dict) + 1,
                  100,
                  weights=[embedding_matrix],
                  input_length=900,
                  trainable=False))
        
        model.add(GRU(54, dropout=0.5))

        model.add(Dense(4, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', min_delta=0.0001)

        model.fit(self.create_model_input(X_T), to_categorical(np.asarray(Y_T)),
                batch_size=19, epochs=125, shuffle=True, verbose=1,
                class_weight=weight_dict2,
                validation_data = (self.create_model_input(X_V), to_categorical(np.asarray(Y_V)))
                )
        
        model.evaluate(self.create_model_input(X_tt), to_categorical(np.asarray(Y_tt)), verbose=1)
   
        self.model = model
        return model
    
    def predict(self, Xin):
        Y_test_pred = self.model.predict(self.create_model_input(Xin), verbose=1)
        return Y_test_pred
    
    def evaluate(self, X, Y):
        loss, accuracy = self.model.evaluate(self.create_model_input(X), to_categorical(np.asarray(Y)), verbose=1)
        return loss, accuracy

#!fit the model on the training data
model = GRUModel()
model.fit(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
print(model.model.summary())
print(model.evaluate(X_test, Y_test))

#! predict on the test data
Y_raw       = model.predict(Xt)
Y_test_pred = np.argmax(Y_raw, axis=1)

print("zero  : " + str(Y_test_pred.tolist().count(0)))
print("one   : " + str(Y_test_pred.tolist().count(1)))
print("two   : " + str(Y_test_pred.tolist().count(2)))
print("three : " + str(Y_test_pred.tolist().count(3)))

#! write out the csv file
#! first column is the id, it is the index to the list of test examples
#! second column is the predction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()