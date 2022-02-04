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
train_data = json.load(open("/content/drive/MyDrive/Colab_Notebooks/genre_train.json", "r"))

X    = train_data['X']
Y_raw  = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
Y    = np.asarray(Y_raw)
docid  = train_data['docid'] # these are the ids of the books which each training example came from

X_train =[] 
Y_train =[] 
X_valid =[] 
Y_valid =[]
X_test =[] 
Y_test =[]

def train_valid_test(X, Y, docid, ratio=4):
    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []

    idcount_dict = {}
    valid_docid = []
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

X_train, Y_train, X_valid, Y_valid, valid_docid = train_valid_test(X,Y,docid)
X_valid, Y_valid, X_test, Y_test,_ = train_valid_test(X_valid,Y_valid,valid_docid,2)

class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(Y_train), y=Y_train) 
class_weights_dict = dict(enumerate(class_weights))  
weight_dict = {0 : len(Y_train)/Y_train.count(0), 1 : len(Y_train)/Y_train.count(1), 2 : len(Y_train)/Y_train.count(2), 3 : len(Y_train)/Y_train.count(3)}
weight_dict2 = {0 : len(Y_valid)/Y_valid.count(0), 1 : len(Y_valid)/Y_valid.count(1), 2 : len(Y_valid)/Y_valid.count(2), 3 : len(Y_valid)/Y_valid.count(3)}

print("class_weight_dict: " + str(class_weights_dict))
print("weight_dict: " + str(weight_dict))

# ros = RandomOverSampler(random_state=0)
# X_resampled, Y_resampled = ros.fit_resample(np.asarray(X_train).reshape(-1,1), np.asarray(Y_train).reshape(-1, 1))

#! load the test data the test data does not have labels, our model needs to generate these
test_data = json.load(open("/content/drive/MyDrive/Colab_Notebooks/genre_test.json", "r"))
Xt = test_data['X']

#! tokenizing load the Glove
embeddings_dict = dict()
f = open('/content/drive/MyDrive/Colab_Notebooks/glove.6B.100d.txt', encoding='utf8')
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

embedding_matrix = np.zeros((len(w2i_dict) + 1, 100))
for word, index in w2i_dict.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

class GRUModel(object):
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
                batch_size=16, epochs=125, shuffle=True, verbose=1,
                class_weight=weight_dict,
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

class BiLSTMModel(object):
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