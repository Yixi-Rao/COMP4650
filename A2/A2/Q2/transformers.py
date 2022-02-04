import os

import json

import numpy as np
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


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

weight_dict = {0 : len(Y_train)/Y_train.count(0), 1 : len(Y_train)/Y_train.count(1), 2 : len(Y_train)/Y_train.count(2), 3 : len(Y_train)/Y_train.count(3)}
#! load the test data the test data does not have labels, our model needs to generate these
test_data = json.load(open("/content/drive/MyDrive/Colab_Notebooks/genre_test.json", "r"))
Xt = test_data['X']

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

# Bert layers
text_input        = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs           = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.4, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(4, activation='softmax', name="output")(l)

# Use inputs and outputs to construct a final model
model  = tf.keras.Model(inputs=[text_input], outputs = [l])

METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=METRICS)

model.fit(np.asarray(X_train), to_categorical(np.asarray(Y_train)),
          batch_size=32,shuffle=True, verbose=1,
          epochs=10, 
          validation_data=(np.asarray(X_valid),to_categorical(np.asarray(Y_valid))),
          class_weight = weight_dict)

print(model.evaluate(np.asarray(X_test), to_categorical(np.asarray(Y_test))))

#! predict on the test data
Y_raw=model.predict(Xt)
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