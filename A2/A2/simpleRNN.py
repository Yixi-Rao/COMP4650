# LSTM with dropout for sequence classification 
from keras.backend import dropout
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.preprocessing import sequence,text
from keras.layers.embeddings import Embedding
import pandas as pd


# fix random seed for reproducibility
numpy.random.seed(7)

#fetching sms spam dataset
url = 'https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])

#binarizing
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
sms.head()

X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)

###################################
tk = text.Tokenizer(num_words=200, lower=True)
tk.fit_on_texts(X)

x = tk.texts_to_sequences(X)

print(len(tk.word_counts))

###################################
max_len = 80
print("max_len "+ str(max_len))
print('Pad sequences (samples x time)')

x = sequence.pad_sequences(x, maxlen=max_len)



max_features = 200
model = Sequential()
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x, y=y, batch_size=500, epochs=1, verbose=1, validation_split=0.2,  shuffle=True)