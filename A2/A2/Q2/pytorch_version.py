# NOTE: This file contains is a very poor model which looks for manually
# chosen keywords and if none are found it predicts randomly according
# to the class distribution in the training set

import json

import numpy as np
import torch
import torch.nn as nn
from torchtext.legacy.data import Field, Example
from torchtext.vocab import GloVe, Vectors
import tqdm

#! load the training data
train_data = json.load(
    open("/content/drive/MyDrive/Colab_Notebooks/genre_train.json", "r"))

X = train_data['X']
# id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
Y = train_data['Y']
Y = np.asarray(Y)
# these are the ids of the books which each training example came from
docid = train_data['docid']
X_train = []
Y_train = []
X_valid = []
Y_valid = []

idcount_dict = {}
for i, doc in enumerate(X):
    doc_id = docid[i]
    if doc_id in idcount_dict:
        idcount_dict[doc_id] = idcount_dict[doc_id] + 1
        num_doc = idcount_dict[doc_id]
        if (num_doc % 4) == 0:
            X_valid.append(doc)
            Y_valid.append(Y[i])
        else:
            X_train.append(doc)
            Y_train.append(Y[i])
    else:
        idcount_dict[doc_id] = 1
        X_train.append(doc)
        Y_train.append(Y[i])
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)


#! load the test data the test data does not have labels, our model needs to generate these
test_data = json.load(
    open("/content/drive/MyDrive/Colab_Notebooks/genre_test.json", "r"))
Xt = test_data['X']

TEXT = Field(sequential=True, lower=True, tokenize='spacy', batch_first=True)
LABEL = Field(sequential=False, use_vocab=False)
fields = [("comment", TEXT), ("label", LABEL)]

examples_train = []
examples_test = []

for text, label in zip(X, Y):
    example = Example.fromlist([text, label], fields=fields)
    examples_train.append(example)
for text in Xt:
    example = Example.fromlist([text, None], fields=fields)
    examples_test.append(example)

new_corpus = [example.comment for example in examples_train + examples_test]
X_train = new_corpus[0:len(examples_train)]
X_test  = new_corpus[len(examples_train):]

if not os.path.exists('.vector_cache'):
    os.mkdir('.vector_cache')
vectors = Vectors(
    name='/content/drive/MyDrive/Colab_Notebooks/glove.6B.100d.txt')
TEXT.build_vocab(new_corpus, vectors=vectors)


class RNNLM(nn.Module):
    def __init__(self, vocab_size, emb_size=100, gru_size=128):
        super(RNNLM, self).__init__()

        # store layer sizes
        self.emb_size = emb_size
        self.gru_size = gru_size

        # for embedding characters (ignores those with value 0: the padded values)
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        # GRU layer
        self.gru = nn.GRU(emb_size, gru_size, batch_first=True)
        # linear layer for output
        self.linear = nn.Linear(gru_size, 4)

    def forward(self, x, h_last=None):

        emb_x = self.emb(x)
        out, h = self.gru(emb_x, h_last)
        out = self.linear(out)
        return out, h


def train_model(model, Xtrain, Ytrain, Xval, Yval, max_epoch):

    # construct the adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    # construct the cross-entropy loss function
    # we want to ignore padding cells with value == 1
    lossfn = nn.CrossEntropyLoss(ignore_index=1)

    # calculate number of batches
    batch_size = 32
    num_batches = int(Xtrain.shape[0] / batch_size)
    if Xtrain.shape[0] % batch_size != 0:
        num_batches += 1

    # run the main training loop over many epochs
    for epoche in tqdm.tqdm(range(max_epoch)):
        # NOTE: implement the training loop of the RNNLM model
        for b in range(num_batches):
            s = b * batch_size
            e = (b + 1) * batch_size
            if e > Xtrain.shape[0]:
                e = Xtrain.shape[0]

            yout, _ = model(Xtrain[s: e])
            optim.zero_grad()
            loss = lossfn(yout[i], Ytrain[s: e])
            loss.backward()
            optim.step()

        print(f'epoch {epoche}, loss has value:')
        print(calc_val_loss(model, Xval, Yval))
        print('\n')


#!fit the model on the training data
model = RNNLM(len(TEXT.vocab), 100, 128)
pre_weight_matrix = TEXT.vocab.vectors
model.emb.weight.data.copy_(pre_weight_matrix)

train_model(model, TEXT.process(X_train), Y_train, TEXT.process(X_valid), Y_valid, 10)

#! predict on the test data
Y_raw = model.predict(Xt)
Y_test_pred = np.argmax(Y_raw, axis=1)

print("predicting finished")
print("zero  : " + str(Y_test_pred.tolist().count(0)))
print("one   : " + str(Y_test_pred.tolist().count(1)))
print("two   : " + str(Y_test_pred.tolist().count(2)))
print("three : " + str(Y_test_pred.tolist().count(3)))

#! write out the csv file
#! first column is the id, it is the index to the list of test examples
#! second column is the predction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
# Y_test_pred is in the same order as the test data
for i, line in enumerate(Y_test_pred):
    fout.write("%d,%d\n" % (i, line))
fout.close()
