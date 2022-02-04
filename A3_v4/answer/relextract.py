import itertools
import json
import pandas as pd
import numpy as np
#import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')

from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from scipy.sparse import vstack

# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))

def print_example(data, index):
    """Prints a single example from the dataset. Provided only
        as a way of showing how to access various fields in the
        training and testing data.

        Args:
            data (list(dict)): A list of dictionaries containing the examples 
            index (int): The index of the example to print out.
    """
    # NOTE: You may Delete this function if you wish, it is only provided as 
    #   an example of how to access the data.
    
    # print the sentence (as a list of tokens)
    print("Tokens:")
    print(data[index]["tokens"])

    # print the entities (position in the sentence and type of entity)
    print("Entities:")
    for entity in data[index]["entities"]:
        print("%d %d %s" % (entity["start"], entity["end"], entity["label"]))
    
    # print the relation in the sentence if this is the training data
    if "relation" in data[index]:
        print("Relation:")
        relation = data[index]["relation"]
        print("%d:%s %s %d:%s" % (relation["a_start"], relation["a"],
            relation["relation"], relation["b_start"], relation["b"]))
    else:
        print("Test examples do not have ground truth relations.")

def write_output_file(relations, filename = "q3.csv"):
    """The list of relations into a csv file for the evaluation script

    Args:
        relations (list(tuple(str, str))): a list of the relations to write
            the first element of the tuple is the PERSON, the second is the
            GeoPolitical Entity
        filename (str, optional): Where to write the output file. Defaults to "q3.csv".
    """
    out = []
    for person, gpe in relations:
        out.append({"PERSON": person, "GPE": gpe})
    df = pd.DataFrame(out)
    df.to_csv(filename, index=False)

#! build a training/validation/testing pipeline for relation extraction
# then write the list of relations extracted from the *test set* to "q3.csv"
# using the write_output_file function.

# reference: https://devblogs.microsoft.com/cse/2016/09/13/training-a-classifier-for-relation-extraction-from-medical-literature/
#*##################################### pos and neg trainning data ###############################################
label_X_train = [] # list of example dictionary that has the relation nationality 
label_Y_train = [] # 0 - No relation 1 - nationality relation

label_X_test = [] # list of tokens  with a specific entities pair (PER, GPE) of testing data
X_test_entities = [] # coresponding entities pair of the label_X_test

# search for all the positive example and negative example
for i in range(len(train_data)):
    if train_data[i]["relation"]["relation"] == "/people/person/nationality":
        label_X_train.append(train_data[i])
        label_Y_train.append(1)
    elif "PERSON" in [e["label"] for e in train_data[i]["entities"]] and "GPE" in [e["label"] for e in train_data[i]["entities"]]:
        label_X_train.append(train_data[i])
        label_Y_train.append(0)
print(len(label_X_train))
print(len(label_Y_train))

#*##################################### trimmed data ###############################################
train_POS = [] # POS token list
test_POS = [] # POS token list

WINDOW_SIZE = 3 # the number of context words to be chosen
# trim the label_X_train to get the words between (PER, GPE) and the context words with WINDOW_SIZE = 3
for i in range(len(label_X_train)):
    sent    = label_X_train[i]
    a_start = sent["relation"]["a_start"]
    b_end   = sent["relation"]["b_start"] + len(sent["relation"]["b"]) - 1
    S       = max(a_start - WINDOW_SIZE , 0)                if a_start <= b_end else max(b_end - WINDOW_SIZE , 0)
    E       = min(b_end + WINDOW_SIZE, len(sent["tokens"])) if a_start <= b_end else min(a_start + WINDOW_SIZE, len(sent["tokens"]))  
    
    train_POS.append(" ".join(sent["pos"][S : E]))
    sent["tokens"] = sent["tokens"][S : E]
    
# process of finding the coresponding entities pair (PER, GPE) to the trimmed tokens list
for i in range(len(test_data)):
    sent    = test_data[i]
    PERSONs = [e for e in sent["entities"] if e["label"] == "PERSON"]
    GPEs    = [e for e in sent["entities"] if e["label"] == "GPE"]
    # if the sentence does not have the PERSONs and GPEs, we will ignore it.
    if len(PERSONs) > 0 and len(GPEs) > 0:
        # find all possible combinations of the (PER, GPE) pair 
        for pe, ge in itertools.product(PERSONs, GPEs): 
            a_start = pe["start"]
            a_end   = pe["end"]
            b_start = ge["start"]
            b_end   = ge["end"]
            S       = max(a_start - WINDOW_SIZE , 0)                if a_start <= b_end else max(b_end - WINDOW_SIZE , 0)
            E       = min(b_end + WINDOW_SIZE, len(sent["tokens"])) if a_start <= b_end else min(a_start + WINDOW_SIZE, len(sent["tokens"]))
            
            label_X_test.append(sent["tokens"][S : E])
            test_POS.append(" ".join(sent["pos"][S : E]))
            X_test_entities.append((" ".join(sent["tokens"][a_start:a_end]), " ".join(sent["tokens"][b_start:b_end])))

print(f'X_test: {len(label_X_test)}, test_POS: {len(test_POS)}')
print(f'X_train: {len(label_X_train)}, train_POS: {len(train_POS)}')

#*##################################### Normalization ###############################################
# do the Steming, stopwords removal, punctuation removal, digits removal.
# and transform the element in the label_X_train or label_X_test to a string by joinning the tokens
porter = nltk.PorterStemmer()
for i in range(len(label_X_train)):
    cleaned_tokens = []
    for t in label_X_train[i]["tokens"]:
        if (t in nltk.corpus.stopwords.words('english') or t.isdigit() or len(t) < 2):
            continue
        stemmed = porter.stem(t)
        cleaned_tokens.append(stemmed)

    label_X_train[i] = " ".join(cleaned_tokens)


for i in range(len(label_X_test)):
    cleaned_tokens = []
    for t in label_X_test[i]:
        if (t in nltk.corpus.stopwords.words('english') or t.isdigit() or len(t) < 2):
            continue
        stemmed = porter.stem(t)
        cleaned_tokens.append(stemmed)

    label_X_test[i] = " ".join(cleaned_tokens)

#*##################################### regression ###############################################
# BOW of the corpus using bigram
vectorizer_data = CountVectorizer(analyzer = "word", binary = True, ngram_range=(2,2))
data_features   = vectorizer_data.fit_transform(label_X_train + label_X_test)
feature_X_train = data_features[:len(label_X_train)]
feature_X_test  = data_features[len(label_X_train):]

# BOW of the POS using bigram
vectorizer_pos = CountVectorizer(analyzer = "word", binary = True, ngram_range=(2,2))
pos_features   = vectorizer_pos.fit_transform(train_POS + test_POS)
pos_X_train    = pos_features[:len(train_POS)]
pos_X_test     = pos_features[len(train_POS):]

# combine two feature together
final_X_train = hstack((feature_X_train, pos_X_train))
final_X_test = hstack((feature_X_test, pos_X_test))

# train valid test split
x_train, x_test_valid, y_train, y_test_vlid = train_test_split(final_X_train, label_Y_train, test_size=0.25, stratify=label_Y_train)
x_valid, x_test, y_valid, y_test = train_test_split(x_test_valid, y_test_vlid, test_size=0.33, stratify=y_test_vlid)

# tunning the hyperparameters
C_set = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5]
Scores = []

for c in C_set:
    clf = LogisticRegression(C=c, solver="newton-cg")
    clf.fit(x_train, y_train)
    y_pred_valid = clf.predict(x_valid)
    score = accuracy_score(y_valid, y_pred_valid)
    Scores.append(score)
Best_C = C_set[max([(s, i) for (i, s) in enumerate(Scores)])[1]] 

clf = LogisticRegression(C=Best_C, solver="newton-cg")
clf.fit(vstack((x_train, x_valid)), y_train + y_valid)
y_pred_valid = clf.predict(x_test)
print(classification_report(y_test, y_pred_valid))

# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file
#! remove this and write out the relations you extracted (obviously don't hard code them)

# post-processing: finding the highest probability of nationality of each person should belongs to
y_pred_raw = clf.predict_proba(final_X_test)
y_pred     = clf.predict(final_X_test)
print(list(y_pred).count(0))
print(list(y_pred).count(1))
relation_prob = {}
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        PERSON = X_test_entities[i][0]
        GPE    = X_test_entities[i][1]
        if PERSON not in relation_prob:
            relation_prob[PERSON] = (y_pred_raw[i][1], GPE)
        elif y_pred_raw[i][1] > relation_prob[PERSON][0]:
            relation_prob[PERSON] = (y_pred_raw[i][1], GPE)
        else:
            continue
relations = []
for PERSON, GPE_pair in relation_prob.items():
    GPE = GPE_pair[1]
    relations.append((PERSON, GPE))
write_output_file(relations)