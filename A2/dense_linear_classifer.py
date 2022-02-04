import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

tokenizer = nltk.TreebankWordTokenizer()

# read the data Q1\data\labelled_movie_reviews.csv
df = pd.read_csv("Q1/data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]
# Q1\data\word_vectors.npz
data = dict(np.load("Q1/data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}

# convert a document into a vector
def document_to_vector(doc):
    """Takes a string document and turns it into a vector
        by aggregating its word vectors.

        Args:
            doc (str): The document as a string

        Returns:
            np.array: The word vector this will be 300 dimensionals.
    """
    #: tokenize the input document
    doc_words = tokenizer.tokenize(doc)
    doc_vecs = [w2v[w] for w in doc_words if w in w2v]

    #: aggregate the vectors of words in the input document
    vec = np.mean(doc_vecs, axis=0)

    return vec
            

# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

        Args:
            Xtr (list(str)): The input training examples. Each example is a
                document as a string.
            Ytr (list(str)): The list of class labels, each element of the 
                list is either 'neg' or 'pos'.
            C (float): Regularization parameter C for LogisticRegression

        Returns:
            LogisticRegression: The trained logistic regression model.
    """
    #: convert each of the training documents into a vector
    Xtr_v = np.array([document_to_vector(doc) for doc in Xtr])
    #: train the logistic regression classifier

    model = LogisticRegression(solver='liblinear', C=C).fit(Xtr_v, Ytr)
    return model

# fit a linear model 
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

        Args:
            model (LogisticRegression): The previously trained model.
            Xtst (list(str)): The input examples. Each example
                is a document as a string.
            Ytst (list(str)): The input class labels, each element
                of the list is either 'neg' or 'pos'.

        Returns:
            float: The accuracy of the model on the data.
    """
    #: convert each of the testing documents into a vector
    Xtst_v = [document_to_vector(doc) for doc in Xtst]
    #: test the logistic regression classifier and calculate the accuracy
    Ypdt = model.predict(Xtst_v)
    score = accuracy_score(Ytst, Ypdt)
    return score


#: search for the best C parameter using the validation set, -4, -3, -2, -1
C_set      = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
base_set   = [2, 3, 4, 5, 6, 7, 8, 9, 10]
C_score    = []
base_score = []
print("C    set: " + str(C_set)) 
print("base set: " + str(base_set))

for b in base_set:
    model = fit_model(X_train, Y_train, b ** -2)  
    base_score.append(test_model(model, X_val, Y_val))
    
print("base score: " + str(base_score))    
Best_b = base_set[max([(s, i) for (i, s) in enumerate(base_score)])[1]]    
print("Best b: " + str(Best_b))

for c in C_set:
    model = fit_model(X_train, Y_train, Best_b ** c)  
    C_score.append(test_model(model, X_val, Y_val))

print("C_score: " + str(C_score))    
Best_C = C_set[max([(s, i) for (i, s) in enumerate(C_score)])[1]]    
print("Best_C: " + str(Best_C))

#: fit the model to the concatenated training and validation set
#   test on the test set and print the result
print("---------------------------------------------------------------------")
model = fit_model(X_train + X_val, Y_train + Y_val, Best_b ** Best_C)
print("Final accuracy: " + str(test_model(model, X_test, Y_test)))


