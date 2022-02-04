import numpy as np
import glob
import os
import pickle
from collections import defaultdict

from string_processing import *


def read_doc(file_path):
    """Read a document from a path, tokenize, process it and return
        the list of tokens.

        Args:
            file_path (str): path to document file

        Returns:
            list(str): list of processed tokens
    """
    data = open(file_path,  "r", encoding = 'utf-8').read()
    toks = tokenize_text(data)
    toks = process_tokens(toks)
    return toks

def gov_list_docs(docs_path):
    """List the documents in the gov directory.
        Makes explicit use of the gov directory structure and is not
        a general solution for finding documents.

        Args:
            docs_path (str): path to the gov directory

        Returns:
            list(str): list of paths to the document 
    """
    path_list = []
    #* get all directories in gov root folder
    dirs = glob.glob(os.path.join(docs_path, "*"))
    for d in dirs:
        #* get all the files in each of the sub directories
        files = glob.glob(os.path.join(d, "*"))
        path_list.extend(files)
    return path_list

def make_doc_ids(path_list):
    """Assign unique doc_ids to documents.

        Args:
            path_list (list(str)): list of document paths 

        Returns:
            dict(str : int): dictionary of document paths to document ids
    """
    cur_docid = 0
    doc_ids = {}
    for p in path_list:
        # assign docid
        doc_ids[p] = cur_docid
        # increase docid
        cur_docid += 1
    return doc_ids

def get_token_list(path_list, doc_ids):
    """Read all the documents and get a list of all the tokens

        Args:
            path_list (list(str)): list of paths
            doc_ids (dict(str : int)): dictionary mapping a path to a doc_id

        Returns:
            list(tuple(str, int)): an asc sorted list of token, doc_id tuples
    """
    all_toks = []
    for path in path_list:
        doc_id = doc_ids[path]
        toks = read_doc(path)
        for tok in toks:
            all_toks.append((tok, doc_id))
    return sorted(all_toks)

def index_from_tokens(all_toks):
    """Construct an index from the sorted list of token, doc_id tuples.

        Args:
            all_toks (list(tuple(str, int))): an asc sorted list of (token, doc_id) tuples
                this is sorted first by token, then by doc_id

        Returns:
            tuple(dict(str: list(tuple(int, int))), dict(str : int)): a dictionary that maps tokens to
            list of doc_id, term frequency tuples. Also a dictionary that maps tokens to document 
            frequency.
    """
    index     = {}          # inverted index
    doc_freq  = {}          # document frequence
    curr_pair = all_toks[0] # current investigated token
    term_F    = 0           # term frequence
    
    for i in range(len(all_toks)):
        tokId_pair       = all_toks[i] # token-id pair in the list
        tok_name, tok_id = curr_pair   
        #! meet the exactly same term in the same document so increase tf
        if tokId_pair == curr_pair:
            term_F = term_F + 1
        #! meet another token-id pair so process the previous token-id pair and add it to index and doc-freq
        else:
            index[tok_name]    = [(tok_id, term_F)] if tok_name not in index else index[tok_name] + [(tok_id, term_F)]
            doc_freq[tok_name] = len(index[tok_name])
            term_F             = 1                    # reflesh tf
            curr_pair          = tokId_pair           # reflesh current investigated token
            tok_name, tok_id   = curr_pair
        # deal with the last element in the list    
        if i == len(all_toks) - 1:
            index[tok_name]    = [(tok_id, term_F)] if tok_name not in index else index[tok_name] + [(tok_id, term_F)]
            doc_freq[tok_name] = len(index[tok_name])
        
    return index, doc_freq

# run the index example given in the assignment text
print(index_from_tokens([("cat", 1), ("cat", 1), ("cat", 2), ("door", 1), ("water", 3)]))

# get a list of documents 
doc_list = gov_list_docs("./gov/documents")
print("Found %d documents." % len(doc_list))
num_docs = len(doc_list)

# assign unique doc_ids to each of the documents
doc_ids = make_doc_ids(doc_list)
ids_to_doc = {v:k for k, v in doc_ids.items()}
print("1")
# get the list of tokens in all the documents
tok_list = get_token_list(doc_list, doc_ids)
print("2")
# build the index from the list of tokens
index, doc_freq = index_from_tokens(tok_list)
del tok_list # free some memory
print("3")
# store the index to disk
pickle.dump((index, doc_freq, doc_ids, num_docs), open("stored_index.pik", "wb"))
print("4")


