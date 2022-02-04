from collections import defaultdict
import pickle
import os
import numpy as np

from string_processing import *

def intersect_query(doc_list1, doc_list2):
    '''intersect two lists

        Args:
            doc_list1 (list(int)): document list
            doc_list2 (list(int)): document list

        Returns:
            (list(int)): result
    '''
    res = []
    p2 = 0 # index of doc_list2
    p1 = 0 # index of doc_list1
    while p1 != len(doc_list1) and p2 != len(doc_list2):
        if doc_list1[p1] == doc_list2[p2]:
            res.append(doc_list1[p1])
            p1 += 1
            p2 += 1
        elif doc_list1[p1] < doc_list2[p2]:
            p1 += 1
        else:
            p2 += 1
    
    return res

def union_query(doc_list1, doc_list2):
    '''union two lists

        Args:
            doc_list1 (list(int)): document list
            doc_list2 (list(int)): document list

        Returns:
            (list(int)): result
    '''
    res = []
    p2 = 0
    p1 = 0
    while p1 != len(doc_list1) and p2 != len(doc_list2):
        if doc_list1[p1] == doc_list2[p2]:
            res.append(doc_list1[p1])
            p1 += 1
            p2 += 1
        elif doc_list1[p1] < doc_list2[p2]:
            res.append(doc_list1[p1])
            p1 += 1
        else:
            res.append(doc_list2[p2])
            p2 += 1
            
    if p1 == len(doc_list1):
        res.extend(doc_list2[p2:len(doc_list2)])
    else:
        res.extend(doc_list1[p1:len(doc_list1)])
            
    return res

def run_boolean_query(query, index):
    """Runs a boolean query using the index.

        Args:
            query (str): boolean query string
            index (dict(str : list(tuple(int, int)))): The index aka dictonary of posting lists

        Returns:
            list(int): a list of doc_ids which are relevant to the query
    """
    if query == "":
        return []
    
    query_tokens = query.split(" ")                         # cut the query based on space
    relevant_docs = [x for (x,_) in index[query_tokens[0]]] # default is the first token result

    for i in range(1, len(query_tokens), 2):
        # first check the operation and then apply different function with second token
        if query_tokens[i] == "OR":
            relevant_docs = union_query(relevant_docs, [x for (x,_) in index[query_tokens[i + 1]]])
        else:
            relevant_docs = intersect_query(relevant_docs, [x for (x,_) in index[query_tokens[i + 1]]])
    
    return relevant_docs


# load the stored index
(index, doc_freq, doc_ids, num_docs) = pickle.load(open("stored_index.pik", "rb"))

print("Index length:", len(index))
if len(index) != 906290:
    print("Warning: the length of the index looks wrong.")
    print("Make sure you are using `process_tokens_original` when you build the index.")
    raise Exception()

# the list of queries asked for in the assignment text
queries = [
    "Welcoming",
    "unwelcome OR sam",
    "ducks AND water",
    "plan AND water AND wage",
    "plan OR record AND water AND wage",
    "space AND engine OR football AND placement"
]

# run each of the queries and print the result
ids_to_doc = {v:k for k, v in doc_ids.items()}
for q in queries:
    res = run_boolean_query(q, index)
    res.sort(key=lambda x: ids_to_doc[x])
    print(q)
    for r in res:
        print(ids_to_doc[r])





