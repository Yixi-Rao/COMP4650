from collections import defaultdict

import numpy as np


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
    index     = {}
    doc_freq  = {}
    curr_pair = all_toks[0]
    term_F    = 0
    
    for i in range(len(all_toks)):
        tokId_pair       = all_toks[i]
        tok_name, tok_id = curr_pair
        
        if tokId_pair == curr_pair:
            term_F = term_F + 1
        else:
            index[tok_name]    = [(tok_id, term_F)] if tok_name not in index else index[tok_name] + [(tok_id, term_F)]
            doc_freq[tok_name] = len(index[tok_name])
            term_F             = 1
            curr_pair          = tokId_pair
            tok_name, tok_id   = curr_pair
            
        if i == len(all_toks) - 1:
            index[tok_name]    = [(tok_id, term_F)] if tok_name not in index else index[tok_name] + [(tok_id, term_F)]
            doc_freq[tok_name] = len(index[tok_name])
        
    return index, doc_freq

def get_doc_to_norm(index, doc_freq, num_docs):
    """Precompute the norms for each document vector in the corpus.

        Args:
            index (dict(str : list(tuple(int, int)))): The index aka dictonary of posting lists
            doc_freq (dict(str : int)): document frequency for each term
            num_docs (int): number of documents in the corpus

        Returns:
            dict(int: float): a dictionary mapping doc_ids to document norms
    """
    # TODO: Edit this function to implement tfidf

    doc_norm = defaultdict(int)
    # calculate square of norm for all docs
    for term in index.keys():
        for doc_id, doc_tf in index[term]:
            doc_norm[doc_id] += doc_tf **2

    # take square root squared norms
    for doc_id in doc_norm.keys():
        doc_norm[doc_id] = np.sqrt(doc_norm[doc_id])

    return doc_norm

#
    # def index_from_tokens(all_toks):
    #     """Construct an index from the sorted list of token, doc_id tuples.

    #         Args:
    #             all_toks (list(tuple(str, int))): an asc sorted list of (token, doc_id) tuples
    #                 this is sorted first by token, then by doc_id

    #         Returns:
    #             tuple(dict(str: list(tuple(int, int))), dict(str : int)): a dictionary that maps tokens to
    #             list of doc_id, term frequency tuples. Also a dictionary that maps tokens to document 
    #             frequency.
    #     """
    #     index     = {}
    #     doc_freq  = {}
    #     curr_pair = all_toks[0]
    #     term_F    = 0
        
    #     for i in range(len(all_toks)):
    #         tokId_pair = all_toks[i]
    #         tok_name, tok_id = curr_pair
    #         if tokId_pair == curr_pair:
    #             term_F = term_F + 1
    #             if i == len(all_toks) - 1:
    #                 if tok_name not in index:
    #                     index[tok_name] = [(tok_id, term_F)]
    #                     doc_freq[tok_name] = 1
    #                 else:
    #                     index[tok_name].append((tok_id, term_F))
    #                     doc_freq[tok_name] = len(index[tok_name])
    #         else:
    #             if tok_name not in index:
    #                 index[tok_name] = [(tok_id, term_F)]
    #                 doc_freq[tok_name] = 1
    #             else:
    #                 index[tok_name].append((tok_id, term_F))
    #                 doc_freq[tok_name] = len(index[tok_name])
    #             curr_pair = tokId_pair
    #             tok_name, tok_id = curr_pair
    #             if i == len(all_toks) - 1:
    #                 if tok_name not in index:
    #                     index[tok_name] = [(tok_id, term_F)]
    #                     doc_freq[tok_name] = 1
    #                 else:
    #                     index[tok_name].append((tok_id, term_F))
    #                     doc_freq[tok_name] = len(index[tok_name])
            
    #     return index, doc_freq
#
print(index_from_tokens([("cat", 1),  ("cat", 1),  ("cat", 2),
                         ("door", 1), ("door", 1), ("door", 2),  ("door", 3),("door", 3), ("dopr", 1),
                         ("water", 1),("water", 2),("water", 2), ("water", 3),
                         ("xoor", 1)]))

print(index_from_tokens([("cat", 1), ("cat", 1), ("cat", 2), ("door", 1), ("water", 3)]))
({'cat'  : [(1, 2), (2, 1)],
  'door' : [(1, 2), (2, 1), (3, 2)],
  'dopr' : [(1, 1)],
  'water': [(1, 1), (2, 2), (3, 1)],
  'xoor' : [(1, 1)]},
 
 {'cat': 2, 'door': 3, 'dopr': 1, 'water': 3, 'xoor': 1})

# print(get_doc_to_norm({'cat': [(1, 2), (2, 1)], 'door': [(1, 2), (2, 1), (3, 1)], 'water': [(1, 1), (3, 1)]}, 1,1))
# print(index_from_tokens([("cat", 1), ("cat", 1), ("cat", 2)]))
# print(index_from_tokens([("cat", 1), ("cat", 1)]))
# print(index_from_tokens([("cat", 1), ("cat", 2)]))
# print(index_from_tokens([("cat", 1)]))

# print(index_from_tokens([("a", 1), ("b", 1), ("c", 2), ("d", 1), ("e", 3)]))
