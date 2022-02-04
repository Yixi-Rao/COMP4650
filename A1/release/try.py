# import string
# trantab1 = str.maketrans("","", string.punctuation)
# test = "this is string example....wow!!!"
# print(test.translate(trantab1))

# sentence = 'The brown fox is quick and he is jumping over the lazy dog'
# import nltk
# tokens = nltk.word_tokenize(sentence)
# tagged_sent = nltk.pos_tag(tokens)
# print(tagged_sent)

# import string


# "+" in string.punctuation

def intersect_query(doc_list1, doc_list2):
    
    # in your run_boolean_query implementation
    # for full marks this should be the O(n + m) intersection algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks
    res = []
    p2 = 0
    p1 = 0
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
