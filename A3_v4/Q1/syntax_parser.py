import json
import numpy as np
from collections import defaultdict

class RuleWriter(object):
    """
    This class is for writing rules in a format 
    the judging software can read
    Usage might look like this:

    rule_writer = RuleWriter()
    for lhs, rhs, prob in out_rules:
        rule_writer.add_rule(lhs, rhs, prob)
    rule_writer.write_rules()

    """
    def __init__(self):
        self.rules = []

    def add_rule(self, lhs, rhs, prob):
        """Add a rule to the list of rules
        Does some checking to make sure you are using the correct format.

        Args:
            lhs (str): The left hand side of the rule as a string
            rhs (Iterable(str)): The right hand side of the rule. 
                Accepts an iterable (such as a list or tuple) of strings.
            prob (float): The conditional probability of the rule.
        """
        assert isinstance(lhs, str)
        assert isinstance(rhs, list) or isinstance(rhs, tuple)
        assert not isinstance(rhs, str)
        nrhs = []
        for cl in rhs:
            assert isinstance(cl, str)
            nrhs.append(cl)
        assert isinstance(prob, float)

        self.rules.append((lhs, nrhs, prob))

        
    def write_rules(self, filename="q1.json"):
        """Write the rules to an output file.

        Args:
            filename (str, optional): Where to output the rules. Defaults to "q1.json".
        """
        json.dump(self.rules, open(filename, "w"))

# load the parsed sentences
psents = json.load(open("parsed_sents_list.json", "r"))
# psents = [['A', ['B', ['C', 'blue']], ['B', 'cat']]] # test case

# print a few parsed sentences
# NOTE: you can remove this if you like
# for sent in psents[:10]:
#     print(sent)
#     print()

#! estimate the conditional probabilities of the rules in the grammar
def find_rules(sent: list)-> list:
    '''given a parsed sentense, find all the rules of it.

        Args:
            sent (list): elements of psents

        Returns:
            list: [(α, β)...]
    '''
    answer = []
    # first add the first element to the list, and then recursively go through rest of the list
    if len(sent) == 2 and isinstance(sent[1], str):
        answer.append((sent[0], sent[1]))
        return answer
    else:
        answer.append((sent[0], tuple([β[0] for β in sent[1:]])))
        for α in sent[1:]:
            answer = answer + find_rules(α)
        return answer

# dictionary of this form: dict(str:dict(tuple or str:int)) -> {α : {β : count, ... }, ... }        
Rule_count_dict = {}
for sent in psents:
    rules = find_rules(sent)
    for α, β in rules:
        if Rule_count_dict.get(α) is None:
            Rule_count_dict[α] = {}
            Rule_count_dict[α][β] = 1
        elif Rule_count_dict.get(α).get(β) is None:
            Rule_count_dict[α][β] = 1
        else:
            Rule_count_dict[α][β] += 1    

#! write the rules to the correct output file using the write_rules method
rule_writer = RuleWriter()
for α, βs in Rule_count_dict.items():
    count_α = sum(βs.values())
    for β, count_αβ in βs.items():
        rule_writer.add_rule(α, β if isinstance(β, tuple) else [β], count_αβ / count_α)
rule_writer.write_rules()


