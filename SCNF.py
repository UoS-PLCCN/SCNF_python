"""SCNF.py
Package implementing the algorithms outlined in ``Tractable Learning and Inference for Large-Scale Probabilistic Boolean Networks'' by Ifigeneia Apostolopoulou and Diana Marculescu.
"""

import copy
import operator
import numpy as np
from sympy import *

NEG_SIGN = '~'

def SCNF_Learn(transitions, literals):
    """Compute the SCNF clause for node i.

    args:
        transitions [tuple]: The list of transitions of a node i
        literals [string]: The set of the available literals.
    returns:
        ???: The SCNF clause for node i
    """
    S0, S1, SC = _split_transitions(transitions) #7
    if len(S0) == 0: #8
        Phi = ['True'] #9
    elif len(S1) + len(SC) == 0: #10
        Phi = ['False'] #11
    else: #12
        Phi = CNF_Logic_Learn(S0, S1 + SC, literals) #13
        p = 1 #14
    print(Phi)
    if not len(SC) == 0: #15
        raise Exception('')
    else:#21
        Theta = ['True']#22
    raise Exception('')

def eval_disjunction(state, disjunction, literal_positions):
    disjunction = copy.deepcopy(disjunction)
    #print("eval_disjunction")
    #print(state)
    #print(disjunction)
    #print(literal_positions)
    value_mask = np.ones(len(disjunction), dtype=bool)
    for i in range(len(disjunction)):
        literal = disjunction[i]
        if literal[0] == '~':
            value_mask[i] = False
            disjunction[i] = literal[1:]

    output = False
    i = 0
    for i in range(len(disjunction)):
        literal = disjunction[i]
        position = literal_positions.index(literal)
        if state[position] == value_mask[i]:
            output = True
    return output

def CNF_Logic_Learn(H0, H1, L):
    """Return the CNF given the positive and negative clauses

    args:
        H0 [list]: The set of negative transitions
        H1 [list]: The set of positive transitions
        L [string]: The set of available literals
    returns:
        Phi: A CNF clause, which returns 0 for all states in H0, and 1 for all states in H1
    """
    H0 = copy.deepcopy(H0)
    L = copy.deepcopy(L)
    k = 0 #8
    Phi = []
    while not len(H0) == 0:#9
#        print(f"|H0| {len(H0)}")
        phi = CNF_Disjunction_Learn(H0, H1, L, [], L[:int(len(L)/2)], debug = False) #10
#        print(f"output: {phi}")
#        raise Exception('')
        used_literals = [x for x in phi]
#        print(f"L: {L}")
        H0_new = [] #11
        for h in H0:
            #print(h)
            output_after_disjunction = eval_disjunction(h, phi,L[:int(len(L)/2)])
            #print(output_after_disjunction)
            #input()
            if output_after_disjunction:
                H0_new += [h]
#        print(H0_new)
#        print("|H0_new|: {0}".format(len(H0_new)))
        H0 = copy.deepcopy(H0_new)
#        print(H0)
#        print(f"|H0|: {len(H0)}")
#        input("Appending disjunction")
        Phi = Phi + [phi] # 12
        k += 1 # 13
    return Phi #15
def CNF_Disjunction_Learn(H0, H1, L, phi, literal_positions_global, debug = False):
    if debug:
        print("CNF_Disjunction_Learn")
        print(f"phi: {phi}")
        print(f"L: {L}")
        print(f"|H0|: {len(H0)}")
        print(f"|H1|: {len(H1)}")
        input()
    if len(L) == 0: #9
        return [] #10
    score = {}
    for l in L: # 12
        if debug:
            print(f"Evaluating {l}")
        literal_index = L.index(l)
        negation = False
        if l[0] == NEG_SIGN:
            literal_index -= int((len(L)/2))
            negation = True
        if len(H0) == 0: #13
            s = len([x for x in H1 if eval_disjunction(x, phi + [l], literal_positions_global)])
            score[l] = s/len(H1) #14
            if debug:
                print(f"|H0| is empty.")
                print(f"|Satisfied H1|: {s}")
                print(f"score: {score[l]}")
        elif len(H1) == 0: #15
            s = len([x for x in H0 if not eval_disjunction(x, phi + [l], literal_positions_global)])
            score[l] = s/len(H0) #16
            if debug:
                print(f"|H1| is empty.")
                print(f"|Satisfied H0|: {s}")
                print(f"score: {score[l]}")
        else: #17
            sp = len([x for x in H1 if eval_disjunction(x, phi + [l], literal_positions_global)])
            sn = len([x for x in H0 if eval_disjunction(x, phi + [l], literal_positions_global)])
            score[l] = sp/len(H1) - sn/len(H0) #20
            if debug:
                print(f"|H1| and |H0| non-empty")
                print(f"|desirable transitions|: {sp}")
                print(f"|undesirable transitions|: {sn}")
                print(f"score: {score[l]}")
        if debug:
            input()
    #L22 in alg
    best_literal = max(score.items(), key = operator.itemgetter(1))[0] # 23
    if debug:
        print(f"score(l): {score}")
        print(f"l*: {best_literal}")
    literal_index = L.index(best_literal)
    negation = False
    if best_literal[0] == NEG_SIGN:
        literal_index -= int((len(L)/2))
        negation = True
    fulfilled_H1 = [x for x in H1 if eval_disjunction(x, phi + [l], literal_positions_global)]
    fulfilled_H0 = [x for x in H0 if eval_disjunction(x, phi + [l], literal_positions_global)]
    H1_remaining = []
    H0_remaining = []
    for h in H1: #24
        if not h in fulfilled_H1:
            H1_remaining += [h]
    for h in H0: #25
        if not h in fulfilled_H0:
            H0_remaining += [h]
    if debug:
        print(f"|H0^|: {len(H0_remaining)}")
        print(f"|H1^|: {len(H1_remaining)}")
        input()
    #L26 in alg
    if H1_remaining == H1 or len(H0_remaining) == 0: # 26
        if debug:
            print("Leads to invalid disjunction")
        lit_remaining = copy.deepcopy(L)
        lit_remaining.remove(best_literal)
        return CNF_Disjunction_Learn(H0, H1, lit_remaining, phi, literal_positions_global,debug = debug) #27

    if len(H1_remaining) == 0: # 29
        if debug:
            print("All H1s satisfied")
        phi_new = phi +[best_literal] #30
        return phi_new #31

    lit_remaining = copy.deepcopy(L)
    lit_remaining.remove(best_literal)
    if negation:
        if best_literal[1:] in lit_remaining:
            lit_remaining.remove(best_literal[1:])
    else:
        if NEG_SIGN + best_literal in lit_remaining:
            lit_remaining.remove(NEG_SIGN + best_literal)
    if debug:
        print("Going deeper")
    phi_new = CNF_Disjunction_Learn(H0_remaining, H1_remaining, lit_remaining, phi + [best_literal],literal_positions_global, debug = debug) #33
    if len(phi_new) == 0: #34
        lit_remaining = copy.deepcopy(L)
        lit_remaining.remove(best_literal)
        if debug:
            print("Ran out of literals to add?")
        return CNF_Disjunction_Learn(H0, H1, lit_remaining, phi,literal_positions_global, debug = debug) #5

    return phi_new #37
def _split_transitions(transitions):
    """Split the transitions according to equations 20, 21 and 22 in the paper.

    args:
        transitions [tuple]: the list of transitions

    returns:
        S0: All states that only result the target node in being 0
        S1: All states that only result the target node in being 1
        SC: All states that result in target being in either 0 or 1
    """
    S0 = []
    S1 = []
    SC = []
    ones = []
    zeros = []
    for s_i, val in transitions:
        if val:
            ones += [s_i]
        else:
            zeros += [s_i]
    for state_1 in ones:
        if state_1 in zeros and not state_1 in SC:
            SC += [state_1]
    for both_entry in SC:
        ones.remove(both_entry)
        zeros.remove(both_entry)
    S0 = zeros
    S1 = ones
    return S0, S1, SC
