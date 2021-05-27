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
    S0, S1, SC = _split_transitions(transitions)
    if len(S0) == 0:
        Phi = True
    elif len(S1) + len(SC) == 0:
        Phi = False
    else:
        Phi = CNF_Logic_Learn(S0, S1 + SC, literals)
        p = 1
    print(Phi)
    raise Exception('')

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
    k = 0
    Phi = symbols('1')
    while not len(H0) == 0:
        print(f"|H0| {len(H0)}")
        phi = CNF_Disjunction_Learn(H0, H1, L, symbols('False'), debug = False)
        print(f"output: {phi}")
        used_literals = [x for x in phi.atoms(Symbol) if not x == symbols('False')]
        print(f"L: {L}")
        H0_fulfilled = []
        for h in H0:
            print(h)
            phi_new = copy.deepcopy(phi)
            for l in used_literals:
#                print(l)
                l = str(l)
                literal_index = L.index(l)
                negation = False
                if l[0] == NEG_SIGN:
                    literal_index -= int((len(L)/2))
                    negation = True
                literal_value = h[literal_index]
                if negation:
                    literal_value = not literal_value
#                print(literal_index)
#                print(literal_value)
                phi_new = phi_new.subs(l, literal_value)
            print(f"phi_new: {bool(phi_new)}")
            print(type(bool(phi_new)))
            if not bool(phi_new):
                print("Bewoop")
                H0_fulfilled += [h]
            input()
        print(len(H0_fulfilled))
        raise Exception('')
        print(len(H0_fulfilled))
        for h in H0_fulfilled:
            H0.remove(h)
        print(f"|H0|: {len(H0)}")
        input()
        Phi = Phi and (phi)
    print(Phi)
    raise Exception('')
def CNF_Disjunction_Learn(H0, H1, L, phi, debug = False):
    if debug:
        print("CNF_Disjunction_Learn")
        print(f"phi: {phi}")
        print(f"L: {L}")
        print(f"|H0|: {len(H0)}")
        print(f"|H1|: {len(H1)}")
        input()
    if len(L) == 0:
        return symbols('1')
    score = {}
    for l in L:
        if debug:
            print(f"Evaluating {l}")
        literal_index = L.index(l)
        negation = False
        if l[0] == NEG_SIGN:
            literal_index -= int((len(L)/2))
            negation = True
        if len(H0) == 0:
            s = len([x for x in H1 if not x[literal_index] == negation])
            score[l] = s/len(H1)
            if debug:
                print(f"|H0| is empty.")
                print(f"|Satisfied H1|: {s}")
                print(f"score: {score[l]}")
        elif len(H1) == 0:
            s = len([x for x in H0 if x[literal_index] == negation])
            score[l] = s/len(H0)
            if debug:
                print(f"|H1| is empty.")
                print(f"|Satisfied H0|: {s}")
                print(f"score: {score[l]}")
        else:
            sp = len([x for x in H1 if not x[literal_index] == negation])
            sn = len([x for x in H0 if not x[literal_index] == negation])
            score[l] = sp/len(H1) - sn/len(H0)
            if debug:
                print(f"|H1| and |H0| non-empty")
                print(f"|desirable transitions|: {sp}")
                print(f"|undesirable transitions|: {sn}")
                print(f"score: {score[l]}")
        if debug:
            input()
    #L22 in alg
    best_literal = max(score.items(), key = operator.itemgetter(1))[0]
    if debug:
        print(f"score(l): {score}")
        print(f"l*: {best_literal}")
    literal_index = L.index(best_literal)
    negation = False
    if best_literal[0] == NEG_SIGN:
        literal_index -= int((len(L)/2))
        negation = True
    if negation:
        fulfilled_H1 = [x for x in H1 if not x[literal_index]]
        fulfilled_H0 = [x for x in H0 if not x[literal_index]]
    else:
        fulfilled_H1 = [x for x in H1 if x[literal_index]]
        fulfilled_H0 = [x for x in H0 if x[literal_index]]
    H1_remaining = []
    H0_remaining = []
    for h in H1:
        if not h in fulfilled_H1:
            H1_remaining += [h]
    for h in H0:
        if not h in fulfilled_H0:
            H0_remaining += [h]
    if debug:
        print(f"|H0^|: {len(H0_remaining)}")
        print(f"|H1^|: {len(H1_remaining)}")
        input()
    #L26 in alg
    if H1_remaining == H1 or len(H0_remaining) == 0:
        lit_remaining = copy.deepcopy(L)
        lit_remaining.remove(best_literal)
        return CNF_Disjunction_Learn(H0, H1, lit_remaining, phi, debug = debug)

    if len(H1_remaining) == 0:
        phi_new = phi | symbols(best_literal)
        return phi_new

    lit_remaining = copy.deepcopy(L)
    lit_remaining.remove(best_literal)
    if negation:
        if best_literal[1:] in lit_remaining:
            lit_remaining.remove(best_literal[1:])
    else:
        if NEG_SIGN + best_literal in lit_remaining:
            lit_remaining.remove(NEG_SIGN + best_literal)
    phi_new = CNF_Disjunction_Learn(H0_remaining, H1_remaining, lit_remaining, phi | symbols(best_literal), debug = debug)
    if len(phi_new.atoms(Symbol))-1 == 0:
        lit_remaining = copy.deepcopy(L)
        lit_remaining.remove(best_literal)
        return CNF_Disjunction_Learn(H0, H1, lit_remaining, phi, debug = debug)

    return phi_new
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
