"""SCNF.py
Package implementing the algorithms outlined in ``Tractable Learning and Inference for Large-Scale Probabilistic Boolean Networks'' by Ifigeneia Apostolopoulou and Diana Marculescu.
"""

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
    print(H0)
    print(H1)
    print(L)
    k = 0
    Phi = None

    raise Exception('')

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
