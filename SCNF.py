"""SCNF.py
Package implementing the algorithms outlined in ``Tractable Learning and Inference for Large-Scale Probabilistic Boolean Networks'' by Ifigeneia Apostolopoulou and Diana Marculescu.
"""

import copy
import operator
import numpy as np
from sympy import *
import itertools
import math
import scipy.optimize as optimize

NEG_SIGN = '~'

class Psi_map():
    def __init__(self):
        self.state_indexes = []
        self.disjunctions = []

    def get(self, state):
        relevant_index = self.state_indexes.index(state)
        relevant_disjunction = self.disjunctions[relevant_index]
        return [relevant_disjunction]

    def assign_probabilities(self, Theta, p):
        for i in range(len(Theta)):
            relevant_theta = Theta[i]
            relevant_p = p[i]
            for j in range(len(self.disjunctions)):
                disjunction, prob_old = self.disjunctions[j]
                if disjunction == relevant_theta:
                    self.disjunctions[j] = disjunction, relevant_p

    def assign(self, state, disjunction):
        if not state in self.state_indexes:
            self.state_indexes += [state]
        state_index = self.state_indexes.index(state)
        if len(self.disjunctions)-1 < state_index:
            self.disjunctions += [(disjunction[0], None)]
        else:
            old_disjunctions = self.disjunctions[state_index]
            old_disjunctions = old_disjunctions + [(disjunction[0], None)]
            self.disjunctions[state_index] = old_disjunctions
    def __str__(self):
        output = "Psi:\n"
        for i in range(len(self.state_indexes)):
            suboutput = "{0}: {1} \n".format(self.state_indexes[i], self.disjunctions[i])
            output += suboutput
        return output

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
        Phi = [('True', 1)] #9
    elif len(S1) + len(SC) == 0: #10
        Phi = [('False', 1)] #11
    else: #12
#        print("|S0|: {0}".format(len(S0)))
#        print("|S1+SC|: {0}".format(len(S1+SC)))
        Phi = CNF_Logic_Learn(S0, S1 + SC, literals) #13
        p = 1 #14
        Phi_new = []
        for phi in Phi:
            Phi_new += [(phi, 1)]
        Phi = Phi_new
#    print("Phi: {0}".format(Phi))
    if not len(SC) == 0: #15
#        print("|SC|: {0}".format(len(SC)))
#        print("|S1|: {0}".format(len(S1)))
        Theta = CNF_Logic_Learn(SC, S1, literals, debug = False)
        #print("Theta: {0}".format(Theta))
        Theta = CNF_Parameter_Learn(literals, SC, Theta, transitions)
#        print(Theta)
    else:#21
        Theta = [(['True'], 1)]#22
#        print("Theta: {0}".format(Theta))
    return Theta + Phi

def CNF_Parameter_Learn(literals, SC, Theta, full_transitions):
    literal_position = literals[:int(len(literals)/2)]
    Psi = Psi_map()
#    print(Theta)
    for h in SC:
        h_disjunctions = []
        for disjunction in Theta:
            if not eval_disjunction(h, disjunction, literal_position):
                h_disjunctions += [disjunction]
        Psi.assign(h, h_disjunctions)
    p = np.random.rand(len(Theta))
    l = 0.1
    loss = lambda x: compute_loss(x, Theta, Psi, SC, l, full_transitions, literal_position)
    bnds = []
    for _ in range(len(Theta)):
        bnds += [(10**-8, 1 - 10 ** -8)]
    bnds = tuple(bnds)
    min_loss = optimize.minimize(loss, p, bounds=bnds).x
    #print("Initial guess: {0}".format(p))
    #print("Optimized: {0}".format(min_loss))
    Theta_New = []
    for i in range(len(Theta)):
        Theta_New += [(Theta[i], min_loss[i])]
    Theta = Theta_New
    return Theta

def compute_loss(p, Theta, Psi, transitions, l, full_transitions, literal_position):
#    print("compute_loss")
    Theta_New = []
    for i in range(len(Theta)):
        Theta_New += [(Theta[i], p[i])]
    Psi.assign_probabilities(Theta, p)
    Theta = Theta_New
#    print(Theta)   
#    print(Psi)
    output = 0
    for t in transitions:
        P0 = compute_P0(t, Theta, Psi)
        output -= compute_log_likelihood(t, full_transitions, P0)
    #epsilons = 0
#    for t in transitions:
#        epsilons += compute_epsilon(P0)

#    epsilons *= l
#    output += epsilons
#    print(output)
    return output
#def compute_epsilon(P0):

def compute_P0(previous_state, Theta, Psi):
    output = 0
#    print("Computing P0")
#    print(previous_state)
#    print(Theta)
#    print(Psi)
    relevant_disjunctions = Psi.get(previous_state)
#    print(relevant_disjunctions)
    for M in range(len(relevant_disjunctions)):
#        print(M)
        m = M+1
        disjunction_subsets = list(itertools.combinations(relevant_disjunctions,m))
#        print(disjunction_subsets)
        probability_sum = 0
        for disjunction_subset in disjunction_subsets:
#            print(disjunction_subset)
            p = 1
            for _, probability in disjunction_subset:
                p *= probability
            probability_sum += probability
#        print(probability_sum)
        output += ((-1)**(m+1)) * probability_sum
    return output
def compute_log_likelihood(transition, full_transitions, P0):
    N0 = 0
    N1 = 0
    for state, value in full_transitions:
        if state == transition:
            if value:
                N1 += 1
            else:
                N0 += 1
    logloss = N0 * math.log(P0) + N1 * math.log(1-P0)
    return logloss

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
    for literal in disjunction:
        if literal == 'True':
            return True
    for i in range(len(disjunction)):
        literal = disjunction[i]
        position = literal_positions.index(literal)
        if state[position] == value_mask[i]:
            output = True
    return output

def CNF_Logic_Learn(H0, H1, L, debug = False):
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
        phi = CNF_Disjunction_Learn(H0, H1, L, [], L[:int(len(L)/2)], debug = debug) #10
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
        print(f"H0: {H0}")
        print(f"H1: {H1}")
        input()
    if len(L) == 0: #9
        return []
#        return ['True'] #10
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
        #if debug:
        #    input()
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
    fulfilled_H1 = [x for x in H1 if eval_disjunction(x, phi + [best_literal], literal_positions_global)]
    fulfilled_H0 = [x for x in H0 if eval_disjunction(x, phi + [best_literal], literal_positions_global)]
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
        phi_new = phi +[best_literal] #30
        if debug:
            print("All H1s satisfied")
            print("returning {0}".format(phi_new))
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
        while both_entry in ones:
            ones.remove(both_entry)
        while both_entry in zeros:
            zeros.remove(both_entry)
    S0 = zeros
    S1 = ones
    return S0, S1, SC

def SCNF_To_PBN(SCNF, literal_order):
    PBN = []
    for target_SCNF in SCNF:
        Phi = []
        Theta = []
        for disjunction, probability in target_SCNF:
            if not probability == 0 and not probability == 1:
                Phi += [(disjunction, probability)]
            else:
                Theta += [(disjunction, probability)]
        literals_used = []
        for disjunction, _ in Phi:
            for literal in disjunction:
                if literal[0] == '~':
                    literal = literal[1:]
                if not literal in literals_used:
                    literals_used += [literal]
        for disjunction, _ in Theta:
            for literal in disjunction:
                if not literal == 'True':
                    if literal[0] == '~':
                        literal = literal[1:]
                    if not literal in literals_used:
                        literals_used += [literal]
        literals_used.sort(key = lambda x: literal_order.index(x))
        shape = [2]*len(literals_used)
        function_vector = np.zeros(shape)

        Phi_powerset = list(itertools.chain.from_iterable(itertools.combinations(Phi, r) for r in range(len(Phi)+1)))
        f = Theta
        n_functions = len(Phi_powerset)

        for subpowerset in Phi_powerset:
            probability = 1
            for clause in Phi:
                _, prob = clause
                if clause in subpowerset:
                    probability *= prob
                else:
                    probability *= (1-prob)
            logical_function = []
            for func, _ in Theta:
                logical_function += [func]
            for func, _ in subpowerset:
                logical_function += [func]
            vector = evaluate_function(logical_function, literals_used) * probability
            function_vector += vector
        input_mask = np.zeros(len(literal_order), dtype=bool)
        for l in literals_used:
            pos = literal_order.index(l)
            input_mask[pos] = True
        PBN += [(input_mask, function_vector)]
    return PBN

def evaluate_function(function, literal_order):
    print(function)
    shape = [2]*len(literal_order)
    output = np.zeros(shape)
    inputs = gen_inputs(len(literal_order))
    for inp in inputs:
        disjunction_results = []
        for disjunction in function:
            conjunction_result = False
            for literal in disjunction:
                if literal == 'True':
                    conjunction_result = True
                else:
                    negation = False
                    if literal[0] == '~':
                        negation = True
                        literal = literal[1:]
                    pos = literal_order.index(literal)
                    value = inp[pos]
                    if not value == negation:
                        conjunction_result = True
            disjunction_results += [conjunction_result]
        total_result = True
        for r in disjunction_results:
            if not r:
                total_result = False
        output[inp] = int(total_result)
    return output

def gen_inputs(n):
    output = list(itertools.product(range(2), repeat = n))
    return output
