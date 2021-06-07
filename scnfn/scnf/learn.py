"""learn.py

Implementations for SCNF clause learning and its associated algorithms.
"""
import copy
import operator

import numpy as np
import scipy.optimize as optimize

from scnfn.consts import DEBUG, NEG_SIGN
from scnfn.scnf.eval import eval_disjunction
from scnfn.scnf.optim import Psi0, compute_loss
from scnfn.scnf.types import CNF, SCNF, Disjunction
from scnfn.scnf.utils import split_transitions
from scnfn.types import State, TrimmedTransition


def SCNF_Learn(transitions: list[TrimmedTransition], literals: list[str]) -> SCNF:
    """Compute the SCNF clause for node `i`. (Algorithm 1 from the paper.)

    Args:
        transitions (list[TrimmedTransition]): The list of trimmed transitions of a given node `i`.
        literals (list[str]): The set of the available literals (gene names).
    Returns:
        SCNF: The SCNF clause for node `i`, so a list of tuples of disjunctions and their associated probabilities.
    """
    # Split the transitions into sets of states.
    S0, S1, SC = split_transitions(transitions)  # 7

    # Compute the deterministic portion of the SCNF rule.
    if len(S0) == 0:  # 8
        Phi = [(['True'], 1)]  # 9
    elif len(S1.union(SC)) == 0:  # 10
        Phi = [(['False'], 1)]  # 11
    else:  # 12
        Phi = [(phi, 1) for phi in CNF_Logic_Learn(S0, S1 + SC, literals)]  # 13

    # Compute the stochastic portion of the SCNF rule.
    if not len(SC) == 0:  # 15
        Theta = CNF_Logic_Learn(SC, S1, literals)
        Theta = CNF_Parameter_Learn(literals, SC, Theta, transitions)
    else:  # 21
        Theta = [(['True'], 1)]  # 22

    return Theta + Phi  # 24 and 25


def CNF_Logic_Learn(H0: set[State], H1: set[State], L: list[str]) -> CNF:
    """Return the CNF given the positive and negative clauses.\
        (Algorithm 2 from the paper.)

    Args:
        H0 (set[State]): The set of states resulting in a negative node value.
        H1 (set[State]): The set of states resulting in a positive node value.
        L (list[str]): The set of available literals.

    Returns:
        CNF: A CNF clause, which returns 0 for all states in H0, and 1 for all states in H1.
    """
    # Making a copy of the arguments since Python *is* pass by reference.
    H0 = copy.deepcopy(H0)
    L = copy.deepcopy(L)
    L_order = L[:len(L) // 2]  # Order of only the actual literals, not their negations
    Phi = []

    while not len(H0) == 0:  # 9
        phi = CNF_Disjunction_Learn(H0, H1, L, [], L_order)  # 10
        H0 = [h for h in H0 if eval_disjunction(h, phi, L_order) is True]  # 11
        Phi.append(phi)  # 12

    return Phi  # 15


def CNF_Disjunction_Learn(
    H0: set[State], H1: set[State], L: list[str], phi: Disjunction,
    literal_positions_global: list[str]
) -> Disjunction:
    """Learn a disjunction that satisfies two conditions:\
         evaluating to 0 for at lesat one state in the H0 set of states,\
        and evaluating 1 for all states in the H1 set of states.
       Recursive function: each recursive call inserts a new literal in the current disjunction.

    Args:
        H0 (set[State]): The set of states H0, from which at least one state will evaluate the learnt disjunction to 0.
        H1 (set[State]): The set of states H1, from which all states will evaluate the learnt disjunction to 1.
        L (list[str]): The set of available literals.
        phi (CNF): The currently formed disjunction.
        literal_positions_global (list[str]): All the available literals, globally\
            (i.e. not in the current recursive call).

    Returns:
        Disjunction: The learned disjunction.
    """
    if DEBUG:
        print("CNF_Disjunction_Learn")
        print(f"phi: {phi}")
        print(f"L: {L}")
        print(f"|H0|: {len(H0)}")
        print(f"|H1|: {len(H1)}")
        print(f"H0: {H0}")
        print(f"H1: {H1}")
        input()

    # Base case.
    if len(L) == 0:  # 9
        return []

    # Calculate a positive score for each literal, according to the number of positive transitions
    # that will be satisfied if the literal is included and a respective negative score.
    score = {}
    for literal in L:  # 12
        if DEBUG:
            print(f"Scoring {literal}")

        if len(H0) == 0:  # 13
            s = len([x for x in H1 if eval_disjunction(x, phi + [literal], literal_positions_global)])
            score[literal] = s / len(H1)  # 14

            if DEBUG:
                print("|H0| is empty.")
                print(f"|Satisfied H1|: {s}")
                print(f"score: {score[literal]}")

        elif len(H1) == 0:  # 15
            s = len([x for x in H0 if not eval_disjunction(x, phi + [literal], literal_positions_global)])
            score[literal] = s / len(H0)  # 16

            if DEBUG:
                print("|H1| is empty.")
                print(f"|Satisfied H0|: {s}")
                print(f"score: {score[literal]}")

        else:  # 17
            score_positive = len([x for x in H1 if eval_disjunction(x, phi + [literal], literal_positions_global)])
            score_negative = len([x for x in H0 if eval_disjunction(x, phi + [literal], literal_positions_global)])
            score[literal] = score_positive / len(H1) - score_negative / len(H0)  # 20

            if DEBUG:
                print("|H1| and |H0| non-empty")
                print(f"|desirable transitions|: {score_positive}")
                print(f"|undesirable transitions|: {score_negative}")
                print(f"score: {score[literal]}")

    # Select the literal with the maximum score.
    best_literal = max(score.items(), key=operator.itemgetter(1))[0]  # 23, basically just argmax with a dict

    if DEBUG:
        print(f"score(l): {score}")
        print(f"l*: {best_literal}")

    # Check what the remaining states to fulfil constraints for look like after the addition of the "best" literal.
    fulfilled_H1 = set([x for x in H1 if eval_disjunction(x, phi + [best_literal], literal_positions_global)])
    fulfilled_H0 = set([x for x in H0 if eval_disjunction(x, phi + [best_literal], literal_positions_global)])
    H1_remaining = H1 - fulfilled_H1  # 24
    H0_remaining = H0 - fulfilled_H0  # 25

    if DEBUG:
        print(f"|H0^|: {len(H0_remaining)}")
        print(f"|H1^|: {len(H1_remaining)}")
        input()

    # If the "best" literal didn't make any difference to the H1 transitions satisfied
    # then just remove it from the list of available literals and recur further.
    if H1_remaining == H1 or len(H0_remaining) == 0:  # 26
        if DEBUG:
            print("Leads to invalid disjunction")

        lit_remaining = copy.deepcopy(L)
        lit_remaining.remove(best_literal)
        return CNF_Disjunction_Learn(H0, H1, lit_remaining, phi, literal_positions_global)  # 27

    # If no more positive transitions to add literals for remain after the addition of the "best" literal
    # Then just return the disjunction with the new literal.
    if len(H1_remaining) == 0:  # 29
        phi_new = phi + [best_literal]  # 30

        if DEBUG:
            print("All H1s satisfied")
            print("returning {0}".format(phi_new))

        return phi_new  # 31

    # Otherwise, if the addition of "best" literal did make a difference but not all transitions are satisfied
    # then add it *and* recur further.

    # Remove it from the remaining literals. Part of line 33.
    lit_remaining = copy.deepcopy(L)
    lit_remaining.remove(best_literal)

    # Remove its (non-)negated counterpart from the remaining literals too. Part of line 33.
    if best_literal[0] == NEG_SIGN:
        if best_literal[1:] in lit_remaining:
            lit_remaining.remove(best_literal[1:])
    else:
        if NEG_SIGN + best_literal in lit_remaining:
            lit_remaining.remove(NEG_SIGN + best_literal)

    if DEBUG:
        print("Going deeper")

    phi_new = CNF_Disjunction_Learn(
        H0_remaining, H1_remaining, lit_remaining, phi + [best_literal], literal_positions_global
    )  # 33

    # If for whatever reason the new learnt disjunction after adding the best literal and removing it
    # is empty, then try again but without removing the logical negation of the literal.
    if len(phi_new) == 0:  # 34
        lit_remaining = copy.deepcopy(L)
        lit_remaining.remove(best_literal)

        if DEBUG:
            print("Ran out of literals to add?")

        return CNF_Disjunction_Learn(H0, H1, lit_remaining, phi, literal_positions_global)  # 5

    return phi_new  # 37


def CNF_Parameter_Learn(
    literals: list[str], SC: set[State], Theta: CNF, all_transitions: list[TrimmedTransition]
) -> SCNF:
    """Learn the SCNF Bernoulli variable parameters through solving an optimization problem.\
        Implementation of the MLE described in page 2726 of the publication.

    Args:
        literals (list[str]): The available literals.
        SC (set[State]): The set of states that result in the target node's value being either 0 or 1.
        Theta (CNF): The CNF (list of disjunctions) to learn the parameters for.
        all_transitions (list[TrimmedTransition]): All of the transitions for node `i`.

    Returns:
        SCNF: An SCNF clause with probabilities for each disjunction of the underlying CNF\
            learnt to maximize the probability of the original distribution.
    """
    # Get the positions of the literals in the state.
    literal_positions = literals[:len(literals) // 2]

    # Equation 23.
    Psi = Psi0()
    for _lambda in SC:
        Psi0_lambda = [
            disjunction for disjunction in Theta if not eval_disjunction(_lambda, disjunction, literal_positions)
        ]
        Psi[_lambda] = Psi0_lambda

    # Start with random probabilities p.
    p = np.random.rand(len(Theta))

    # Closure over the compute_loss function for a call signature to use with scipy's optimize library.
    def loss(x):
        return compute_loss(x, Theta, Psi, SC, all_transitions)

    # Define the bounds on the probability variables that are optimized
    bounds = []
    for _ in range(len(Theta)):
        bounds.append((10**-8, 1 - 10 ** -8))
    bounds = tuple(bounds)

    # Calculate p to get the minimum loss.
    # Equation 31
    min_loss_p = optimize.minimize(loss, p, bounds=bounds).x

    return list(zip(Theta, min_loss_p))  # Return SCNF
