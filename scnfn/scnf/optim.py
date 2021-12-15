"""optim.py

Optimization Problem-related functions and utilities to learn the Bernounlli variable parameters of SCNF clauses.
"""
import copy
import itertools
import math
from typing import List, Set, Union

import numpy as np

from scnfn.scnf.types import CNF, SCNF
from scnfn.types import State, TrimmedTransition


class Psi0:
    """A class encapsulating a map to act as the Psi0 function as defined in Equation 23 of the paper.

    The keys of the map are states (tuples) and the values are relevant disjunctions OR SCNF clauses.
    """

    def __init__(self) -> None:
        self.map = {}

    def __getitem__(self, key: State) -> Union[CNF, SCNF]:
        return self.map[key]

    def __setitem__(self, key: State, value: Union[CNF, SCNF]):
        if type(value[0]) != tuple:
            value = [(disjunction, None) for disjunction in value]
        self.map[key] = value

    def assign_probabilities(self, Theta: CNF, p: np.ndarray):
        """Assign probabilities p to disjunctions in the CNF clause Theta.

        Args:
            Theta (CNF): The CNF clause with the disjunctions to associate a probability to.
            p (np.ndarray): The probability to assign.
        """
        for relevant_theta, relevant_p in zip(Theta, p):
            for state, disjunctions in self.map.items():
                for k, (disjunction, _) in enumerate(disjunctions):
                    if disjunction == relevant_theta:
                        self.map[state][k] = disjunction, relevant_p


def compute_loss(
    p: np.ndarray,
    Theta: CNF,
    Psi: Psi0,
    SC: Set[State],
    all_transitions: List[TrimmedTransition],
) -> float:
    """Loss function for the optimization problem to solve in order to set the Bernoulli variable parameters\
        in an SCNF clause.

    Equation 31 from the paper.

    Args:
        p (np.ndarray): the current set of probabilities.
        Theta (CNF): the SCNF disjunctions the probabilities of which to compute loss for.
        Psi (Psi0): The Psi0 map which gives the relevant disjunctions for a certain state.
        SC (Set[State]): The set of states that result in the target node's value being either 0 or 1.
        all_transitions (List[TrimmedTransition]): All of the transitions for node `i`.

    Returns:
        float: The loss based on the logarithmic likelihood of the time series of transitions.
    """
    # Copy of Theta since python is pass by reference and we modify it.
    Theta = copy.deepcopy(Theta)

    # Assign the given probabilities to the CNF disjunctions.
    Psi.assign_probabilities(Theta, p)
    Theta = zip(Theta, p)  # Concat probabilities to the CNF to get an SCNF
    Sigma = 0  # the final sum / output

    # Equation 31
    for _lambda in SC:
        P0_lambda = compute_P0(_lambda, Psi)
        log_likelihood_lambda = compute_log_likelihood(
            _lambda, all_transitions, P0_lambda
        )
        Sigma += log_likelihood_lambda

    return -1 * Sigma


def compute_P0(_lambda: State, Psi: Psi0) -> float:
    """Compute P0 of lambda: the probability that the previous state lambda of the system will\
        switch the node `i` to 0 in the next step.

    Equation 25 from the paper.

    Args:
        previous_state (State): The previous state lambda to calculate the P0 factor for.
        Psi (Psi0): The Psi0 map which gives the relevant disjunctions for a certain state.

    Returns:
        float: The P0 factor.
    """
    Sigma = 0  # the final sum / output
    Psi0_lambda = Psi[_lambda]  # The relevant disjunctions

    for m in range(1, len(Psi0_lambda) + 1):  # from 1, not 0
        # All the disjunction subsets of cardinality m
        Psi0_lambda_m = list(itertools.combinations(Psi0_lambda, m))

        # Sum of the product of probabilities for all disjunction subsets
        probability_sum = sum(
            [
                math.prod([p_jm_n for _, p_jm_n in Theta_jm])
                for Theta_jm in Psi0_lambda_m
            ]
        )

        # Add to final sum
        Sigma += ((-1) ** (m + 1)) * probability_sum

    return Sigma


def compute_log_likelihood(
    _lambda: State, all_transitions: List[TrimmedTransition], P0_lambda: float
) -> float:
    """Compute the logarithmic likelihood of a time series of transitions for a certain state.

    Equations 27, 28 and 32 from the paper.

    Args:
        _lambda (State): The state in question.
        all_transitions (List[TrimmedTransition]): The time series of transitions.
        P0_lambda (float): The P0 factor for the state.

    Returns:
        float: The logarithmic likelihood of a time series of transitions for a certain state.
    """
    N0_lambda = all_transitions.count((_lambda, False))  # Equation 27
    N1_lambda = all_transitions.count((_lambda, True))  # Equation 28

    # Equation 32
    log_likelihood_lambda = N0_lambda * math.log(P0_lambda) + N1_lambda * math.log(
        1 - P0_lambda
    )
    return log_likelihood_lambda
