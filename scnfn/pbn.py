"""pbn.py

Utilities to convert to and from PBN.
"""
from itertools import chain, combinations
from math import prod

import numpy as np

from scnfn.scnf.eval import eval_function
from scnfn.scnf.types import SCNF
from scnfn.types import PBN


def SCNF_To_PBN(SCNFN: SCNF, literal_order: list[str]) -> PBN:
    """Convert SCNFN to PBN according to Proposition I on pages 2272-2273 of the paper.

    Args:
        SCNFN (SCNF): The list SCNF clauses making an SCNFN.
        literal_order (list[str]): The order of the literals in the SCNFN state.

    Returns:
        PBN: Data defining a PBN.
    """
    _PBN = []  # PBN data buffer

    for target_SCNF in SCNFN:  # Iterate through the SCNF clauses of the SCNFN
        # Decompose SCNF clause into Phi (deterministic disjunctions) and Theta (stochastic disjunctions).
        Phi = []
        Theta = []
        for disjunction, probability in target_SCNF:
            if probability == 0 or probability == 1:
                Phi += [(disjunction, probability)]
            else:
                Theta += [(disjunction, probability)]

        # Enumerate literals in use.
        literals_used = []

        def _add_literal(literal: str):
            _literal = literal
            if literal[0] == '~':
                _literal = literal[1:]
            if literal not in literals_used and literal not in ["True", "False"]:
                literals_used.append(_literal)

        for disjunction, _ in Phi:
            for literal in disjunction:
                _add_literal(literal)

        for disjunction, _ in Theta:
            for literal in disjunction:
                _add_literal(literal)

        # Sort literals within the list.
        literals_used.sort(key=lambda x: literal_order.index(x))
        function_vector = np.zeros([2] * len(literals_used))  # f^(i)

        # Powerset of all possible subsets
        Theta_powerset = [chain.from_iterable(combinations(Theta, r) for r in range(len(Theta) + 1))]

        for beta_j in Theta_powerset:
            # Equation 12 + 13.
            # Equation 13 is basically just filtering to beta_j so it can be simplified
            f_j = [f for f, _ in Phi] + [f for f, _ in beta_j]

            # Equation 14.
            probability = prod([q_z if (x, q_z) in beta_j else (1 - q_z) for x, q_z in Theta])

            # Convert f_j to its truth table.
            vector = eval_function(f_j, literals_used) * probability
            function_vector += vector

        # Set up input mask for the PBN
        input_mask = np.zeros(len(literal_order), dtype=bool)
        for literal in literals_used:
            input_mask[literal_order.index(literal)] = True

        _PBN += [(input_mask, function_vector)]

    return _PBN
