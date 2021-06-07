"""eval.py

Implementation for Boolean disjunction / clause evaluation.
"""
import copy

import numpy as np

from scnfn.consts import NEG_SIGN
from scnfn.scnf.types import Disjunction
from scnfn.types import State


def eval_disjunction(state: State, disjunction: Disjunction, literal_positions: list[str]) -> bool:
    """Evaluate a single disjunction given a state.
    TODO: Would be nice to use this for evaluating entire functions.

    Args:
        state (State): The state.
        disjunction (Disjunction): The disjunction to evaluate.
        literal_positions (list[str]): A list of the literals\
            in the same positions their in the state values can be found in.

    Returns:
        bool: The value of the disjunction.
    """
    # Copy the disjunction since python is pass by reference.
    disjunction = copy.deepcopy(disjunction)

    # Mask to hold what the value of the literal needs to be in the state for it to be considered
    # logically positive (True).
    value_mask = np.ones(len(disjunction), dtype=bool)

    # Clean logical negation symbols from the expression, populate value_mask appropriately.
    for i, literal in enumerate(disjunction):
        if literal[0] == NEG_SIGN:
            value_mask[i] = False
            disjunction[i] = literal[1:]

    for i, literal in enumerate(disjunction):
        # If there's a 'True' literal, just return
        if literal == 'True':
            return True

        # Otherwise, get the value of the literal in the state
        literal_value = state[literal_positions.index(literal)]

        if literal_value == value_mask[i]:
            return True

    return False
