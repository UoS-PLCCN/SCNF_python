"""eval.py

Implementation for Boolean disjunction / clause evaluation.
"""
import itertools

import numpy as np

from scnfn.consts import NEG_SIGN
from scnfn.scnf.types import CNF, Disjunction
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
    disjunction = [literal for literal in disjunction]

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


def eval_function(function: CNF, literal_order: list[str]) -> np.ndarray:
    """Take a CNF expression, and convert it in to a truth-table.
    """
    def _gen_inputs(n: int) -> list:
        """Get all possible combinations of a boolean state with n variables.

        Args:
            n (int): The cardinality.

        Returns:
            list: A list of all the possible combinations.
        """
        output = list(itertools.product([0, 1], repeat=n))
        return output

    output = np.zeros([2] * len(literal_order))  # Truth table, all possible combinations.
    inputs = _gen_inputs(len(literal_order))  # Get all possible CNF inputs.

    for _input in inputs:
        disjunction_results = []

        for disjunction in function:
            result = False

            for literal in disjunction:
                if literal in ["True", "False"]:
                    result = bool(literal)
                else:
                    # Handle negated literals
                    false_value = False
                    if literal[0] == "~":
                        false_value = True
                        literal = literal[1:]

                    value = _input[literal_order.index(literal)]  # Literal value from input
                    result = value != false_value

                    if result:
                        break

            disjunction_results.append(result)

        total_result = True

        for result in disjunction_results:
            if not result:
                total_result = False
                break

        output[_input] = int(total_result)
    return output
