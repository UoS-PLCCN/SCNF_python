"""utils.py

Utilities for SCNFN.
"""
from typing import List, Tuple

# Type hints
State = List[bool]
Transition = Tuple[State, State]
TrimmedTransition = Tuple[State, bool]


def trim_genedata(
    transitions: List[Transition], selected_variables: List[int]
) -> List[Transition]:
    """Given a set of transitions and selected variables, remove data which does not belong to the selected variables.

    Args:
        transitions (List[Transition]): A list of transitions. Each transition is a tuple of states.
        selected_variables (List[int]): A list of indices for selected variables.

    Returns:
        List[Transition]: A list of tuples with the corresponding data from `selected_variables`.
    """

    def _trim(_state):
        return [_state[i] for i in selected_variables]

    return [(_trim(state), _trim(next_state)) for state, next_state in transitions]


def trim_transitions(
    transitions: List[Transition], node: int
) -> List[TrimmedTransition]:
    """Trim the next state in a transition to just the value of a single node.

    Args:
        transitions (List[Transition]): A list of transitions. Each transition is a tuple of states.
        node (int): The index of the node whose value in the next state we're interested in.

    Returns:
        List[TrimmedTransition]: A list of trimmed transitions.\
            A trimmed transition is a tuple of a state and the boolean value of the item.
    """
    return [(state, next_state[node]) for state, next_state in transitions]
