"""utils.py

Utilities for SCNFN.
"""
# Type hints
State = list[bool]
Transition = tuple[State, State]
TrimmedTransition = tuple[State, bool]


def trim_genedata(transitions: list[Transition], selected_variables: list[int]) -> list[Transition]:
    """Given a set of transitions and selected variables, remove data which does not belong to the selected variables.

    Args:
        transitions (List[Transition]): A list of transitions. Each transition is a tuple of states.
        selected_variables (list[int]): A list of indices for selected variables.

    Returns:
        list[Transition]: A list of tuples with the corresponding data from `selected_variables`.
    """
    def _trim(_state):
        return [_state[i] for i in selected_variables]

    return [(_trim(state), _trim(next_state)) for state, next_state in transitions]


def trim_transitions(transitions: list[Transition], node: int) -> list[TrimmedTransition]:
    """Trim the next state in a transition to just the value of a single node.

    Args:
        transitions (List[Transition]): A list of transitions. Each transition is a tuple of states.
        node (int): The index of the node whose value in the next state we're interested in.

    Returns:
        list[TrimmedTransition]: A list of trimmed transitions.\
            A trimmed transition is a tuple of a state and the boolean value of the item.
    """
    return [(state, next_state[node]) for state, next_state in transitions]
