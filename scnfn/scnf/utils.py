"""utils.py

Helper methods for the learning algorithm.
"""

from scnfn.types import State, TrimmedTransition


def split_transitions(transitions: list[TrimmedTransition]) -> tuple[set[State], set[State], set[State]]:
    """Split the transitions into the S0, S1 and SC sets, according to equations 20, 21 and 22 in the paper.

    Args:
        transitions (list[TrimmedTransition]): A list of trimmed transitions.

    Returns:
        tuple[set[State], set[State], set[State]]:\
            The tuple of three sets of trimmed transitions, in the following order:
            S0: All states that only result in the target node's value being 0
            S1: All states that only result in the target node's value being 1
            SC: All states that result in the target node's value being either 0 or 1
    """
    # Init sets
    S0, S1, SC = set(), set(), set()

    # Split the states in the trimmed transitions based on the target node value is in the next state
    for state, next_state_val in transitions:
        if next_state_val is True:
            S1.add(tuple(state))
        else:
            S0.add(tuple(state))

    # Add any states with ambiguous next target node value to SC
    SC = S1.intersection(S0)

    # Remove any states that are also in SC
    S1 -= SC
    S0 -= SC

    return S0, S1, SC
