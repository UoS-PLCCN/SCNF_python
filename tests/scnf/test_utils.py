from scnfn.scnf.utils import split_transitions


def test_split_transitions():
    one_states = [
        (True, True, False),
        (False, True, False),
    ]
    zero_states = [
        (False, True, True),
        (False, False, False),
    ]
    ambiguous_states = [
        (True, True, True),
        (False, False, True)
    ]
    transitions = [
        (one_states[0], True),
        (ambiguous_states[0], True),
        (zero_states[0], False),
        (one_states[1], True),
        (ambiguous_states[0], False),
        (zero_states[1], False),
        (ambiguous_states[1], False),
        (one_states[0], True),
        (ambiguous_states[1], True),
        (zero_states[1], False),
    ]
    S0, S1, SC = split_transitions(transitions)
    assert S0 == set(zero_states)
    assert S1 == set(one_states)
    assert SC == set(ambiguous_states)
