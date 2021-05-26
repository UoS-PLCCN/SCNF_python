"""utils.py
Utilities for SCNFN
"""

def trim_genedata(data, selected_variables):
    """Given a set of transitions and selected variables, remove data which does not belong to the selected variables.
    args:
        data [tuples]: A list of tuples. Each tuple is a single transition.
        selected_variables [int]: A list of indices for selected variables
    returns:
        [tuples]: a list of tuples with the corresponding data from selected_variables
    """
    output = []
    for s_i, s_j in data:
        t_i = []
        t_j = []
        for var in selected_variables:
            t_i += [s_i[var]]
            t_j += [s_j[var]]
        output += [(t_i, t_j)]
    return output

def trim_outputs(data, position):
    """Return the same dataset where the outputs are a single value at position
    args:
        data [tuples]: A list of tuples. Each tuple is a single transition.
        position (int): the position of the variable to be chosen
    returns:
        [tuples]: a list of tuples with the corresponding data from position
    """
    output = []
    for s_i, s_j in data:
        output += [(s_i, s_j[position])]
    return output
