"""main.py

Example file for the use of the SCNF module.
The genedata is from Schiebinger et. al.
Hence the biological names.
"""
import pickle

import numpy as np

import PBN_env
import SCNF
import utils

# Import genedata
genedata = pickle.load(open("schiebinger_data_full.pkl", "rb"))  # Pre-processed genedata
# Map from indexes in the original dataset to the names. Positions match.
namemap = pickle.load(open("schiebinger_namemap_full.pkl", "rb"))

N = 10  # Size of the network to infer

rerun = False  # Re-run from pre-selected genes

# Select genes
if rerun:
    genes = pickle.load(open("rerun_genes.tst.pkl", "rb"))
else:
    genes = np.floor(np.random.rand(N) * len(namemap)).astype(int)  # Randomly select genes.
    pickle.dump(genes, open("rerun_genes.tst.pkl", "wb"))  # Save genes to allow re-running withi this selection.

# Process literals
genedata = utils.trim_genedata(genedata, genes)  # Trim the dataset.

literals = [list(namemap.values())[x] for x in genes]  # Computing the list for literals using gene names.
neg_literals = ['~' + literal for literal in literals]  # Computing the list for negated literals.

literals += neg_literals
literal_order = literals[:len(literals) // 2]

# Learn SCNFN
SCNFs = []

for i, gene in enumerate(genes):
    relevant_transitions = utils.trim_transitions(genedata, i)  # Transitions are trimmed
    clause = SCNF.SCNF_Learn(relevant_transitions, literals)
    SCNFs += [clause]

# Convert to PBN
PBN = SCNF.SCNF_To_PBN(SCNFs, literal_order)
function, mask = PBN[0]
env = PBN_env.PBN(PBN_data=PBN)

# Test
env.reset()
for _ in range(10):
    print(env.get_state().astype(int))
    env.step()
