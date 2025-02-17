"""main.py

Example file for the use of the SCNF module.
The genedata is from Schiebinger et. al.
Hence the biological names.
"""
import pickle
import sys
from pathlib import Path
from time import time

import gym
import numpy as np

from scnfn.pbn import SCNF_To_PBN
from scnfn.scnf.learn import SCNF_Learn
from scnfn.utils import trim_genedata, trim_transitions

# Import genedata
genedata = pickle.load(
    open("data/schiebinger_data_full.pkl", "rb")
)  # Pre-processed genedata
# Map from indexes in the original dataset to the names. Positions match.
namemap = pickle.load(open("data/schiebinger_namemap_full.pkl", "rb"))

N = 750  # Size of the network to infer

rerun = False  # Re-run from pre-selected genes

# Select genes
if rerun:
    genes = pickle.load(open("data/rerun_genes.tst.pkl", "rb"))
else:
    genes = np.floor(np.random.rand(N) * len(namemap)).astype(
        int
    )  # Randomly select genes.
    pickle.dump(
        genes, open("data/rerun_genes.tst.pkl", "wb")
    )  # Save genes to allow re-running withi this selection.

# Process literals
genedata = trim_genedata(genedata, genes)  # Trim the dataset.

literals = [
    list(namemap.values())[x] for x in genes
]  # Computing the list for literals using gene names.
neg_literals = [
    "~" + literal for literal in literals
]  # Computing the list for negated literals.

literals += neg_literals
literal_order = literals[: len(literals) // 2]

# Learn SCNFN
SCNFs = []
sys.setrecursionlimit(20_000)

start = time()
for i, gene in enumerate(genes):
    print(f"Computing PBN: {i+1}/{N} - Time Elapsed: {time() - start:2f}s", end="\r")
    relevant_transitions = trim_transitions(genedata, i)  # Transitions are trimmed
    clause = SCNF_Learn(relevant_transitions, literals)
    SCNFs += [clause]
print(end="\n")

# Convert to PBN
PBN = SCNF_To_PBN(SCNFs, literal_order)
function, mask = PBN[0]
env = gym.make("gym_PBN:PBN", PBN_data=PBN)

# Save to file
model_path = Path("models") / f"SCNFN_PBN_{N}.tst.pkl"
if not model_path.parent.exists():
    model_path.parent.mkdir()

with open(model_path, "wb") as f:
    pickle.dump(env, f)

# Test
env.reset()
for _ in range(10):
    print(env.render(mode="float"))
    env.step()
