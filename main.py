"""main.py
Example file for the use of the SCNF module.
The genedata is from Schiebinger et. al.
Hence the biological names.
"""

import pickle
import numpy as np
import sympy
import SCNF
import utils
import PBN_env

genedata = pickle.load(open("schiebinger_data_full.pkl","rb"))      #Pre-processed genedata
namemap = pickle.load(open("schiebinger_namemap_full.pkl","rb"))    #Map from indexes in the original
                                                                    #dataset to the names. Positions match.

N = 10  #Size of the network to infer

rerun = False

if rerun:
    genes = pickle.load(open("rerun_genes.tst.pkl","rb"))
else:
    genes = np.floor(np.random.rand(N) * len(namemap)).astype(int)  #randomly selected genes.
    pickle.dump(genes, open("rerun_genes.tst.pkl","wb"))

genedata = utils.trim_genedata(genedata, genes)                 #dataset is trimmed.
literals = [list(namemap.values())[x] for x in genes]           #Computing the list for literals. Using gene names.
neg_literals = ['~'+list(namemap.values())[x] for x in genes]           #Computing the list for literals. Using gene names.
literals += neg_literals
literal_order = literals[:int(len(literals)/2)]
i = 0
SCNFs = []
for gene in genes:
    relevant_transitions = utils.trim_outputs(genedata, i)  #Outputs are trimmed
    clause = SCNF.SCNF_Learn(relevant_transitions, literals)
    SCNFs += [clause]
    i += 1
PBN = SCNF.SCNF_To_PBN(SCNFs, literal_order)
function, mask = PBN[0]
env = PBN_env.PBN(PBN_data = PBN)
env.reset()
for _ in range(10):
    print(env.get_state().astype(int))
    env.step()

