import itertools
from itertools import combinations
import numpy as np

# Analyze the integrated information from a State-by-State TPM. Start with a simple case of a 3 node network A B and C
tpm = np.array([[1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,1,0,0,0,0],
                [1,0,0,0,0,0,0,0]])

# We wish to find the integrated information. We need the mutual information of the entire network, and the
# mutual information of each bipartition to find to the integrated information

n_rows = len(tpm[0])

# Assume all the past states are Discretely Uniformly Prior
Xt_past_prior = [1 / n_rows] * n_rows

At_past_prior = Xt_past_prior
Bt_past_prior = Xt_past_prior
Ct_past_prior = Xt_past_prior

Xt_prior = np.column_stack((At_past_prior, Bt_past_prior, Ct_past_prior))

print(Xt_prior)

# Present states depend on past states:
Xt_pr = np.sum(tpm, axis=0) / n_rows
Xt_pr = Xt_pr[Xt_pr > 0] # Eliminating 0 case to avoid 0 * log(0) undefined error (should just be 0)

# Compute the mutual information (no partition)
def entropy_marginal(X):
    return -np.sum(Xt_pr * np.log2(Xt_pr))

H_Xt = entropy_marginal(Xt_pr)
H_Xt_given_past = 0 # Only holds in this case because tpm is deterministic

def mi(H_Xt, H_Xt_given_past):
    return H_Xt - H_Xt_given_past

I_Xt_Xtpast = mi(H_Xt, H_Xt_given_past)
# Should be 2 for this case. So we have 2 bits of MI across the whole network

# Now we compute partitions. We need to find bi-partitions depending on the size

# Number of nodes. Log base needs to change depending on the number of states each node can have.
n_nodes: int = int(np.log2(n_rows))

# The indexes for the partitions. Labeled 0 up to n_nodes - 1
S = range(n_nodes)
bipartitions = []

# Iterate over all possible non-empty subsets. Only up to half-size since symmetric bipartitions are equivalent
for i in range(1, len(S)//2 + 1):
    for subset in combinations(S, i):
        A = set(subset)
        B = set(S) - A
        bipartitions.append((A, B))

# Ensure the bipartitions were created correctly
print(bipartitions)

# We need to loop over the bipartitions but start with the simple case first
m1 = list(bipartitions[0][0])
m2 = list(bipartitions[0][1])
print(m1, m2)

# Compute marginals
#m1t_pr =