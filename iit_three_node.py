import itertools
from collections import defaultdict
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
    return -1 * np.sum(Xt_pr * np.log2(Xt_pr))

H_Xt = entropy_marginal(Xt_pr)
H_Xt_given_past = 0 # Only holds in this case because tpm is deterministic

def mi(H_Xt, H_Xt_given_past):
    return H_Xt - H_Xt_given_past

mi_Xt_Xtpast = mi(H_Xt, H_Xt_given_past)
# Should be 2 for this case. So we have 2 bits of MI across the whole network

# Now we compute partitions. We need to find bi-partitions depending on the size

# Number of nodes. Log base needs to change depending on the number of states each node can have.
n_nodes: int = int(np.log2(n_rows))

# The indexes for the partitions. Labeled 0 up to n_nodes - 1
S = range(n_nodes)
bipartitions = []

# Will be used to store the minimum MI across partitions. Set to a safe "max"
min_mi_m1m2 = mi(H_Xt, H_Xt_given_past) * 100

# Iterate over all possible non-empty subsets. Only up to half-size since symmetric bipartitions are equivalent
for i in range(1, len(S)//2 + 1):
    for subset in combinations(S, i):
        A = set(subset)
        B = set(S) - A
        bipartitions.append((A, B))

# Ensure the bipartitions were created correctly
print(bipartitions)

min_partition = bipartitions[0][0], bipartitions[0][1]

# We need to loop over the bipartitions but start with the simple case first
m1 = list(bipartitions[0][0])
m2 = list(bipartitions[0][1])

def all_binary_states(n):
    """Return list of all 2^n binary tuples."""
    return list(itertools.product([0, 1], repeat=n))


def compute_subsystem_priors_and_tpm_matrix(tpm, p_past, subset):
    """
    Compute subsystem priors and TPMs given full TPM matrix + prior.
    - tpm: (N x N) numpy array (N = 2^n)
    - p_past: list or np.array of length N (prior over past states)
    - subset: tuple/list of variable indices (e.g. (0,) or (1,2))

    Returns:
      P_past_sub : dict past_sub -> P(past_sub)
      P_pres_sub : dict pres_sub -> P(pres_sub)
      q_sub      : dict past_sub -> dict pres_sub -> P(pres_sub | past_sub)
    """
    n_nodes = int(np.log2(tpm.shape[0]))
    all_states = all_binary_states(n_nodes)
    joint = defaultdict(float)
    past_sub_marg = defaultdict(float)
    pres_sub_marg = defaultdict(float)

    for i, past in enumerate(all_states):
        p_x = p_past[i]
        past_sub = tuple(past[j] for j in subset)
        past_sub_marg[past_sub] += p_x

        for j, pres in enumerate(all_states):
            p_trans = tpm[i, j]
            if p_trans == 0:
                continue
            pres_sub = tuple(pres[k] for k in subset)
            joint[(past_sub, pres_sub)] += p_x * p_trans
            pres_sub_marg[pres_sub] += p_x * p_trans

    # normalize conditionals
    q_sub = {}
    for (past_sub, pres_sub), mass in joint.items():
        denom = past_sub_marg[past_sub]
        if denom > 0:
            q = mass / denom
        else:
            q = 0.0
        q_sub.setdefault(past_sub, {})[pres_sub] = q

    # sort for readability
    P_past_sub = dict(sorted(past_sub_marg.items()))
    P_pres_sub = dict(sorted(pres_sub_marg.items()))
    q_sub_sorted = {k: dict(sorted(v.items())) for k, v in sorted(q_sub.items())}
    return P_past_sub, P_pres_sub, q_sub_sorted


# Compute priors and marginal TPMs for each partition
P_past_m1, P_pres_m1, q_m1 = compute_subsystem_priors_and_tpm_matrix(tpm, Xt_past_prior, m1)
P_past_m2, P_pres_m2, q_m2 = compute_subsystem_priors_and_tpm_matrix(tpm, Xt_past_prior, m2)

# Compute marginal entropy and conditional entropy
def conditional_entropy(P_past_sub, q_sub):
    """
    Compute H(S_t | S_{t-1}) for a subsystem.
    P_past_sub: dict past_sub -> P(past_sub)
    q_sub: dict past_sub -> dict present_sub -> P(present_sub | past_sub)
    Returns: conditional entropy in bits.
    """
    H = 0.0
    for past_sub, P_past in P_past_sub.items():
        if P_past == 0:  # skip if impossible past
            continue
        for pres_sub, condP in q_sub[past_sub].items():
            if condP > 0:
                H -= P_past * condP * np.log2(condP)
    return H


def marginal_entropy(P_dist):
    """
    Compute Shannon entropy H(X) in bits from a probability distribution.

    Parameters
    ----------
    P_dist : dict
        Keys = states (e.g., tuples or ints)
        Values = probabilities (must sum to 1)

    Returns
    -------
    H : float
        Entropy in bits.
    """
    H = 0.0
    for p in P_dist.values():
        if p > 0:
            H -= p * np.log2(p)
    return H

def mi_across_partitions(H_m1, H_m2, H_m1_given_past, H_m2_given_past):
    mi_m1 = H_m1 - H_m1_given_past
    mi_m2 = H_m2 - H_m2_given_past
    return mi_m1 + mi_m2

# Compute marginal and conditional entropies
H_m1 = marginal_entropy(P_pres_m1)
H_m2 = marginal_entropy(P_pres_m2)
H_m1_given_past = conditional_entropy(P_past_m1, q_m1)
H_m2_given_past = conditional_entropy(P_past_m2, q_m2)

# Use the entropies to compute the mutual information across the partition
mi_m1m2 = mi_across_partitions(H_m1, H_m2, H_m1_given_past, H_m2_given_past)

# Find the current minimum mi across all paritions (i.e; least damaging cut)
if mi_m1m2 < min_mi_m1m2:
    min_mi_m1m2 = mi_m1m2
    min_partition = m1, m2

ii = mi_Xt_Xtpast - min_mi_m1m2
print("Integrated information: ", ii, "\n",
      "Mutual information across the network: ", mi_Xt_Xtpast, "\n",
      "Minimum Parition: ", min_partition, "\n",
      "Minimum Mutual Information across partitions", min_mi_m1m2)



