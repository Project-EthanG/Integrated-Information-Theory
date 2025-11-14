import itertools
from itertools import combinations
from collections import defaultdict
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

n_states = len(tpm[0])

# Assume all the past states are Discretely Uniformly Prior
Xt_past_prior = [1 / n_states] * n_states

## Verify the size of prior - Is it nxp? nx1? Using nx1 for now... ##
#At_past_prior = Xt_past_prior
#Bt_past_prior = Xt_past_prior
#Ct_past_prior = Xt_past_prior

#Xt_prior = np.column_stack((At_past_prior, Bt_past_prior, Ct_past_prior))

print("P(Xt-1) =", Xt_past_prior)

def marginal_probability(X, condX):
    # Use law of total probability: P(Xt) = sum_{Xt-1} P(Xt | Xt-1) * P(Xt-1)
    X_marg = np.empty(n_states)
    for pres_state in range(n_states):
        # Iteratively add to each marginal present state probability
        X_marg[pres_state] = 0
        for past_state in range(n_states):
            X_marg[pres_state] += condX[past_state][pres_state] * X[past_state]

    return X_marg

Xt_pr = marginal_probability(Xt_past_prior, tpm)
print("P(Xt) =", Xt_pr)


# Compute the mutual information (no partition)
def marginal_entropy(X):

    H = 0
    for pres_state in range(n_states):
        # No change to entropy if we get the 0 * log(0) case (no uncertainty if nothing
        # ever happens). Normally this is undefined, but we treat it as 0 in IIT.
        # Otherwise compute shannon entropy as normal. Use log2 only if there are two
        # states per node, otherwise adjust accordingly

        if X[pres_state] != 0:
            H -= X[pres_state] * np.log2(X[pres_state])
    return H


def joint_prob(X, condX):
    n = len(X)
    p_joint = np.empty((n,n))
    # Through marginal states (i.e; past states)
    for i in range(n):
        # Through other (i.e; present states)
        for j in range(n):
            p_joint[i][j] = X[i] * condX[i][j]
    return p_joint


def conditional_entropy(X, condX, jointX):
    H = 0
    n = len(X)
    # Iterate through past states
    for i in range(n):
        # Iterate through present states
        for j in range(n):
            # No changes under 0 log(0) case
            if X[j] != 0 and condX[i][j] != 0:
                H -= jointX[i, j] * np.log2(condX[i][j])
    return H


X_joint = joint_prob(Xt_past_prior, tpm)
print("Joint pmf:", X_joint)

H_Xt = marginal_entropy(Xt_pr)
print("H(Xt) =", H_Xt)

H_Xt_Xtpast = conditional_entropy(Xt_pr, tpm, X_joint)
print("H(Xt | Xt-1) =", H_Xt_Xtpast)

def mi(H_Xt, H_Xt_Xtpast):
    return H_Xt - H_Xt_Xtpast

# Should be 2 for this case. So we have 2 bits of MI across the whole network
mi_Xt_Xtpast = mi(H_Xt, H_Xt_Xtpast)
print("MI(Xt) =", mi_Xt_Xtpast)

# Now we compute partitions. We need to find bi-partitions depending on the size

# Number of nodes. Log base needs to change depending on the number of states each node can have.
n_nodes: int = int(np.log2(n_states))

# The indexes for the partitions. Labeled 0 up to n_nodes - 1
S = range(n_nodes)
bipartitions = []

# Will be used to store the max MI across partitions. Starting at 0, it will get larger
max_mi_m1m2 = 0

# Iterate over all possible non-empty subsets. Only up to half-size since symmetric bipartitions are equivalent
for i in range(1, len(S)//2 + 1):
    for subset in combinations(S, i):
        A = set(subset)
        B = set(S) - A
        bipartitions.append((A, B))

# Ensure the bipartitions were created correctly
print("Partitions found:", bipartitions)

# Find all binary states. Used to compute probabilities for groups of nodes
def all_binary_states(n):
    # Finds every permutation of binary states. For 2 "modes" of a state,
    # there are 2^n states possible
    return list(itertools.product([0, 1], repeat=n))

m1 = list(bipartitions[0][0])
m2 = list(bipartitions[0][1])

# Need to find the subset priors first
def partition_pr_prior(Xt_past_prior, subset):
    n_partition_states = len(subset)

    # We need only the priors for where the state i is found (in a 1 node example,
    # there are 2 states, so for state 0 we need every prior where A=0 to sum)
    # We can do this with a dict to update where the keys hold that value

    all_states = all_binary_states(n_nodes)
    subset_states = all_binary_states(n_partition_states)

    # Accumulates across permutations so initialize to 0
    prior_probs = {s: 0.0 for s in subset_states}

    # Loop through each node within the corresponding prior. This is used to
    # accumulate node by node
    for state, state_pr_prior in zip(all_states, Xt_past_prior):
        # Store the "node"th element of the current permutation as a key. This
        # key is used to know which probability to "increment" each time it is
        # found in a new permutation. In the 1 node partition case, the only
        # possible keys are 0 and 1, so for the 3 node network, there are 8
        # permutations, with 4 having "key" 0, so they add 1/8 four times to get
        # P(m1 = 0) = 1/2. It does the same for "key" 1.
        key = tuple(state[i] for i in subset)
        prior_probs[key] += state_pr_prior
    return list(prior_probs.values())

m1_pr_prior = partition_pr_prior(Xt_past_prior, m1)
m2_pr_prior = partition_pr_prior(Xt_past_prior, m2)
print("P(m1t-1) =", m1_pr_prior)
print("P(m2t-1) =", m2_pr_prior)

# Compute marginal present probabilities in a similar manner
def partition_pr(Xt_pr, subset):
    n_partition_states = len(subset)

    all_states = all_binary_states(n_nodes)
    subset_states = all_binary_states(n_partition_states)

    probs = {s: 0.0 for s in subset_states}
    for state, state_pr in zip(all_states, Xt_pr):
        key = tuple(state[i] for i in subset)
        probs[key] += state_pr
    return list(probs.values())

m1_pr = partition_pr(Xt_pr, m1)
m2_pr = partition_pr_prior(Xt_pr, m2)
print("P(m1t) =", m1_pr)
print("P(m2t) =", m2_pr)


def partition_pr_cond(condX, subset):

    n_states = condX.shape[0]
    n_nodes = int(np.log2(n_states))

    all_states = all_binary_states(n_nodes)
    k = len(subset)
    subset_states = all_binary_states(k)

    # Index like in the marginal.
    past_groups = {s: [] for s in subset_states}
    for full_index, full_state in enumerate(all_states):
        sub = tuple(full_state[i] for i in subset)
        past_groups[sub].append(full_index)

    # Now compute conditional P(subset_t | subset_t-1)
    cond_subset = []

    for past_sub in subset_states:
        row = np.zeros(len(subset_states))

        # For each full past state mapping to this subset state
        for past_full_idx in past_groups[past_sub]:

            # Get full conditional distribution from TPM row
            full_row = condX[past_full_idx]

            # For each full present state, map to subset present
            for pres_full_idx, p in enumerate(full_row):
                pres_full_state = all_states[pres_full_idx]
                pres_sub = tuple(pres_full_state[i] for i in subset)
                row[subset_states.index(pres_sub)] += p

        # Normalize (since it may sum across many full states)
        row /= row.sum()

        cond_subset.append(row)

    return np.array(cond_subset)

m1_pr_cond = partition_pr_cond(tpm, m1)
m2_pr_cond = partition_pr_cond(tpm, m2)

print("P(m1_t | m1_{t-1}) =", m1_pr_cond)
print("P(m2_t | m2_{t-1}) =", m2_pr_cond)




def compute_subsystem_priors_and_tpm_matrix(tpm, p_past, partition):
    """
    Compute subsystem priors and TPMs given full TPM matrix + prior.
    - tpm: (N x N) numpy array (N = 2^n)
    - p_past: list or np.array of length N (prior over past states)
    - partition: tuple/list of variable indices (e.g. (0) or (1,2))

    Returns:
      P_past_state_sub : dict past_state_sub -> P(past_state_sub)
      P_pres_state_sub : dict pres_state_sub -> P(pres_state_sub)
      q_sub      : dict past_state_sub -> dict pres_state_sub -> P(pres_state_sub | past_state_sub)
    """

    # Find number of nodes. Adjust log2 to logn for n possible states per node
    n_nodes = int(np.log2(tpm.shape[0]))

    # Get every permutation of states. By default, nodes "change state" starting from the right, so we have
    # (0, 0), (0, 1), (1, 0), (1, 1) for a 2 node example. It is of type list(tuple), so all_states[0] returns the
    # 0th state permutation (0, 0) in a 2 node example
    all_states = all_binary_states(n_nodes)

    # Use defaultdict to allow for keys to be added as needed without any errors
    joint = defaultdict(float)
    past_state_sub_marg = defaultdict(float)
    pres_state_sub_marg = defaultdict(float)

    # Looping through each possible past state - i gives the index of the state and past_state gives the tuple of
    # nodes for that state. In the context of the tpm, i is the row index and past_state is the corresponding
    # permutation at that index
    for i, past_state in enumerate(all_states):
        # Prior for current past state
        p_x = p_past[i]

        # Extract each possible state only on the given partition from the permutations and store in a tuple.
        # This would be (0, 0, 1, 1) for a 2 node example
        past_state_sub = tuple(past_state[j] for j in partition)

        # The marginal prior from the extracted past states. Accumulates for repeated keys (why we use a
        # defaultdict)
        past_state_sub_marg[past_state_sub] += p_x

        # On the current past state, loop through the present state (i.e; cells in the current row of the tpm)
        for j, pres in enumerate(all_states):
            # Only 0 or 1 for deterministic tpm
            p_trans = tpm[i, j]

            # Skip over 0 probabilities since they do not contribute to the sum
            if p_trans == 0:
                continue

            # Extract the possible present states from all permutations
            pres_state_sub = tuple(pres[k] for k in partition)

            #
            joint[(past_state_sub, pres_state_sub)] += p_x * p_trans
            pres_state_sub_marg[pres_state_sub] += p_x * p_trans

    # normalize conditionals
    q_sub = {}
    for (past_state_sub, pres_state_sub), mass in joint.items():
        denom = past_state_sub_marg[past_state_sub]
        if denom > 0:
            q = mass / denom
        else:
            q = 0.0
        q_sub.setdefault(past_state_sub, {})[pres_state_sub] = q

    # sort for readability
    P_past_state_sub = dict(sorted(past_state_sub_marg.items()))
    P_pres_state_sub = dict(sorted(pres_state_sub_marg.items()))
    q_sub_sorted = {k: dict(sorted(v.items())) for k, v in sorted(q_sub.items())}
    return P_past_state_sub, P_pres_state_sub, q_sub_sorted


# Compute priors and marginal TPMs for each partition
P_past_m1, P_pres_m1, q_m1 = compute_subsystem_priors_and_tpm_matrix(tpm, Xt_past_prior, m1)
P_past_m2, P_pres_m2, q_m2 = compute_subsystem_priors_and_tpm_matrix(tpm, Xt_past_prior, m2)

# Compute marginal entropy and conditional entropy
def conditional_entropy(P_past_state_sub, q_sub):
    """
    Compute H(S_t | S_{t-1}) for a subsystem.
    P_past_state_sub: dict past_state_sub -> P(past_state_sub)
    q_sub: dict past_state_sub -> dict present_sub -> P(present_sub | past_state_sub)
    Returns: conditional entropy in bits.
    """
    H = 0.0
    for past_state_sub, P_past in P_past_state_sub.items():
        if P_past == 0:  # skip if impossible past
            continue
        for pres_state_sub, condP in q_sub[past_state_sub].items():
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

# Find the current maximum mi across all paritions (i.e; least damaging cut)
if mi_m1m2 > max_mi_m1m2:
    max_mi_m1m2 = mi_m1m2
    min_partition = m1, m2

ii = mi_Xt_Xtpast - max_mi_m1m2
print("Integrated information: ", ii, "\n",
      "Mutual information across the network: ", mi_Xt_Xtpast, "\n",
      "Minimum Parition: ", min_partition, "\n",
      "Minimum Mutual Information across partitions", max_mi_m1m2)



