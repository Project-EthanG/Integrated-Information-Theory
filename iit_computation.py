import itertools
from itertools import combinations
import numpy as np
from tpm_generator import generate_random_det_tpm

# Define all necessary function prior to computing integrated information

def uniform_prior(tpm):
    n_states = tpm.shape[0]
    return [1 / n_states] * n_states


def marginal_probability(X, tpm):
    # Number of states; corresponds to number of rows in tpm (or column)
    n_states = tpm.shape[0]

    # Use law of total probability: P(Xt) = sum_{Xt-1} P(Xt | Xt-1) * P(Xt-1)
    X_marg = np.empty(n_states)
    for pres_state in range(n_states):
        # Iteratively add to each marginal present state probability
        X_marg[pres_state] = 0
        for past_state in range(n_states):
            X_marg[pres_state] += tpm[past_state][pres_state] * X[past_state]

    return X_marg


# Compute the mutual information (no partition)
def marginal_entropy(X):
    # Number of states (rows)
    n_states = X.shape[0]

    H = 0
    for pres_state in range(n_states):
        # No change to entropy if we get the 0 * log(0) case (no uncertainty if nothing
        # ever happens). Normally this is undefined, but we treat it as 0 in IIT.
        # Otherwise compute shannon entropy as normal. Use log2 only if there are two
        # states per node, otherwise adjust accordingly

        if X[pres_state] > 0:
            H -= X[pres_state] * np.log2(X[pres_state])
    return H


def joint_prob(X, tpm):
    n = len(X)
    p_joint = np.empty((n, n))
    # Through marginal states (i.e; past states)
    for i in range(n):
        # Through other (i.e; present states)
        for j in range(n):
            p_joint[i][j] = X[i] * tpm[i][j]
    return p_joint


def conditional_entropy(X, tpm, jointX):
    H = 0
    n = len(X)
    # Iterate through past states
    for i in range(n):
        # Iterate through present states
        for j in range(n):
            # No changes under 0 log(0) case
            if X[j] > 0 and tpm[i][j] > 0:
                H -= jointX[i, j] * np.log2(tpm[i][j])
    return H


def mi(H_full, H_pres_past):
    return H_full - H_pres_past


#########################
# PARTITION COMPUTATION #
#########################

# Need to define the functions to compute the prior, present and conditional
# probabilities, used to compute marginal and conditional entropies to find
# the maximum mutual information (MI in the least damaging cut). This will
# give the integrated information across the system

# Find all binary states. Used to compute probabilities for groups of nodes
def all_binary_states(n):
    # Finds every permutation of binary states. For 2 "modes" of a state,
    # there are 2^n states possible
    return list(itertools.product([0, 1], repeat=n))


def generate_bipartitions(tpm):
    # Generate the bipartitions based on every possible state of nodes
    n_states = tpm.shape[0]
    n_nodes = int(np.log2(n_states))

    S = range(n_nodes)
    bipartitions = []

    # Iterate over all possible non-empty subsets. Only up to half-size since symmetric bipartitions are equivalent
    for i in range(1, len(S) // 2 + 1):
        for subset in combinations(S, i):
            A = set(subset)
            B = set(S) - A
            bipartitions.append((A, B))

    return bipartitions


# Need to find the subset priors first
def partition_pr_prior(prior, subset):
    n_partition_states = len(subset)
    n_nodes = int(np.log2(len(prior)))  # cast as int since it is coming from np

    all_states = all_binary_states(n_nodes)
    subset_states = all_binary_states(n_partition_states)

    # Accumulates across permutations so initialize to 0
    prior_probs = {s: 0.0 for s in subset_states}

    # Loop through each node within the corresponding prior. This is used to
    # accumulate node by node
    for state, state_pr_prior in zip(all_states, prior):
        # Store the "node"th element of the current permutation as a key. This
        # key is used to know which probability to "increment" each time it is
        # found in a new permutation. In the 1 node partition case, the only
        # possible keys are 0 and 1, so for the 3 node network, there are 8
        # permutations, with 4 having "key" 0, so they add 1/8 four times to get
        # P(m1 = 0) = 1/2. It does the same for "key" 1.
        key = tuple(state[i] for i in subset)
        prior_probs[key] += state_pr_prior
    return list(prior_probs.values())


# Compute marginal present probabilities in a similar manner to the prior for the
# partition
def partition_pr(full_pres, subset):
    n_nodes = int(np.log2(len(full_pres)))
    n_partition_states = len(subset)

    all_states = all_binary_states(n_nodes)
    subset_states = all_binary_states(n_partition_states)

    probs = {s: 0.0 for s in subset_states}
    for state, state_pr in zip(all_states, full_pres):
        key = tuple(state[i] for i in subset)
        probs[key] += state_pr

    return list(probs.values())


def partition_pr_cond(tpm, prior, subset):
    # Inputs:
    #   tpm (2D np.array(float)): transition probability matrix for the full system
    #   prior (list(float)):  prior probability for system
    #   subset (2D list(int)): partition of interest

    # Outputs:
    #   Q_s (2D np.array(float)): conditional probability distribution for subset

    n_states = tpm.shape[0]
    n_nodes = int(np.log2(n_states))

    ## 1) Compute joint ##

    # Method A) Use explicit looping

    # To hold joint prob dist. of subsystem
    joint_full = np.zeros((n_states, n_states), dtype=float)

    # Loop through past states
    for i in range(n_states):
        # Loop through present states
        for j in range(n_states):
            joint_full[i][j] = prior[i] * tpm[i][j]

    ## 2) Project full indices to subsystem indices ##

    # Every permutation of states in the full system
    all_n_states = all_binary_states(n_nodes)

    # Number of nodes in subsystem
    k = len(subset)

    # Every permutation of states in the subsystem
    all_s_states = all_binary_states(k)

    # Store the subsystem states in a dict to map to corresponding indices

    # Optimal method for extracting indices and values from the list
    s_index_mapping = {state: s for s, state in enumerate(all_s_states)}

    # Mapping from the full system indices to the subsystem indices
    full_to_sub = np.zeros(shape=n_states, dtype=int)

    # Using both the indices and the permutations from all states
    for n_index, full_state in enumerate(all_n_states):
        # Obtain the corresponding nodes states in the current permutation
        sub_key = tuple(full_state[k] for k in subset)

        # Store the corresponding index value for the subsystem key selected
        s_index = s_index_mapping[sub_key]

        # Store in a list to properly order the subsystem lexicographically
        full_to_sub[n_index] = s_index

    # Aggregate joint into subsystem

    # Number of state permutations in the subsystem
    m = 2 ** k

    # To hold the joint prob. dist. for the subsystem. Should be (MxM)
    joint_s = np.zeros((m, m))

    # Looping through the past states of the full joint network
    for i in range(0, n_states):
        # Index for the past state of the subsystem
        u = full_to_sub[i]

        # Looping through the present states of the full joint network
        for j in range(0, n_states):
            # Index for the present state of the subsystem
            v = full_to_sub[j]

            # Add the values (prior already accounted for in determining the full joint)
            joint_s[u][v] += joint_full[i][j]

    ## 4) Compute marginal past from J(s) and the cond. pres. given past under the subsystem

    # Prior
    prior = partition_pr_prior(prior, subset)

    # Conditional dist.
    Q_s = np.zeros((m, m))
    # Method 1) Raw dog
    for u in range(m):
        for v in range(m):
            # Need to account for 0 case
            if prior[u] > 0:
                Q_s[u][v] = joint_s[u][v] / prior[u]

    return Q_s

# Compute marginal entropy and conditional entropy
def partition_conditional_entropy(s_prior, s_pr_cond):
    # Inputs:
    #   s_prior list(float): prior probability distribution
    #   s_pr_cond: cpd for present probabilities conditioned on prior

    # Outputs:
    #   H (float): conditional entropy of the system
    H = 0.0
    k = len(s_prior)
    n = len(s_pr_cond[0])
    for i in range(k):
        for j in range(n):
            if s_pr_cond[i][j] > 0 and s_prior[i] > 0:
                H -= (s_prior[i] * s_pr_cond[i][j]
                      * np.log2(s_pr_cond[i][j]))
    return H


def partition_marginal_entropy(s_pr):
    # Inputs:
    #   s_pr list(float): probability distribution

    # Outputs:
    #   H (float): entropy of the system under the probability distribution
    H = 0.0
    for p in s_pr:
        if p > 0:
            H -= p * np.log2(p)
    return H


def mi_across_partitions(H_m1, H_m2, H_m1_m1past, H_m2_m2past):
    mi_m1 = H_m1 - H_m1_m1past
    mi_m2 = H_m2 - H_m2_m2past
    return mi_m1 + mi_m2


def max_mi_bipartition(prior, full_pres, tpm) -> tuple[float, tuple[list[int]]]:
    # Generate every possible partition. Assumed to only be bipartitions (i.e; least
    # damaging if we cut the least amount possible)
    bipartitions = generate_bipartitions(tpm)

    # We will need to store the partition with the maximum mutual information
    max_partition = tuple()
    max_mi_m1m2 = 0

    for m1, m2 in bipartitions:
        m1 = list(m1)
        m2 = list(m2)

        # Prior probabilities
        m1_pr_prior = partition_pr_prior(prior, m1)
        m2_pr_prior = partition_pr_prior(prior, m2)

        # Present probabilities; treated as marginal even in the multiple case
        m1_pr = partition_pr(full_pres, m1)
        m2_pr = partition_pr(full_pres, m2)

        # Conditional probabilities
        m1_pr_cond = partition_pr_cond(tpm, prior, m1)
        m2_pr_cond = partition_pr_cond(tpm, prior, m2)

        # Conditional entropies
        H_m1_m1past = partition_conditional_entropy(m1_pr_prior, m1_pr_cond)
        H_m2_m2past = partition_conditional_entropy(m2_pr_prior, m2_pr_cond)

        # Marginal entropies
        H_m1 = partition_marginal_entropy(m1_pr)
        H_m2 = partition_marginal_entropy(m2_pr)

        # Use the entropies to compute the mutual information across the partition
        mi_m1m2 = mi_across_partitions(H_m1, H_m2, H_m1_m1past, H_m2_m2past)

        # Find the current maximum mi across all paritions (i.e; least damaging cut)
        if mi_m1m2 > max_mi_m1m2:
            max_mi_m1m2 = mi_m1m2
            max_partition = m1, m2

    return max_mi_m1m2, max_partition


def integrated_information(tpm, prior) -> tuple[float, float, tuple[list[int]] | None, float]:
    # We wish to find the integrated information. We need the mutual information of
    # the entire network, and the maximum mutual information across each bipartition
    # to find to the integrated information

    # Compute the marginal present probabilities and joint prior and present
    full_pres = marginal_probability(prior, tpm)
    X_joint = joint_prob(prior, tpm)

    # Marginal present entropy and entropy from pres conditional on past across the whole system
    H_full = marginal_entropy(full_pres)
    H_pres_past = conditional_entropy(full_pres, tpm, X_joint)

    # Mutual information across the whole system based on the entropies
    mi_Xt_Xtpast: float = mi(H_full, H_pres_past)

    # Now we compute partitions. We need to find bi-partitions depending on the size

    # Find the prior, present and conditional probabilities for every bipartition. Use
    # these intermediary quantities to find the maximum mutual information
    max_mi, max_bipartition = max_mi_bipartition(prior, full_pres, tpm)

    # Compute the integrated information
    ii = mi_Xt_Xtpast - max_mi

    # If a non-empty max_bipartition tuple (i.e; least damaging cut exists)
    if max_bipartition:
        return ii, mi_Xt_Xtpast, max_bipartition, max_mi

    # Otherwise every cut contributes the same (i.e; 0 MI on every cut; fully
    # integrated system)
    return (ii, mi_Xt_Xtpast, None,
            max_mi)
