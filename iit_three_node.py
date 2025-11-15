import itertools
from itertools import combinations
from collections import defaultdict
import numpy as np

def uniform_prior(tpm):
    n_states = tpm.shape[0]
    return [1 / n_states] * n_states


def marginal_probability(X, condX):
    # Number of states; corresponds to number of rows in tpm (or column)
    n_states = condX.shape[0]

    # Use law of total probability: P(Xt) = sum_{Xt-1} P(Xt | Xt-1) * P(Xt-1)
    X_marg = np.empty(n_states)
    for pres_state in range(n_states):
        # Iteratively add to each marginal present state probability
        X_marg[pres_state] = 0
        for past_state in range(n_states):
            X_marg[pres_state] += condX[past_state][pres_state] * X[past_state]

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


def mi(H_Xt, H_Xt_Xtpast):
    return H_Xt - H_Xt_Xtpast


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
def partition_pr_prior(Xt_past_prior, subset):

    n_partition_states = len(subset)
    n_nodes = int(np.log2(len(Xt_past_prior))) # cast as int since it is coming from np
    print("n_nodes", n_nodes)

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


# Compute marginal present probabilities in a similar manner
def partition_pr(Xt_pr, subset):
    n_nodes = int(np.log2(len(Xt_pr)))
    n_partition_states = len(subset)

    all_states = all_binary_states(n_nodes)
    subset_states = all_binary_states(n_partition_states)

    probs = {s: 0.0 for s in subset_states}
    for state, state_pr in zip(all_states, Xt_pr):
        key = tuple(state[i] for i in subset)
        probs[key] += state_pr
    return list(probs.values())


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


# Compute marginal entropy and conditional entropy
def partition_conditional_entropy(subset_pr_past, subset_pr_cond):
    H = 0.0
    k = len(subset_pr_past)
    n = len(subset_pr_cond[0])
    for i in range(k):
        for j in range(n):
            if subset_pr_cond[i][j] > 0 and subset_pr_past[i] > 0:
                H -= (subset_pr_past[i] * subset_pr_cond[i][j]
                      * np.log2(subset_pr_cond[i][j]))
    return H


def partition_marginal_entropy(subset_pr):
    H = 0.0
    for p in subset_pr:
        if p > 0:
            H -= p * np.log2(p)
    return H


def mi_across_partitions(H_m1, H_m2, H_m1_m1past, H_m2_m2past):
    mi_m1 = H_m1 - H_m1_m1past
    mi_m2 = H_m2 - H_m2_m2past
    return mi_m1 + mi_m2


def max_mi_bipartition(Xt_pr_prior, Xt_pr, tpm):
    # Need the number of states and corresponding "nodes" (how many neurons can we
    # separate?)
    n_states = tpm.shape[0]

    # Number of nodes. Log base needs to change depending on the number of states
    # each node can have
    n_nodes: int = int(np.log2(n_states))

    # Generate every possible partition. Assumed to only be bipartitions (i.e; least
    # damaging if we cut the least amount possible)
    bipartitions = generate_bipartitions(tpm)
    print("Partitions found:", bipartitions, "\n")

    # We will need to store the partition with the maximum mutual information
    max_partition = tuple()
    max_mi_m1m2 = 0

    for m1, m2 in bipartitions:
        print("\nCurrent partition:", m1, m2)
        m1 = list(m1)
        m2 = list(m2)

        # Prior probabilities
        m1_pr_prior = partition_pr_prior(Xt_pr_prior, m1)
        m2_pr_prior = partition_pr_prior(Xt_pr_prior, m2)
        print("P(m1t-1) =", m1_pr_prior)
        print("P(m2t-1) =", m2_pr_prior)

        # Present probabilities; treated as marginal even in the multiple case
        m1_pr = partition_pr(Xt_pr, m1)
        m2_pr = partition_pr_prior(Xt_pr, m2)
        print("P(m1t) =", m1_pr)
        print("P(m2t) =", m2_pr)

        # Conditional probabilities
        m1_pr_cond = partition_pr_cond(tpm, m1)
        print("P(m1_t | m1_{t-1}) =", m1_pr_cond)
        m2_pr_cond = partition_pr_cond(tpm, m2)
        print("P(m2_t | m2_{t-1}) =", m2_pr_cond)

        # Conditional entropies
        H_m1_m1past = partition_conditional_entropy(m1_pr_prior, m1_pr_cond)
        print("H(At | At-1) =", H_m1_m1past)
        H_m2_m2past = partition_conditional_entropy(m2_pr_prior, m2_pr_cond)
        print("H(BtCt | Bt-1Ct-1) =", H_m2_m2past)

        # Marginal entropies
        H_m1 = partition_marginal_entropy(m1_pr_prior)
        print("H(At) =", H_m1)
        H_m2 = partition_marginal_entropy(m2_pr_prior)
        print("H(BtCt) =", H_m2)

        # Use the entropies to compute the mutual information across the partition
        mi_m1m2 = mi_across_partitions(H_m1, H_m2, H_m1_m1past, H_m2_m2past)

        # Find the current maximum mi across all paritions (i.e; least damaging cut)
        if mi_m1m2 > max_mi_m1m2:
            max_mi_m1m2 = mi_m1m2
            max_partition = m1, m2

    return max_mi_m1m2, max_partition


def integrated_information(tpm, Xt_pr_prior):
    # We wish to find the integrated information. We need the mutual information of
    # the entire network, and the maximum mutual information across each bipartition
    # to find to the integrated information

    # Display the inputs (tpm and prior)
    print("Inputted TPM: \n", tpm)
    print("P(Xt-1) =", Xt_pr_prior)

    # Compute the marginal present probabilities for the states from the prior and tpm
    Xt_pr = marginal_probability(Xt_pr_prior, tpm)
    print("P(Xt) =", Xt_pr)

    # Compute jointly present and past states (used in conditional entropy calculation)
    X_joint = joint_prob(Xt_pr_prior, tpm)
    print("Joint pmf:", X_joint)

    # Marginal present entropy across the whole system
    H_Xt = marginal_entropy(Xt_pr)
    print("H(Xt) =", H_Xt)

    # Conditional entropy of the present state of the whole system given the past state
    H_Xt_Xtpast = conditional_entropy(Xt_pr, tpm, X_joint)
    print("H(Xt | Xt-1) =", H_Xt_Xtpast)

    # Mutual information across the whole system based on the entropies
    mi_Xt_Xtpast = mi(H_Xt, H_Xt_Xtpast)
    print("MI(Xt) =", mi_Xt_Xtpast)

    ##########################
    # PARTITION COMPUTATIONS #
    ##########################

    # Now we compute partitions. We need to find bi-partitions depending on the size

    # Find the prior, present and conditional probabilities for every bipartition. Use
    # these intermediary quantities to find the maximum mutual information
    max_mi, max_bipartition = max_mi_bipartition(Xt_pr_prior, Xt_pr, tpm)

    # Compute the integrated information
    ii = mi_Xt_Xtpast - max_mi

    return ii, mi_Xt_Xtpast, max_bipartition, max_mi

# TEST CASES:

# CASE 1: Mutual information is the same across every bipartition

# State-by-state TPM
tpm = np.array([[1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,1,0,0,0,0],
                [1,0,0,0,0,0,0,0]])

# Use uniform prior for this test
prior = uniform_prior(tpm)

# Compute the integrated information and corresponding intermediary quantities
case1 = integrated_information(tpm, prior)

print("\nIntegrated information: ", case1[0], "\n",
          "Mutual information across the network: ", case1[1], "\n",
          "Least Damaging Partition: ", case1[2], "\n",
          "Minimum Mutual Information across partitions", case1[3])


