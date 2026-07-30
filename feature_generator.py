import numpy as np
import networkx as nx


# Some pieces to consider:

# Weighted clustering coefficient: https://pmc.ncbi.nlm.nih.gov/articles/PMC374315/. Easy in python:

def weighted_clustering_coeff(G):
    C = nx.clustering(G, weight="weight")
    return np.mean(list(C.values()))

# Shortest path length. Dijstras assumes small weight = small length, which needs to be inversed since we have probs.
# A simple fix is converting the probabilities to "lengths" using a log transform


def shortest_path_length(G):
    W = nx.to_numpy_array(G, weight="weight")

    # FUTURE: Use a reciprocal transform
    W_log = -np.log(W + 1e-12)
    G_log = nx.from_numpy_array(W, create_using=nx.DiGraph)
    L = nx.average_shortest_path_length(G_log, weight="weight")
    return L

# Small world coeff. Uses C and L, but also the C_rand and L_rand that derives from randomly generated networks.
# If we randomly generate 100 networks, then we can find the small world coeff
# C_rand = 1/R * np.sum(C_r), for the R number of randomly generated networks
# L_rand = 1/R * np.sum(L_r), for the R number of randomly generated networks
# sigma = (C/C_rand) / (L/L_rand)

def randomize_weights(G):
    # Weight generation based on adj, need to convert back
    W = nx.to_numpy_array(G, weight="weight")

    W_rand = W.copy()

    mask = ~np.eye(W.shape[0], dtype=bool)

    weights = W_rand[mask]
    np.random.shuffle(weights)

    W_rand[mask] = weights

    return W_rand


def compute_random_baselines(G, R=100):

    C_random = []
    L_random = []

    for _ in range(R):
        W_rand = randomize_weights(G)
        G_rand = nx.from_numpy_array(W_rand, create_using=nx.DiGraph)

        C_r = weighted_clustering_coeff(G_rand)
        L_r = shortest_path_length(G_rand)

        C_random.append(C_r)
        L_random.append(L_r)

    C_rand = np.mean(C_random)
    L_rand = np.mean(L_random)

    return C_rand, L_rand


def small_world_qty(G, R=100):
    C = weighted_clustering_coeff(G)
    L = shortest_path_length(G)

    C_rand, L_rand = compute_random_baselines(G, R)
    sigma = (C / C_rand) / (L / L_rand)

    return sigma


# Cheeger inequality. Finding the actual cheeger coefficient is O(2^n) since it also involves every bipartition,
# so we need to approximate the term. The inequality is given by:
# 1/2 * lambda2 <= cheeg_coeff <= np.sqrt(2 * lambda2), where lambda2 is the second smallest eigenvalue of the
# normalized Laplacian matrix for the graph G. Laplacian matrix in nx expects an undirected graph, so we can instead
# use the Chung Directed Laplacian:
# L = (np.eye(n) - 0.5 * (Pi_sqrt @ G @ Pi_inv_sqrt + Pi_inv_sqrt @ G.T @ Pi_sqrt)), where Pi is the diagonals
# of the stationary distribution for G.
# Or because that is annoyingly complex, we can just symmetrize the original adj matrix then use nx to get the
# second eigenval


def cheeger_qty(W):
    # FUTURE: Try using the chung directed laplacian
    Gs = (W + W.T) / 2
    G_sym = nx.from_numpy_array(Gs)
    L = nx.normalized_laplacian_matrix(G_sym, weight="weight")

    eigvals = np.sort(np.abs(np.linalg.eigvals(L.toarray())))[::-1]
    lambda2 = eigvals[1]

    return lambda2


# For SSCs, nx does a pretty good job for no SCCs, max SCC and diam of max SCC
def scc_qtys(G):

    num_nodes = G.number_of_nodes()

    sccs = list(nx.strongly_connected_components(G))
    num_sccs = len(sccs)

    max_scc = max(len(s) for s in sccs) / num_nodes

    largest_scc = max(sccs, key=len)
    H = G.subgraph(largest_scc)
    diam = nx.diameter(H)

    return num_sccs, max_scc, diam


# Centrality measures. Closeness has the same issue as Djkstras - need a log transform to convert probs to distance
def centrality_qtys(G):
    W = nx.to_numpy_array(G, weight="weight")
    # FUTURE: Reciprocal
    W_distance = -np.log(W + 1e-12)
    G_distance = nx.from_numpy_array(W_distance, create_using=nx.DiGraph)
    closeness = nx.closeness_centrality(G_distance, distance="weight")
    avg_closeness = np.mean(list(closeness.values()))

    betweenness = nx.betweenness_centrality(G)
    avg_betweenness = np.mean(list(betweenness.values()))

    return avg_closeness, avg_betweenness


# Page rank uses nx as well
def pagerank_qtys(G):
    # FUTURE: figure out what alpha does
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

    pr_values = np.array(list(pagerank.values()))

    max_pagerank = np.max(pr_values)
    min_pagerank = np.min(pr_values)
    mean_pagerank = np.mean(pr_values)

    return max_pagerank, min_pagerank, mean_pagerank


# Spectral values use numpy, no directed graph needed. Main mixing gap for markov chains to consider is 1 - |l2|
def spectral_qtys(W):

    num_nodes = W.shape[0]

    eigvals = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]

    # FUTURE: needs to be second largest
    lambda2 = eigvals[1]
    mixing_gap = 1 - lambda2

    # Spectral entropy
    p = eigvals / np.sum(eigvals)

    spectral_entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(num_nodes)

    return mixing_gap, spectral_entropy


# Weighted density is not available in nx (only undirected density), so we have to do it manually
def weighted_density(W):
    n = W.shape[0]
    total_weight = np.sum(W) - np.trace(W)

    return total_weight / (n * (n - 1))

# For cycle derived qtys are a bit tough since they are O(2^n) in worst case, we will have to come back to this


# For weighted reciprocity (measure of bidirectional causal coupling), we need to do it manually
def weighted_reciprocity(W):
    W_no_diag = W.copy()
    np.fill_diagonal(W_no_diag, 0)

    numerator = np.sum(np.sqrt(W_no_diag * W_no_diag.T))
    denominator = np.sum(W_no_diag)

    return numerator / denominator



'''
1) No. of strongly connected components SCCs
2) Max diameter of SSCs (biggest SSC)
3) Average betweenness centrality 
4) Avg closeness centrality
5) Max/min score for page rank
6) Largest eigenval
7) Spectral gap
8) Spectral entropy
9) Weighted density
10) No. of cycles
11) Average cycle length
12) Max cycle length
13) Weighted reciprocity

14) Weighted clustering coeff 
15) Minimum path length 
16) Small world coeff 
17) Cheeger constant (approximation) 
'''


def sbs_to_sbn(tpm_sbs):
    tpm_sbs = np.asarray(tpm_sbs, dtype=float)

    N, M = tpm_sbs.shape
    n = int(np.log2(N))

    # All binary states in lexicographic ordering
    states = ((np.arange(N)[:, None] >>
               np.arange(n - 1, -1, -1)) & 1).astype(float)

    # Matrix multiplication does the conversion neatly
    tpm_sbn = tpm_sbs @ states

    return tpm_sbn


def sbn_to_nbn(tpm_sbn):
    N, n = tpm_sbn.shape
    W = np.zeros((n, n))

    # Perturb "causal" nodes
    for source in range(n):

        # Represent state idx as a bit relative to the node. Python bits are reversed so need to subtract from n
        bit = n - 1 - source

        # Loop over "effect" nodes
        for target in range(n):

            diffs = []

            for state in range(N):

                # Bit flip to obtain the symmetric state (i.e; 000 has partner 100 under node A perturbation)
                partner = state ^ (1 << bit)

                # Pairs appear twice, only need to take one
                if state < partner:
                    # Perturbation needs all "other causal" node states to be considered to eventually average
                    delta = abs(
                        tpm_sbn[partner, target]
                        - tpm_sbn[state, target]
                    )
                    diffs.append(delta)
            # Average the causal effects across all fixed "other causal" nodes.
            W[source, target] = np.mean(diffs)

    return W


def compute_nbn_features(sbs_tpm):
    sbn_tpm = sbs_to_sbn(sbs_tpm)

    # W is the adj matrix
    W = sbn_to_nbn(sbn_tpm)

    # The adj matrix features:
    mixing_gap, spectral_entropy = spectral_qtys(W)
    wd = weighted_density(W)
    wr = weighted_reciprocity(W)

    # G is the directed graph from the adj matrix
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)

    # The directed graph features:
    weight_cluster_coeff = weighted_clustering_coeff(G)
    short_path_len = shortest_path_length(G)
    small_world_coeff = small_world_qty(G)
    cheeger_coeff = cheeger_qty(W)
    num_sccs, max_scc, diam = scc_qtys(G)
    avg_closeness, avg_betweenness = centrality_qtys(G)
    max_pr, min_pr, mean_pr = pagerank_qtys(G)

    return [
        mixing_gap, spectral_entropy, wd, wr,
        weight_cluster_coeff, short_path_len, small_world_coeff, cheeger_coeff,
        num_sccs, max_scc, diam, avg_closeness,
        avg_betweenness, max_pr, min_pr, mean_pr
    ]






