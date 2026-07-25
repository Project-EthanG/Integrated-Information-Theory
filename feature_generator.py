import numpy as np
import networkx as nx


def compute_scc(G):
    sccs = list(nx.strongly_connected_components(G))

    num_sccs = len(sccs)
    max_scc = max((len(c) for c in sccs), default=0)

    H = G.subgraph(max(sccs, key=len)).copy() if sccs else G
    lengths = dict(nx.all_pairs_shortest_path_length(H))

    diameter = 0
    for _, d in lengths.items():
        if d:
            diameter = max(diameter, max(d.values()))

    return int(num_sccs), int(max_scc), int(diameter)


def weighted_clustering(W):
    W = np.asarray(W)
    n = W.shape[0]

    W_max = np.max(W)
    if W_max == 0:
        return 0.0, 0.0, 0.0

    Wn = W / W_max
    Wg = np.cbrt(Wn)

    C = np.zeros(n)

    for i in range(n):
        numerator = 0.0
        denom = 0.0

        for j in range(n):
            for k in range(n):
                if j != i and k != i and j != k:
                    if Wn[j, i] > 0 and Wn[i, k] > 0 and Wn[k, j] > 0:
                        numerator += Wg[j, i] * Wg[i, k] * Wg[k, j]
                        denom += 1

        C[i] = numerator / denom if denom > 0 else 0.0

    return float(C.mean()), float(C.var()), float(C.max())


def centrality_measures(W):
    n = W.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] != 0:
                G.add_edge(j, i, weight=W[i, j])

    eps = 1e-12

    def inv_weight(u, v, d):
        return 1.0 / (d.get("weight", 0.0) + eps)

    bet = np.array(list(nx.betweenness_centrality(G, weight=lambda u, v, d: inv_weight(u, v, d)).values()))
    clo = np.array(list(nx.closeness_centrality(G, distance=lambda u, v, d: inv_weight(u, v, d)).values()))
    pr  = np.array(list(nx.pagerank(G, weight="weight").values()))

    return (
        float(bet.mean()), float(bet.var()), float(bet.max()),
        float(clo.mean()), float(clo.var()), float(clo.max()),
        float(pr.mean()), float(pr.var()), float(pr.max())
    )


def spectral_features(W):
    eig_vals = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]

    lambda1 = eig_vals[0]
    spec_gap = lambda1 - eig_vals[1] if len(eig_vals) > 1 else 0.0

    total = np.sum(eig_vals)
    p = eig_vals / (total + 1e-12)
    p = np.clip(p, 1e-12, 1.0)

    spec_entropy = -np.sum(p * np.log(p))

    return float(lambda1), float(spec_gap), float(spec_entropy)


def graph_density(W):
    n = W.shape[0]
    max_edges = n * (n - 1)
    weighted_sum = np.sum(W * (~np.eye(n, dtype=bool)))
    weighted_density = weighted_sum / max_edges
    # Look at distribution as well as the main value
    return weighted_density


def cycle_features(G):
    cycles = list(nx.simple_cycles(G))

    if len(cycles) == 0:
        return 0.0, 0.0, 0.0

    lengths = np.array([len(c) for c in cycles])

    return (
        float(len(lengths)),
        float(lengths.mean()),
        float(lengths.max())
    )

def reciprocity(W):
    n = W.shape[0]

    # Remove self loops
    mask = ~np.eye(n, dtype=bool)
    W_masked = W * mask

    W_sym_min = np.minimum(W_masked, W_masked.T)
    weighted_num = np.sum(W_sym_min)
    weighted_den = np.sum(W_masked)

    eps = 1e-12
    weighted_reciprocity = weighted_num / (weighted_den + eps)

    return weighted_reciprocity


def nbn_to_dg(tpm, threshold = 0.2):
    W = np.asarray(tpm)
    n = W.shape[0]

    # Build adjacency and graph
    adj = W > threshold
    n = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    src, dst = np.where(adj)
    edges = list(zip(dst, src))
    G.add_edges_from(edges)
    return W, G


def compute_nbn_features(tpm):
    W, G = nbn_to_dg(tpm)

    # clustering
    c_mean, c_var, c_max = weighted_clustering(W)

    # centrality (9 scalars)
    (
        bt_m, bt_v, bt_x,
        cl_m, cl_v, cl_x,
        pr_m, pr_v, pr_x
    ) = centrality_measures(W)

    # spectral (3 scalars)
    lambda1, spectral_gap, spectral_entropy = spectral_features(W)

    # density
    weighted_density = graph_density(W)

    # reciprocity
    weighted_reciprocity = reciprocity(W)

    # SCC (3 ints)
    num_sccs, max_scc, diam = compute_scc(G)

    # cycles (3 scalars)
    num_cycles, mean_cycle, max_cycle = cycle_features(G)

    return (
        c_mean, c_var, c_max,

        bt_m, bt_v, bt_x,
        cl_m, cl_v, cl_x,
        pr_m, pr_v, pr_x,

        lambda1, spectral_gap, spectral_entropy,

        weighted_density,
        weighted_reciprocity,

        num_sccs, max_scc, diam,

        num_cycles, mean_cycle, max_cycle
    )



# Clustering coefficient
# Minimum path length
# Small world (high clustering + low path lengths) -> by watts and strogatz (1998) collective dynamics of small
# world
# Cheeger constant

# Some pieces to consider:

# Weighted clustering coefficient: https://pmc.ncbi.nlm.nih.gov/articles/PMC374315/. Python equiv is
# nx.clustering(G, weight="weight")


