"""
CBOW embeddings from scratch (NumPy only) where each "sentence" is a 2D array
and each ROW of that array is a "token" (a vector of floats in [0, 1]).

Pipeline
--------
0. (Optional) Bin: snap each entry to the midpoint of its bin (e.g. width 0.1).
1. Tokenize: identical rows are assigned the same integer token ID
   (IDs are shared within and across sentences).
2. Train: for a sentence with T rows, perform T prediction tasks. Each row is
   masked in turn and predicted from the remaining T-1 rows (via their IDs)
   with a CBOW network, updating by backprop/SGD after every task.
3. Return the full (V, d) embedding matrix, V = number of unique row vectors.

For a sentence of shape (2^n, n), T = 2^n, so 2^n prediction tasks are run.
"""

import numpy as np


# --------------------------------------------------------------------------
# Binning: continuous entries -> bin midpoints
# --------------------------------------------------------------------------
def bin_vectors(array, bin_width=0.1):
    """
    Snap every entry of an array (values in [0, 1]) to the midpoint of its bin.

    Bins are [0, w), [w, 2w), ..., with the last bin closed on the right so
    that 1.0 falls in the final bin. For w = 0.1, an entry of 0.02 -> 0.05,
    0.7888 -> 0.75 and 0.48 -> 0.45.
    """
    array = np.asarray(array, dtype=np.float64)
    tol = 1e-9
    if array.size and (array.min() < -tol or array.max() > 1 + tol):
        raise ValueError("Entries must lie in [0, 1] before binning; "
                         f"got range [{array.min()}, {array.max()}]")
    n_bins = int(round(1.0 / bin_width))
    # round() before floor() so values like 0.3 (stored as 0.29999...) are not
    # pushed into the wrong bin by floating-point error
    bin_idx = np.floor(np.round(array / bin_width, 9)).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    # round() again so identical bins give bit-identical floats (needed for
    # exact row matching in the tokenizer)
    return np.round((bin_idx + 0.5) * bin_width, 10)


def bin_corpus(corpus, bin_width=0.1):
    """Apply `bin_vectors` to every sentence array in the corpus."""
    return [bin_vectors(sentence, bin_width) for sentence in corpus]


# --------------------------------------------------------------------------
# Tokenization: row vector -> integer ID
# --------------------------------------------------------------------------
def _row_key(row, decimals=None):
    """Hashable key for a row vector. Identical rows give identical keys."""
    row = np.asarray(row, dtype=np.float64)
    if decimals is not None:
        row = np.round(row, decimals)
    return row.tobytes()


def build_vocab(corpus, decimals=None):
    """
    Assign an integer ID to every unique row vector found in the corpus.

    Parameters
    ----------
    corpus : list[np.ndarray]
        Each array is one "sentence" of shape (num_tokens, vector_dim).
    decimals : int or None
        If given, rows are rounded to this many decimals before comparison,
        guarding against floating-point noise. None means exact matching.

    Returns
    -------
    row_to_idx : dict[bytes, int]
    idx_to_vector : list[np.ndarray]   (idx_to_vector[i] is the vector for ID i)
    """
    row_to_idx = {}
    idx_to_vector = []
    for sentence in corpus:
        sentence = np.asarray(sentence, dtype=np.float64)
        if sentence.ndim != 2:
            raise ValueError(f"Each sentence must be 2D; got shape {sentence.shape}")
        for row in sentence:
            key = _row_key(row, decimals)
            if key not in row_to_idx:
                row_to_idx[key] = len(idx_to_vector)
                idx_to_vector.append(row.copy())
    return row_to_idx, idx_to_vector


def encode_sentence(sentence, row_to_idx, decimals=None):
    """Convert one (num_tokens, vector_dim) array into an array of token IDs."""
    sentence = np.asarray(sentence, dtype=np.float64)
    return np.array([row_to_idx[_row_key(r, decimals)] for r in sentence],
                    dtype=np.int64)


def encode_corpus(corpus, row_to_idx, decimals=None):
    """Convert a list of sentence arrays into a list of token-ID arrays."""
    return [encode_sentence(s, row_to_idx, decimals) for s in corpus]


# --------------------------------------------------------------------------
# CBOW training
# --------------------------------------------------------------------------
def _softmax(scores):
    scores = scores - scores.max()  # numerical stability
    exp = np.exp(scores)
    return exp / exp.sum()


def train_cbow(corpus, d=50, learning_rate=0.05, epochs=1, shuffle=True,
               seed=0, decimals=None, bin_width=None, verbose=False):
    """
    Train CBOW embeddings where tokens are the rows of each sentence array.

    Parameters
    ----------
    corpus : list[np.ndarray]
        Each array has shape (num_tokens, vector_dim), e.g. (2^n, n), with
        float entries in [0, 1]. Repeated rows (within or across sentences)
        are treated as the same token.
    d : int
        Embedding dimensionality.
    learning_rate : float
        SGD step size, applied after every prediction task.
    epochs : int
        Number of passes over the corpus.
    shuffle : bool
        Shuffle sentence order each epoch.
    seed : int
        Random seed.
    decimals : int or None
        Optional rounding used when deciding whether two rows are identical.
    bin_width : float or None
        If given (e.g. 0.1), every entry of every vector is first snapped to
        the midpoint of its bin of this width. Rows that land in the same bins
        then become the same token. None leaves the values untouched.
    verbose : bool
        Print average loss per epoch.

    Returns
    -------
    np.ndarray of shape (V, d)
        Row i is the embedding of the unique row vector with token ID i
        (see `build_vocab` for the ID -> vector mapping).
    """
    rng = np.random.default_rng(seed)

    # Step 0: Discretize continuous entries into bin midpoints
    if bin_width is not None:
        corpus = bin_corpus(corpus, bin_width)
        print(corpus)

    # Step 1: extract token IDs
    row_to_idx, idx_to_vector = build_vocab(corpus, decimals)
    sentences = encode_corpus(corpus, row_to_idx, decimals)
    V = len(idx_to_vector)

    # Step 2: CBOW with full softmax
    W_in = (rng.random((V, d)) - 0.5) / d   # embedding matrix (returned)
    W_out = np.zeros((V, d))                # output weights (discarded)

    for epoch in range(epochs):
        order = rng.permutation(len(sentences)) if shuffle else range(len(sentences))
        total_loss, n_tasks = 0.0, 0

        for s_idx in order:
            sent = sentences[s_idx]
            T = len(sent)
            if T < 2:
                continue  # no context to predict from

            # One prediction task per token (row) in the sentence
            for pos in range(T):
                target = sent[pos]
                context = np.delete(sent, pos)   # IDs of all remaining rows

                # forward
                h = W_in[context].mean(axis=0)           # (d,)
                probs = _softmax(W_out @ h)              # (V,)
                total_loss += -np.log(probs[target] + 1e-12)
                n_tasks += 1

                # backward
                dscores = probs.copy()
                dscores[target] -= 1.0                   # (V,)
                grad_W_out = np.outer(dscores, h)        # (V, d)
                grad_h = W_out.T @ dscores               # (d,) pre-update W_out

                # SGD updates (np.add.at handles repeated IDs in the context)
                W_out -= learning_rate * grad_W_out
                np.add.at(W_in, context, -learning_rate * grad_h / len(context))

        if verbose and n_tasks:
            print(f"epoch {epoch + 1}/{epochs}  avg loss: {total_loss / n_tasks:.4f}")

    return W_in


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # A small pool of distinct vectors in [0, 1]^3 that will repeat everywhere
    pool = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.25, 0.75, 0.1], [0.9, 0.2, 0.6],
    ])

    n = 3
    # 20 "sentences", each of shape (2^n, n) = (8, 3), rows sampled from the pool
    corpus = [pool[rng.integers(0, len(pool), size=2 ** n)] for _ in range(20)]

    row_to_idx, idx_to_vector = build_vocab(corpus)
    print("Sentence shape:", corpus[0].shape)
    print("Unique token vectors found:", len(idx_to_vector))
    print("IDs of first sentence:", encode_sentence(corpus[0], row_to_idx))

    d = 10
    embeddings = train_cbow(corpus, d=d, learning_rate=0.1, epochs=50, verbose=False, bin_width = 0.1)
    print("Embedding matrix shape:", embeddings.shape)   # (V, d)
    print(embeddings)