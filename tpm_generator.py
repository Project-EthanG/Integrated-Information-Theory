# The goal of this file is to generate a tpm given a fixed size, which will be used
# to generate the training data for the eventual neural network
import numpy as np
import random
import time

# Numpy has a built-in random function, but the random library is much faster

start_time = time.time()

def generate_random_det_tpm(dim):

    tpm = np.zeros((dim, dim))
    for row in range(tpm.shape[1]):
        random_col = random.randint(0, dim - 1)
        tpm[row][random_col] = 1
    return tpm

'''
We are looking to implement a function which has the TPM be randomly 
weighted by a linear-style network. Naturally we need the weights to be
independently drawn, then added to some bias. We can then classify the
state as changing or not changing depending on the range decided by the
weight
'''

def bias_generator(dim):
    # Generate biases for the TPM. One bias per row
    biases = np.random.rand(dim)
    return biases


def weight_generator(dim):
    # Generate weights for the tpm. One weight per cell in the tpm
    weights = np.random.rand(dim, dim)
    return weights


def argmax_ac_fn(tpm_linear):
    # Deterministic activation function, namely, the argmax
    max_indices = np.argmax(tpm_linear, axis=1)

    # Apply full weight to the maximized output for each row
    tpm_argmax = np.zeros((dim, dim))
    tpm_argmax[np.arange(dim), max_indices] = 1

    return tpm_argmax


def logistic_ac_fn(tpm_linear):
    # Probabilistic activation function: logistic
    probs = 1 / (1 + np.exp(tpm_linear))

    # Probabilities are not automatically normalized for the sigmoid fn
    tpm_sigmoid = probs / probs.sum(axis=1, keepdims=True)

    return tpm_sigmoid


def generate_lin_weighted_tpm(dim, seed):
    weights = weight_generator(dim)
    biases = bias_generator(dim)

    # Linear output for each cell. [:, newaxis] is needed to ensure
    # there is one bias added per row)
    tpm_linear = weights + biases[:, np.newaxis]

    # On its own this is a new TPM with no activation function.
    # If we want, we can have it use any number of activation functions...

    # Deterministic activation (argmax):
    # argmax_ac_fn(tpm_linear)

    # Probabilistic activation:

    # Logistic activation fn
    tpm_transformed = logistic_ac_fn(tpm_linear)

    # Hyperbolic Tangent activation fn

    return tpm_transformed

# Example usage
dim = 8
tpm = generate_lin_weighted_tpm(dim, seed = 50)
print("Weighted TPM:\n", tpm)