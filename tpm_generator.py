# The goal of this file is to generate a tpm given a fixed size, which will be used
# to generate the training data for the eventual neural network
import numpy as np
from numpy.typing import NDArray
import random

test_seed: int = 50

def generate_random_det_tpm(dim):
    tpm = np.zeros((dim, dim))
    for row in range(tpm.shape[1]):
        random_col = random.randint(0, dim - 1)
        tpm[row][random_col] = 1
    return tpm


def bias_generator(n: int) -> NDArray[np.float64]:
    # Generate biases for the TPM. One bias per node (n nodes in the system). Sample i.i.d from Unif(-1,1)
    return np.random.uniform(-1, 1, n)


def weight_generator(n: int) -> NDArray[np.float64]:
    # Generate weights for the TPM. One weight influence per node. Nodes can have weights
    # on themselves as well, so, there are (n x n) weights. Sample i.i.d from Unif(-1, 1)
    return np.random.uniform(-1, 1, (n, n))


def logistic_scalar(p: float) -> float:
    # An activation function used to calculate the influence of a past state on a specific
    # node's present state
    return 1 / (1 + np.exp(-p))


def tpm_linear_generator(n: int, biases: NDArray[np.float64], weights: NDArray[np.float64], temp: float = 1) -> NDArray[np.float64]:
    # Generates a tpm based on pre-defined biases for each node and weights for each node's influence

    # Num states when there are n nodes
    N = 2**n

    # Empty state-by-state TPM (N x N), where N = 2**n
    tpm_linearly_generated: NDArray[np.float64] = np.zeros((N, N))

    # Start with manual looping through the state-by-state TPM. I will look to optimize this later
    rows, cols = tpm_linearly_generated.shape

    for row in range(rows):
        # Store the state as a vector. Needed to compute the linear term
        past_state: NDArray[np.float64] = np.array([(row >> (n - 1 - i)) & 1 for i in range(n)])
        prob_on: NDArray[np.float64] = np.zeros(n)
        for i in range(n):
            # Linear output is based on biases and weights. Serves as an input for the activation fn
            linear_output: float = biases[i] + (1 / temp) * np.dot(weights[:, i], past_state)
            prob_on[i] = logistic_scalar(linear_output)

        for col in range(cols):
            # Store present state as vector depending on indices (assumed lexicographic).
            # Necessary for determining if the sigmoid function output needs to be
            # taken as a complement (i.e; the output node for the present state is OFF rather than ON)
            present_state: NDArray[np.float64] = np.array([(col >> (n - 1 - i)) & 1 for i in range(n)])

            prob_joint = 1.0
            for i in range(n):
                # If the present state's node is 1, use the probability as per usual
                if present_state[i] == 1:
                    prob_joint *= prob_on[i]
                # Otherwise the complement must be used
                else:
                    prob_joint *= (1 - prob_on[i])

            tpm_linearly_generated[row, col] = prob_joint
    return tpm_linearly_generated