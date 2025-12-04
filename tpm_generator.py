# The goal of this file is to generate a tpm given a fixed size, which will be used
# to generate the training data for the eventual neural network
import numpy as np
import random
import time

# Numpy has a built-in random function, but the random library is much faster

start_time = time.time()

def generate_det_tpm(dim):
    '''
    Generates a deterministic tpm of the fixed size

    :param dim: the size of the tpm. nxn
    :param n: the number of tpms to generate
    :return: tpm
    '''

    tpm = np.zeros((dim, dim))
    for row in range(tpm.shape[1]):
        random_col = random.randint(0, dim - 1)
        tpm[row][random_col] = 1
    return tpm

tpm = generate_det_tpm(8)
print(tpm)