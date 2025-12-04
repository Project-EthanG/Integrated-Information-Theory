# The goal of this file is to generate a tpm given a fixed size, which will be used
# to generate the training data for the eventual neural network
import numpy as np
import random
import time

# Numpy has a built-in random function, but the random library is much faster

start_time = time.time()

def generate_det_tpm(dim, num_tpms):
    '''
    Generates a deterministic tpm of the fixed size

    :param dim: the size of the tpm. nxn
    :param n: the number of tpms to generate
    :return: tpm
    '''

    tpm = np.zeros((num_tpms, dim, dim))
    for i in range(num_tpms):
        tpm[i] = np.zeros((dim, dim))
        for row in range(tpm.shape[1]):
            random_col = random.randint(0, dim - 1)
            tpm[i][row][random_col] = 1


    return tpm

tpm_dim = 2**3
n_nodes = np.log2(tpm_dim)
n_tpms = 100
test1 = generate_det_tpm(tpm_dim, n_tpms)

end_time = time.time()

print(f"Generated {n_tpms} tpms of size {tpm_dim} ({int(n_nodes)} node network) in "
      f"{end_time - start_time} seconds)")