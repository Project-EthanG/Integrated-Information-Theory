from tpm_generator import generate_det_tpm
import iit_computation
import numpy as np

# Example usage for a 3 node systems with uni prior and random tpms
tpm = generate_det_tpm(2**3)
prior = iit_computation.uniform_prior(tpm)

output = iit_computation.integrated_information(tpm, prior)
print("\nIntegrated information: ", output[0], "\n",
      "Mutual information across the network: ", output[1], "\n",
      "Least Damaging Partition: ", output[2], "\n",
      "Maximum Mutual Information across partitions:", output[3])

# Example usage for multiple 3 node systems
num_tpms = 3
n_states = 2**3
shape = (n_states, n_states)

tpm = np.zeros((num_tpms, *shape), dtype=int)
prior = np.zeros((num_tpms, n_states), dtype=float)

for i in range(num_tpms):
    tpm[i] = generate_det_tpm(2**3)
    prior[i] = iit_computation.uniform_prior(tpm[i])

    # Compute and write to file
    iit_computation.integrated_information(tpm[i], prior[i])


