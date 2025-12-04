from tpm_generator import generate_det_tpm
import iit_computation

# Example usage for a 3 node system for random
tpm = generate_det_tpm(2**3)
prior = iit_computation.uniform_prior(tpm)

output = iit_computation.integrated_information(tpm, prior)
print("\nIntegrated information: ", output[0], "\n",
      "Mutual information across the network: ", output[1], "\n",
      "Least Damaging Partition: ", output[2], "\n",
      "Maximum Mutual Information across partitions:", output[3])


