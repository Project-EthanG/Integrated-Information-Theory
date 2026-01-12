from tpm_generator import generate_det_tpm
import iit_computation
import numpy as np
from database import write_to_db, get_row_by_idx, close_db, drop_db, create_db

# Example usage for a 3 node systems with uni prior and random tpms
num_nodes: int = 3
tpm = generate_det_tpm(2**num_nodes)
prior = iit_computation.uniform_prior(tpm)

output = iit_computation.integrated_information(tpm, prior)
print("\nIntegrated information: ", output[0], "\n",
      "Mutual information across the network: ", output[1], "\n",
      "Least Damaging Partition: ", output[2], "\n",
      "Maximum Mutual Information across partitions:", output[3])

# Example usage for multiple 3 node systems
num_tpms = 20
n_states = 2**4
shape = (n_states, n_states)

tpm = np.zeros((num_tpms, *shape), dtype=int)
prior = np.zeros((num_tpms, n_states), dtype=float)

# Stores ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes
network_properties: list[tuple[float, float, tuple[list[int]] | None, float, int]] = []

for i in range(num_tpms):
    tpm[i] = generate_det_tpm(2**4)
    prior[i] = iit_computation.uniform_prior(tpm[i])

    # Append to list for db data entry
    ii, mi_Xt_Xtpast, max_bipartition, max_mi = iit_computation.integrated_information(tpm[i], prior[i])
    network_properties.append((ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes))

drop_db()
create_db()

write_to_db(network_properties)

# Example row
print(get_row_by_idx(20))

close_db()
