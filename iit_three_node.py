from tpm_generator import generate_det_tpm
import iit_computation
import numpy as np
from database import write_to_db, get_row_by_idx, close_db, drop_db, create_db


# Example usage for multiple 4 node systems
num_nodes: int = 4
num_tpms = 20
n_states = 2**num_nodes
shape = (n_states, n_states)

tpm = np.zeros((num_tpms, *shape), dtype=int)
prior = np.zeros((num_tpms, n_states), dtype=float)

# Stores ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes
network_properties: list[tuple[float, float, tuple[list[int]] | None, float, int]] = []
'''
for i in range(num_tpms):
    tpm[i] = generate_det_tpm(2**4)
    prior[i] = iit_computation.uniform_prior(tpm[i])

    # Append to list for db data entry
    ii, mi_Xt_Xtpast, max_bipartition, max_mi = iit_computation.integrated_information(tpm[i], prior[i])
    network_properties.append((ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes))
'''
drop_db()
create_db()


# Simple tractable system example
num_nodes: int = 4


tpm_simple = np.zeros((16,16))

for i in range(16):
    tpm_simple[i, 15] = 1

tpm_simple[0, 15] = 0
tpm_simple[0, 0] = 1

prior_simple = iit_computation.uniform_prior(tpm_simple)

ii, mi_Xt_Xtpast, max_bipartition, max_mi = iit_computation.integrated_information(tpm_simple, prior_simple)
network_properties.append((ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes))

write_to_db(network_properties)

# Example row
print(get_row_by_idx(1))

close_db()


