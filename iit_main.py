from tpm_generator import tpm_linear_generator, bias_generator, weight_generator
import iit_computation
import numpy as np
from database import write_to_db, get_row_by_idx, close_db, drop_db, create_db, get_all_rows
from sklearn.model_selection import train_test_split


# Need to make sure the computation time is still tractable
import time

start_total = time.perf_counter()


def generate_toyset(n: int, num_tpms: int):
    # Monitor computational cost
    tpm_gen_start_time = time.perf_counter()

    # Stores ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes
    network_properties: list[tuple[float, float, tuple[list[int]] | None, float, int]] = []
    node_shape = (n, n)
    biases = np.zeros((num_tpms, n), dtype=float)
    weights = np.zeros((num_tpms, *node_shape), dtype=float)

    # This contrasts the state-by-state TPMs and the corresponding entry for each state in the prior - dependent on N,
    # the number of states
    N: int = 2 ** n
    state_shape = (N, N)
    tpms_linear = np.zeros((num_tpms, *state_shape), dtype=float)
    priors = np.zeros((num_tpms, N), dtype=float)

    for i in range(num_tpms):
        biases[i] = bias_generator(n)
        weights[i] = weight_generator(n)
        priors[i] = [1 / N] * N

        tpms_linear[i] = tpm_linear_generator(n, biases[i], weights[i])

        # Append to list for db data entry
        ii, mi_Xt_Xtpast, max_bipartition, max_mi = iit_computation.integrated_information(tpms_linear[i], priors[i])
        network_properties.append((ii, mi_Xt_Xtpast, max_bipartition, max_mi, n))
        print(
            f"Finished generating TPM number {i + 1} and computing its integrated information. Writing to database...")

    write_to_db(network_properties)

    tpm_gen_end_time: float = time.perf_counter()

    total_tpm_time: float = tpm_gen_end_time - tpm_gen_start_time

    # Log the time taken
    if total_tpm_time > 60:
        print(f"\nComplete! Finished processing {num_tpms} tpms in {int(total_tpm_time / 60)} "
              f"minutes and {total_tpm_time % 60:.4f} seconds")
    else:
        print(f"\nComplete! Finished processing {num_tpms} tpms in {total_tpm_time} seconds")

# For now assume we only use the previously generated dataset. UNCOMMENT when generating a new set
'''
drop_db()
create_db()


# Generate a toyset (let's just do 1000 6 node systems for now for testing computation time)
n = 5
num_tpms = 1000
generate_toyset(n, num_tpms)
'''


# Create training validation testing split
rows = get_all_rows()

# The data still holds the primary key from the DB so we need to get rid of it in the split
full_data = np.array(rows, dtype=object)
y = rows[:, 1].astype(float).reshape(-1,1)
X = rows[:, 2:]
n = len(X)

# Make the split. For now we will use 80-20-20
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=50
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=50
)

'''
# Analyze the DB (only for testing correct databasing)
for i in range(10):
    # Matrices are 0 indexed, dbs are 1 indexed
    print(get_row_by_idx(i+1))
'''


##########################

# Next goal: training a feed forward neural network using the stored values as inputs for fixed size TPMs

# General guidelines involve:
# 1) Creating a training, testing and validation set for cross-validation
# 2) Proper analysis of the data. For example, ensuring a partition [1] [0,2] is properly interpreted (0 has no relation
#    to 1, they are just placeholders for strings)
# 3) Ensuring backpropogation is not computationally intractable
# 4) Check the differences between classification and regression. How well can the system perform for non-integrated
#    systems when most "randomly" generated systems are unlikely to have absolutely no integration?


##########################

close_db()

end_total = time.perf_counter()
print(f"\nTotal runtime: {end_total - start_total:.4f} seconds")


