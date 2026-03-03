from tpm_generator import tpm_linear_generator, bias_generator, weight_generator
import iit_computation
import numpy as np
import numpy.typing as npt
from database import write_to_db, get_row_by_idx, close_db, drop_db, create_db, get_all_rows
from sklearn.model_selection import train_test_split
import ast
import torch
import torch.nn as nn
import torch.optim as optim


# Need to make sure the computation time is still tractable
import time

start_total = time.perf_counter()


def generate_toyset(n: int, num_tpms: int):
    # Monitor computational cost
    tpm_gen_start_time = time.perf_counter()

    # Stores ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes
    network_properties: list[tuple[float, float, tuple[list[int]] | None, float, int, npt.NDArray[np.float64]]] = []
    node_shape: tuple[int, int] = (n, n)
    biases = np.zeros((num_tpms, n), dtype=float)
    weights = np.zeros((num_tpms, *node_shape), dtype=float)

    # This contrasts the state-by-state TPMs and the corresponding entry for each state in the prior - dependent on N,
    # the number of states
    N: int = 2 ** n
    state_shape = (N, N)
    tpms_linear = np.zeros((num_tpms, *state_shape), dtype=float)
    priors: npt.NDArray[np.float64] = np.zeros((num_tpms, N), dtype=np.float64)

    for i in range(num_tpms):
        biases[i] = bias_generator(n)
        weights[i] = weight_generator(n)

        # Assume uniform prior
        priors[i] = [1 / N] * N

        tpms_linear[i] = tpm_linear_generator(n, biases[i], weights[i])

        # Append to list for db data entry
        ii, mi_Xt_Xtpast, max_bipartition, max_mi = iit_computation.integrated_information(tpms_linear[i], priors[i])
        network_properties.append((ii, mi_Xt_Xtpast, max_bipartition, max_mi, n, priors[i]))
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

drop_db()
create_db()


# Generate a toyset (let's just do 1000 6 node systems for now for testing computation time)
n = 6
num_tpms = 1000
generate_toyset(n, num_tpms)

# Flatten the data to feed into the NN
def flatten_predictors(row_slice):
    flat = []
    for val in row_slice:
        if isinstance(val, (list, tuple)):
            for v in val:
                # Flatten nested lists/tuples/arrays
                if isinstance(v, (list, tuple, np.ndarray)):
                    flat.extend(np.ravel(v).tolist())
                else:
                    flat.append(v)
        elif isinstance(val, np.ndarray):
            flat.extend(np.ravel(val).tolist())
        else:
            flat.append(val)
    return flat

rows = get_all_rows()

y = np.array([row[0] for row in rows], dtype=np.float32)
X = np.array([flatten_predictors(row[1:]) for row in rows], dtype=np.float32)

# Make the split. For now we will use 80-20-20
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=50
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=50
)

# Will use tensors for the NN. It needs to handle the data correctly

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Monitor computation time for training the NN
train_start_time = time.perf_counter()

# The neural network. Feed forward for now. The number of hidden layers is specified when making the SimpleFFNN object
class SimpleFFNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int = 1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = SimpleFFNN(input_dim=X.shape[1], hidden_dims=[32, 16], output_dim=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 75

for epoch in range(epochs):
    model.train()

    # Clear gradients on next BP
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    train_loss = criterion(y_pred, y_train_t)

    # Back propogate
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss.item():.6f} | "
              f"Val Loss: {val_loss.item():.6f}")

train_end_time: float = time.perf_counter()
total_train_time: float = train_end_time - train_start_time

print(f"\nFinished training the neural net! Time taken: {total_train_time:.2f} seconds.")
print("Beginning testing...:")

model.eval()
with torch.no_grad():
    test_pred = model(X_test_t)
    test_loss = criterion(test_pred, y_test_t)

print("Finished testing!\n")

print("Test MSE:", test_loss.item())


'''
# Analyze the DB (only for testing correct databasing)
for i in range(10):
    # Matrices are 0 indexed, dbs are 1 indexed
    print(get_row_by_idx(i+1))
'''

close_db()

end_total = time.perf_counter()
print(f"\nTotal runtime: {end_total - start_total:.4f} seconds")


