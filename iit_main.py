from tpm_generator import tpm_linear_generator, bias_generator, weight_generator, generate_random_det_tpm
import iit_computation
import numpy as np
import numpy.typing as npt
from database import write_to_db, get_row_by_idx, close_db, drop_db, create_db, get_all_rows
from sklearn.model_selection import train_test_split
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader

test_seed: int = 50

start_total = time.perf_counter()


def generate_toyset(n: int, num_tpms: int):
    # Monitor computational cost
    tpm_gen_start_time = time.perf_counter()

    # Stores ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes, prior
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
        network_properties.append((tpms_linear[i], ii, mi_Xt_Xtpast, max_bipartition, max_mi, n, priors[i]))
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
n: int = 6
num_tpms: int = 100
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

# Stores all the db entries. For code sanitation this intermediary array should not be necessary.
# FUTURE FEATURE IMPLEMENTATION REQUIRED
rows: list[tuple[float,float,tuple[list[int]] | None,float,int,npt.NDArray[np.float64]]] = get_all_rows()

# The current design matrix considers the following parameters (all but ii in db):
#   I(Xt;Xt-1)
#   MIP
#   I*(Xt;Xt-1) imposed by MIP
#   num_nodes
#   tpm_prior

# However in reality, the goal is to reduce computation, so prediction should occur independent
# of any computationally taxing calculation. This means MIP and I*(Xt; Xt-1) are not realistic features

# Iterating through the db as indices for now, but this should be specified by feature name. FUTURE FEATURE IMPLEMENTATION REQUIRED
feature_cols = [0, 2, 5, 6]
y = np.array([row[1] for row in rows], dtype=np.float32)
X = np.array([flatten_predictors([row[i] for i in feature_cols]) for row in rows], dtype=np.float32)


# Make the split. For now we will use 40-30-30
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.6, random_state=test_seed
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=test_seed
)

# Will use tensors for the NN

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Monitor computation time for training the NN
train_start_time = time.perf_counter()

rows = get_all_rows()
for i, row in enumerate(rows[:3]):
    tpm, ii, mi, max_bipartition, max_mi, num_nodes, tpm_prior = row
    print(f"Row {i}: tpm.shape={tpm.shape}, tpm_prior.shape={tpm_prior.shape}, num_nodes={num_nodes}")
print(f"X_train_t.shape: {X_train_t.shape}, y_train_t.shape: {y_train_t.shape}")
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)




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

epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss_accum = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / len(train_loader)

    # Validation (full pass, no batching needed)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {avg_train_loss:.6f} | "
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

close_db()

end_total = time.perf_counter()
print(f"\nTotal runtime: {end_total - start_total:.4f} seconds")


# feature: add TPM as a feature for the FFNN, update in main and databasing.


