from tpm_generator import tpm_linear_generator, bias_generator, weight_generator, tpm_linear_generator_split
import iit_computation
import numpy as np
import numpy.typing as npt
from database import write_to_db, close_db, drop_db, create_db, get_all_rows
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
import copy


test_seed: int = 50

start_total = time.perf_counter()

# The neural network. Feed forward for now. The number of hidden layers is specified when making the SimpleFFNN object
class SimpleFFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        num_hidden = len(hidden_dims)

        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())

            # Taper dropout in the final 2 hidden layers, none after last
            if i < num_hidden - 2:
                rate = dropout_rate
            elif i < num_hidden - 1:
                rate = dropout_rate / 2
            else:
                rate = 0.0

            if rate > 0:
                layers.append(nn.Dropout(rate))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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

    rng = np.random.default_rng(seed=test_seed)

    for i in range(num_tpms):
        biases[i] = bias_generator(n)
        weights[i] = weight_generator(n)

        # Assume uniform prior
        priors[i] = [1 / N] * N

        tpms_linear[i] = tpm_linear_generator_split(n, biases[i], weights[i], temp=1, p=0.1, rng=rng)

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
        print(f"\nComplete! Finished processing {num_tpms} tpms in {total_tpm_time:.4f} seconds")


def gen_and_write_to_db(n: int = 4, num_tpms: int = 100) -> None:
    # Generate a toyset for training and testing. Compute integrated information directly, and store all necessary
    # quantities in the database.

    # n: number of nodes in the system
    # num_tpms: number of tpms to generate

    drop_db()
    create_db()

    generate_toyset(n, num_tpms)


# Flatten the data to feed into the NN. This version, to be updated, simply flattens any multidimensional input to a
# vector.

# Optimized flatten predictors algorithm
def flatten_predictors(row_slice):
    def _iter_flat(val):
        if isinstance(val, np.ndarray):
            yield from val.ravel()
        elif isinstance(val, (list, tuple)):
            for v in val:
                yield from _iter_flat(v)
        else:
            yield val

    return list(_iter_flat(row_slice))

# COMMENT if the dataset already exists. UNCOMMENT if we need to generate a new dataset
#gen_and_write_to_db(n=6, num_tpms=20_000)


# Stores all the db entries. For code sanitation this intermediary array should not be necessary.
# FUTURE FEATURE IMPLEMENTATION REQUIRED
rows: list[tuple[float,float,tuple[list[int]] | None,float,int,npt.NDArray[np.float64]]] = get_all_rows()

def define_features(feature_cols: list[int]):
    y = np.array([row[1] for row in rows], dtype=np.float32)
    X = np.array([flatten_predictors([row[i] for i in feature_cols]) for row in rows], dtype=np.float32)
    return X, y


def fit_FNN(X, y, prop_train: float = 0.4, prop_test: float = 0.3, prop_val: float = 0.3):

    if prop_train+prop_test+prop_val != 1:
        raise ValueError("Invalid train-test-validation split")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1-prop_train, random_state=test_seed
    )

    relative_test_size = prop_test / (prop_test + prop_val)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=test_seed
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print("Beginning training the neural net...\n")

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleFFNN(input_dim=X.shape[1], hidden_dims=[32, 16], output_dim=1, dropout_rate=0.2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=0.001)

    best_val_loss = float('inf')
    best_model_weights = None
    patience = 10
    epochs_no_improve = 0
    val_threshold = 1e-8
    max_epochs = 500

    # Learning Rate Scheduling for reducing learning rate as validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4
    )

    for epoch in range(max_epochs):
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

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            scheduler.step(val_loss)

        if val_loss < best_val_loss - val_threshold:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best model saved at epoch {epoch}")
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4e} | Val Loss: {val_loss:.4e}")

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} — no improvement for {patience} consecutive epochs.")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\nRestored best model weights (val_loss={best_val_loss:.4e})")

    # Compute test loss against the restored best weights
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t)

    print("\nTest MSE:", f"{test_loss.item():.4e}")

    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)

    # Per-sample squared errors for comparing models
    nn_sq_errors = (y_test.flatten() - test_pred.numpy().flatten())**2
    baseline_sq_errors = (y_test.flatten() - baseline_pred.flatten())**2

    # Compare errors across all observations
    error_diffs = baseline_sq_errors - nn_sq_errors
    mean_diff = np.mean(error_diffs)
    sd_diff = np.std(error_diffs, ddof=1)

    if sd_diff > 0:
        cohens_d = mean_diff / sd_diff
    else:
        cohens_d = np.inf

    print(f"Baseline MSE: {np.mean(baseline_sq_errors):.4e}")
    print(f"Cohen's d (paired): {cohens_d:.4f}")

# The current indices are:

# 0: TPM
# 1: phi value (measure of integrated information)
# 2: mutual information for the full system
# 3: MIP
# 4: mutual information imposed by the MIP
# 5: number of units in the system
# 6: prior

# Let's compare how the FFNN performs when fed just the TPM vs. just the mutual information for the full system. Number
# of units and prior are currently fixed, so they do not have any influence as predictors

# Model 1: TPM, prior and no. nodes
print(f"Defining features...")
X_raw, y_raw = define_features([0, 5, 6])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

# Model 2: MI, prior and no. nodes
print(f"Defining features...")
X_raw, y_raw = define_features([2, 5, 6])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

# Model 3: TPM, MI, prior and no. nodes
print(f"Defining features...")
X_raw, y_raw = define_features([0, 2, 5, 6])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

# Model 4: TPM
print(f"Defining features...")
X_raw, y_raw = define_features([0])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

# Model 5: MI
print(f"Defining features...")
X_raw, y_raw = define_features([2])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)


'''
Current results:

Model 1: TPM + no. units + prior
Model 2: MI + no. units + prior
Model 3: TPM + MI + no. units + prior
Model 4: TPM
Model 5: MI

Batch 1 (n=1_000 but no validation. Train for 100 epochs no matter what)

Train MSE:
Model 1: 0.000132
Model 2: 0.000208
Model 3: 0.000129
Model 4: 0.000067
Model 5: 0.000198

Test MSE:
Model 1: 0.0004475
Model 2: 0.0002220
Model 3: 0.0003407
Model 4: 0.0004980
Model 5: 0.0001864


Batch 2 (n=20_000 but no validation. Train for 100 epochs no matter what)

Train MSE:
Model 1: 0.000388
Model 2: 0.000217
Model 3: 0.000246
Model 4: 0.000389
Model 5: 0.000214

Test MSE:
Model 1: 0.000386078
Model 2: 0.000209664
Model 3: 0.000238302
Model 4: 0.000384595
Model 5: 0.000208751


Batch 3 (n=20_000 and validation. Stop training after 10 epochs of little improvement)

Train MSE: 
Model 1: 0.00040090
Model 2: 0.00022285
Model 3: 0.00026617
Model 4: 0.00110660 
Model 5: 0.00021159 

Test MSE:
Model 1: 0.00038639
Model 2: 0.00021075+
Model 3: 0.00025522
Model 4: 0.00040049
Model 5: 0.00020711

General observations:
-> Increasing observations seemed to help
-> Validation process is certainly causing further overfitting
-> Not having MI in the feature space adds quite a bit of MSE
-> Shrinkage of prior and no. nodes is not being sufficiently shrunk - some MSE leakage

Considerations for next week:
	- Validation should also take into consideration shrinkage in some way. Maybe we shrink parameters relative to the amount of training error loss compared to validation loss?
	- Computation time for training took longer than expected. Maybe possible to optimize the predictor flattening algorithm?
	- Consider looking at breaking TPM into intermediary qtys or map to a high-dimensional space
	- "Patience" threshold needs to be changed to make sure epochs aren't being cut off too early
	- Generation is restricted to linearly weighted systems - extremely unlikely to get 0 integrated information systems. Maybe we make ~10% no integration and check performance?
	- Generation can still be optimized - this is the least priority since we already have a dataset of 20_000 systems, but still something to consider as we start working with bigger networks

Notes:
    - Regularization exists already, but I have implemented a dropout rate for the hidden dimensions in the FFNN
    - Predictor flattening algorithm optimized
    - Training the NN is a bit slow, this one might be tough to optimize, will look back.
    - val_loss relative to patience was decreased to allow for new updates on epochs.
    - Adding 10% non-integrated systems will require a bit of time since that involves running the simulation again...
    - Intermediary qtys requires more research. Will have this done for next week.
    - Not worried about tpm generation at the moment.

Koen's D value for effect size measuring change in error rate divided by standard deviation errors

'''

close_db()

end_total = time.perf_counter()
print(f"\nTotal runtime: {end_total - start_total:.4f} seconds")

# FINISHED: Validate regularization penalty, effect size for model comparison (Cohen's d value)
# TO DO: meaningful features to derive from the TPM. Some ideas:
#   average row entropy, effect information, system degeneracy, spectral gap (using something
#   like Lanczos algorithm to find the eigenvals), entropy of the stationary distribution,
#   pairwise information residuals.