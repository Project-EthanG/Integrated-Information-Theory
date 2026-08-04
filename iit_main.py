from tpm_generator import bias_generator, weight_generator, tpm_linear_generator_split
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
from feature_generator import compute_nbn_features
from sklearn.preprocessing import StandardScaler


test_seed: int = 50

start_total = time.perf_counter()

# Scalar features
NBN_FEATURE_NAMES: list[str] = [
    "mixing_gap", "spectral_entropy", "wd", "wr",
    "weight_cluster_coeff", "short_path_len", "small_world_coeff", "cheeger_coeff",
    "num_sccs", "max_scc", "diam", "avg_closeness",
    "avg_betweenness", "max_pr", "min_pr", "mean_pr"
]

# Features that requre specific handling (i.e; serialization for network feeding)
STRUCTURAL_KEYS = {"tpm", "tpm_prior", "max_bipartition"}


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

    print("Beginning network generation...")

    tpm_gen_start_time = time.perf_counter()

    # Each entry is now a dict: {"tpm", "tpm_prior", "max_bipartition", "features": {...}}
    network_properties: list[dict] = []
    node_shape: tuple[int, int] = (n, n)
    biases = np.zeros((num_tpms, n), dtype=float)
    weights = np.zeros((num_tpms, *node_shape), dtype=float)

    N: int = 2 ** n
    state_shape = (N, N)
    tpms_linear = np.zeros((num_tpms, *state_shape), dtype=float)
    priors: npt.NDArray[np.float64] = np.zeros((num_tpms, N), dtype=np.float64)

    rng = np.random.default_rng(seed=test_seed)

    for i in range(num_tpms):
        biases[i] = bias_generator(n)
        weights[i] = weight_generator(n)

        priors[i] = [1 / N] * N

        tpms_linear[i] = tpm_linear_generator_split(n, biases[i], weights[i], temp=1, p=0.1, rng=rng)

        ii, mi_Xt_Xtpast, max_bipartition, max_mi = iit_computation.integrated_information(tpms_linear[i], priors[i])

        nbn_features: list = compute_nbn_features(tpms_linear[i])

        # Scalar features (baseline + node-by-node) all collapse into one dict.
        # Adding a new feature anywhere upstream (iit_computation or compute_nbn_features)
        # only requires adding its name/value pair here -- nothing else below changes.
        features = {
            "ii": ii,
            "mi": mi_Xt_Xtpast,
            "max_mi": max_mi,
            "num_nodes": n,
            **dict(zip(NBN_FEATURE_NAMES, nbn_features)),
        }

        network_properties.append({
            "tpm": tpms_linear[i],
            "tpm_prior": priors[i],
            "max_bipartition": max_bipartition,
            "features": features,
        })

        print(f"Finished generating TPM number {i + 1} and computing its integrated information. Attributes have been computed. Next TPM...")

    print(f"Writing to database...")
    write_to_db(network_properties)

    tpm_gen_end_time: float = time.perf_counter()
    total_tpm_time: float = tpm_gen_end_time - tpm_gen_start_time

    if total_tpm_time > 60:
        print(f"\nComplete! Finished processing {num_tpms} tpms in {int(total_tpm_time / 60)} "
              f"minutes and {total_tpm_time % 60:.4f} seconds")
    else:
        print(f"\nComplete! Finished processing {num_tpms} tpms in {total_tpm_time:.4f} seconds")


def gen_and_write_to_db(n: int = 4, num_tpms: int = 100) -> None:
    drop_db()
    create_db()
    generate_toyset(n, num_tpms)


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

rows = get_all_rows()


def _get(row: dict, name: str):
    if name in STRUCTURAL_KEYS:
        return row[name]
    return row["features"][name]


def define_features(feature_names: list[str], target = "ii"):
    y = np.array([_get(row, target) for row in rows], dtype=np.float32)
    X = np.array(
        [flatten_predictors([_get(row, name) for name in feature_names]) for row in rows],
        dtype=np.float32,
    )
    return X, y


# For deciding how many hidden layers and the dim for each layer
def suggest_architecture(n_features: int, n_samples: int) -> list[int]:
    # Rough funnel: first hidden layer close to input width, then taper
    base = max(8, min(256, 2 * n_features))

    # Want ~10-30 samples per parameter, roughly
    max_params_budget = n_samples * 5

    dims = [base, max(8, base // 2)]

    # Crude param count check for a 2-layer MLP
    def param_count(dims, in_dim):
        prev = in_dim
        total = 0
        for d in dims:
            total += prev * d + d
            prev = d
        total += prev * 1 + 1
        return total

    while param_count(dims, n_features) > max_params_budget and dims[0] > 8:
        dims = [max(8, d // 2) for d in dims]

    return dims


def fit_FNN(X, y, prop_train: float = 0.4, prop_test: float = 0.3, prop_val: float = 0.3) -> float:

    if prop_train + prop_test + prop_val != 1:
        raise ValueError("Invalid train-test-validation split")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - prop_train, random_state=test_seed
    )

    relative_test_size = prop_test / (prop_test + prop_val)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=test_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print("Beginning training the neural net...\n")

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    hidden_dims = suggest_architecture(n_features=X.shape[1], n_samples=X_train.shape[0])
    model = SimpleFFNN(input_dim=X.shape[1], hidden_dims=hidden_dims, output_dim=1, dropout_rate=0.2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=0.001)

    best_val_loss = float('inf')
    best_model_weights = None
    epoch_patience = 10
    epochs_no_improve = 0
    val_threshold = 0
    max_epochs = 500

    # Threshold is determined relative to change in loss (i.e; more than 1% improvement from before?)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        threshold=1e-3,
        threshold_mode="rel",
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
            if epochs_no_improve >= epoch_patience:
                print(f"\nEarly stopping at epoch {epoch} — no improvement for {epoch_patience} consecutive epochs.")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\nRestored best model weights (val_loss={best_val_loss:.4e})")

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t)

    print("\nTest MSE:", f"{test_loss.item():.4e}")


    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)

    nn_sq_errors = (y_test.flatten() - test_pred.numpy().flatten()) ** 2
    baseline_sq_errors = (y_test.flatten() - baseline_pred.flatten()) ** 2

    error_diffs = baseline_sq_errors - nn_sq_errors
    mean_diff = np.mean(error_diffs)
    sd_diff = np.std(error_diffs, ddof=1)

    cohens_d: float = mean_diff / sd_diff if sd_diff > 0 else np.inf

    # TEMP:
    print("baseline value:", np.mean(y_train))
    print(f"Cohen's d (paired): {cohens_d:.4f} \n")

    return cohens_d


ALL_FEATURE_NAMES = ["tpm"] + list(_get.__wrapped__ if False else [])  # placeholder removed below

# Make sure our predicting factor is not part of the feature space
TARGET = "ii"

all_feature_names = [k for k in rows[0]["features"].keys() if k != TARGET]

'''

print(f"Defining features for model 1...")
X_raw, y_raw = define_features(["tpm", "num_nodes", "tpm_prior"])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

print(f"Defining features for model 2...")
X_raw, y_raw = define_features(["mi", "num_nodes", "tpm_prior"])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

print(f"Defining features for model 3...")
X_raw, y_raw = define_features(["tpm", "mi", "num_nodes", "tpm_prior"])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

print(f"Defining features for model 4...")
X_raw, y_raw = define_features(["tpm"])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

print(f"Defining features for model 5...")
X_raw, y_raw = define_features(["mi"])
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)


print(f"Defining all features for model 6...")
X_raw, y_raw = define_features(["tpm"] + all_feature_names, target=TARGET)
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

# TEMP:
print("y_raw mean/std:", y_raw.mean(), y_raw.std())


print(f"Defining all features for model 7...")
X_raw, y_raw = define_features(all_feature_names, target=TARGET)
fit_FNN(X_raw, y_raw, 0.4, 0.3, 0.3)

# TEMP:
print("y_raw mean/std:", y_raw.mean(), y_raw.std())
'''
close_db()

# The best performing model was Model 7. The goal is to determine which features are the most
# important. Let's start with backward selection (test full model then remove features one by one
# and measure the impact on MSE).

def backward_selection(features: list[str]):

    prop_train = 0.4
    prop_test = 0.3
    prop_val = 0.3

    # Train the full model as baseline

    print(f"Training the full model...")
    X_full, y_full = define_features(features, target=TARGET)
    d_val_old: float = fit_FNN(X_full, y_full, prop_train, prop_test, prop_val)

    feature_effect = {}

    # features[:] makes a copy of the list since we are removing elements from the list we are
    # iterating through. Let's look at the features in reverse...
    for feature in features[::-1]:
        # Remove the feature and fit the model
        features.remove(feature)

        print(f"Removing feature {feature}...")
        X, y = define_features(features, target=TARGET)
        d_val_new = fit_FNN(X, y, prop_train, prop_test, prop_val)

        # Store the change in Coen's d-val as the feature's impact on the model
        # Large d_val_change => model gained accuracy when dropping feature => feature added noise
        # Negative d_val_change => model lost accuracy when dropping feature => important feature to keep
        d_val_change = d_val_new - d_val_old
        feature_effect[feature] = (-1) * d_val_change

        # The new MSE becomes the old MSE
        d_val_old = d_val_new

        # If we go through all the features...
        if len(features) < 2:
            print(f"Out of features to test! \n")
            return feature_effect

    return feature_effect

# Look at the effects of feature selection via backwards selection on the FFNN
feature_effects = backward_selection(all_feature_names)

# Biggest to smallest, print the results
for key, value in sorted(feature_effects.items(), key=lambda item: item[1], reverse=True):
    print(f"Feature {key} provides {value:.6f} in Coen's d value")



end_total = time.perf_counter()
print(f"\nTotal runtime: {end_total - start_total:.4f} seconds")