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
import itertools
from typing import NamedTuple, Optional

test_seed: int = 50

# Seeds for repeated training runs. The train/validation/test split always uses test_seed, so every
# run sees identical data (and therefore an identical baseline). Only the torch seed varies, which
# controls weight initialization, batch order and dropout masks.
N_SEEDS: int = 5
TORCH_SEEDS: list[int] = [test_seed + k for k in range(N_SEEDS)]

# During backward selection, a feature is only removed if doing so lowers the mean validation MSE
# by more than SE_MARGIN standard errors of the current model's seed-to-seed spread.
# Set to 0 to accept any improvement in the mean validation MSE.
SE_MARGIN: float = 1.0


class FitResult(NamedTuple):
    """Outputs of one training run. Test-set fields are None unless evaluate_test=True."""
    val_mse: float
    test_mse: Optional[float] = None
    cohens_d: Optional[float] = None
    baseline_mse: Optional[float] = None

start_total = time.perf_counter()

# Scalar features
NBN_FEATURE_NAMES: list[str] = [
    "mixing_gap", "spectral_entropy", "wd", "wr",
    "weight_cluster_coeff", "short_path_len", "small_world_coeff", "cheeger_coeff",
    "num_sccs", "max_scc", "diam", "avg_closeness",
    "avg_betweenness", "max_pr", "min_pr", "mean_pr"
]

# Features that requre specific handling (i.e; serialization for network feeding)
STRUCTURAL_KEYS = {"tpm", "tpm_prior"}


# The neural network. Feed forward for now. Using softplus activation since ii is non-negative
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
            layers.append(nn.Softplus())

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

    # Each entry is now a dict: {"tpm", "tpm_prior", "features": {...}}
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

        ii, mi_Xt_Xtpast = iit_computation.integrated_information(tpms_linear[i], priors[i])

        nbn_features: list = compute_nbn_features(tpms_linear[i])

        features = {
            "ii": ii,
            "mi": mi_Xt_Xtpast,
            "num_nodes": n,
            **dict(zip(NBN_FEATURE_NAMES, nbn_features)),
        }

        network_properties.append({
            "tpm": tpms_linear[i],
            "tpm_prior": priors[i],
            "features": features,
        })

        print(
            f"Finished generating TPM number {i + 1} and computing its integrated information. Attributes have been computed. Next TPM...")

    print(f"Writing to database...")
    write_to_db(network_properties)

    tpm_gen_end_time: float = time.perf_counter()
    total_tpm_time: float = tpm_gen_end_time - tpm_gen_start_time

    if total_tpm_time > 60:
        print(f"\nComplete! Finished processing {num_tpms} tpms in {int(total_tpm_time / 60)} "
              f"minutes and {total_tpm_time % 60:.4f} seconds")
    else:
        print(f"\nComplete! Finished processing {num_tpms} tpms in {total_tpm_time:.4f} seconds")


def gen_and_write_to_db(n: int = 4, num_tpms: int = 100, rewrite_db: bool = True) -> None:
    # If we want to regenerate every system, otherwise add on to what previously exists
    if rewrite_db:
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
#gen_and_write_to_db(n=6, num_tpms=20_000, rewrite_db=True)

rows = get_all_rows()


def _get(row: dict, name: str):
    if name in STRUCTURAL_KEYS:
        return row[name]
    return row["features"][name]


def define_features(feature_names: list[str], target="ii"):
    y = np.array([_get(row, target) for row in rows], dtype=np.float32)
    X = np.array(
        [flatten_predictors([_get(row, name) for name in feature_names]) for row in rows],
        dtype=np.float32,
    )
    return X, y


# For deciding how many hidden layers and the dim for each layer. This is TEMPORARY, to be changed
# to a validation loop to grid search parameters...
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


def fit_FNN(X, y, prop_train: float = 0.4, prop_test: float = 0.3, prop_val: float = 0.3,
            seed: int = test_seed, evaluate_test: bool = False) -> FitResult:
    """
    Trains the FFNN and returns a FitResult.

    The data split always uses test_seed, so every call sees the same train/validation/test rows.
    `seed` only controls torch (weight initialization, DataLoader shuffling and dropout masks), so
    repeated calls with different seeds measure run-to-run variation on identical data.

    The test set is only evaluated when evaluate_test=True. Feature selection should call this with
    evaluate_test=False and rely on val_mse, so the test set is only used for the final report.
    """
    torch.manual_seed(seed)

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

    print(f"Beginning training the neural net (seed {seed})...")

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
            # print(f"New best model saved at epoch {epoch}")
            # print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4e} | Val Loss: {val_loss:.4e}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= epoch_patience:
                # print(f"\nEarly stopping at epoch {epoch} — no improvement for {epoch_patience} consecutive epochs.")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        # print(f"\nRestored best model weights (val_loss={best_val_loss:.4e})")

    if not evaluate_test:
        return FitResult(val_mse=best_val_loss)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t)

    print("\nTest MSE:", f"{test_loss.item():.8e}")

    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)

    nn_sq_errors = (y_test.flatten() - test_pred.numpy().flatten()) ** 2
    baseline_sq_errors = (y_test.flatten() - baseline_pred.flatten()) ** 2

    baseline_mse: float = float(np.mean(baseline_sq_errors))
    print(f"Baseline MSE: {baseline_mse:.8e}")

    error_diffs = baseline_sq_errors - nn_sq_errors
    mean_diff = np.mean(error_diffs)
    sd_diff = np.std(error_diffs, ddof=1)

    cohens_d: float = mean_diff / sd_diff if sd_diff > 0 else np.inf

    print(f"Cohen's d (paired): {cohens_d:.4f}")

    return FitResult(val_mse=best_val_loss, test_mse=test_loss.item(), cohens_d=cohens_d, baseline_mse=baseline_mse)


# Make sure our predicting factor is not part of the feature space
TARGET = "ii"

# Split proportions used for every fit, so the split (and therefore the baseline) never changes
PROP_TRAIN, PROP_TEST, PROP_VAL = 0.4, 0.3, 0.3


def drop_constant_features(feature_names: list[str]) -> tuple[list[str], list[str]]:
    """
    Splits feature_names into (varying, constant). A constant feature (e.g. num_nodes when every
    system has the same number of nodes) is standardized to all zeros, so it carries no information
    and its removal would only change the network's width and random initialization, not its
    function. Removing these before selection stops backward selection from chasing that noise.
    """
    varying, constant = [], []
    for name in feature_names:
        values = np.array([_get(row, name) for row in rows], dtype=np.float64)
        if np.ptp(values) == 0:
            constant.append(name)
        else:
            varying.append(name)
    return varying, constant


# Need to hardcode for now, but max_mi has been removed from the database so next dataset generation
# will eliminate max_mi from the feature space
candidate_feature_names = [k for k in rows[0]["features"].keys() if k != TARGET and k != "max_mi"]
all_feature_names, constant_features = drop_constant_features(candidate_feature_names)
close_db()

print(f"Dropped {len(constant_features)} constant feature(s) before selection: {constant_features}")

# The best performing model was Model 7. The goal is to determine which features are the most
# important. Let's start with backward selection (test full model then remove features one by one
# and measure the impact on validation MSE).


def _sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def validation_scores(features: list[str]) -> np.ndarray:
    """Validation MSE of a model on `features`, one entry per torch seed. Never touches the test set."""
    X, y = define_features(features, target=TARGET)
    return np.array([
        fit_FNN(X, y, PROP_TRAIN, PROP_TEST, PROP_VAL, seed=s).val_mse for s in TORCH_SEEDS
    ])


def backward_selection(features: list[str]) -> list[str]:
    # Work on a copy so the caller's list is never mutated
    current = list(features)

    print(f"There are {len(current)} features in the full model. Beginning backwards selection...")
    print(f"Features: {current}")
    print(f"Each model is trained with {N_SEEDS} seeds and compared on mean validation MSE.\n")

    current_scores = validation_scores(current)
    iteration = 0

    while len(current) > 1:
        current_mean = float(current_scores.mean())
        current_se = _sd(current_scores) / np.sqrt(len(current_scores))

        # A removal is only accepted if it beats the current mean by more than SE_MARGIN standard
        # errors of the current model's seed-to-seed spread
        threshold = current_mean - SE_MARGIN * current_se

        print(f"Iteration {iteration} of backwards selection ({len(current)} features):")
        print(f"Current mean validation MSE: {current_mean:.8e} (SE {current_se:.8e}). "
              f"A removal must reach below {threshold:.8e}.\n")

        best_feature = None
        best_scores = None
        best_mean = threshold

        for feature in current:
            candidate = [f for f in current if f != feature]
            scores = validation_scores(candidate)
            mean = float(scores.mean())

            print(f"Without {feature}: mean validation MSE {mean:.8e}")

            # The removal with the lowest mean validation MSE among those beating the threshold wins
            if mean < best_mean:
                best_mean = mean
                best_feature = feature
                best_scores = scores

        if best_feature is None:
            print("\nNo removal beat the threshold. Optimal features recovered.")
            break

        print(f"\nFeature {best_feature} is permanently removed (mean validation MSE {best_mean:.8e}).")
        current.remove(best_feature)

        # The accepted candidate was already trained on every seed, so reuse its scores
        current_scores = best_scores
        iteration += 1
        print(f"There are {len(current)} features in the new model. Moving to next index...\n")

    return current


def evaluate_on_test(features: list[str]) -> list[FitResult]:
    """One-off final evaluation on the test set, one FitResult per torch seed."""
    X, y = define_features(features, target=TARGET)
    return [
        fit_FNN(X, y, PROP_TRAIN, PROP_TEST, PROP_VAL, seed=s, evaluate_test=True) for s in TORCH_SEEDS
    ]


def report(label: str, results: list[FitResult]) -> None:
    mse = np.array([r.test_mse for r in results])
    d = np.array([r.cohens_d for r in results])
    print(f"{label} MSE value: {mse.mean():.8e} (SD across seeds {_sd(mse):.8e})")
    print(f"{label} Cohen's d value: {d.mean():.8f} (SD across seeds {_sd(d):.8f})")


# The full model is every non-constant feature. backward_selection works on a copy of it.
full_features = list(all_feature_names)
features_sub = backward_selection(full_features)
removed_features = [f for f in full_features if f not in features_sub]

# Only now is the test set used, once per model and seed
print("\nSelection complete. Evaluating the full and reduced models on the test set...\n")
full_results = evaluate_on_test(full_features)

if features_sub == full_features:
    print("\nNo features were removed, so the reduced model is identical to the full model.")
    sub_results = full_results
else:
    sub_results = evaluate_on_test(features_sub)

# The split and target are fixed, so the baseline must be identical for every model and seed
baseline_mse = full_results[0].baseline_mse
assert all(np.isclose(r.baseline_mse, baseline_mse) for r in full_results + sub_results)

print(f"\nTest results, averaged over {N_SEEDS} seeds:")
print(f"Baseline MSE (predict train mean): {baseline_mse:.8e}")
report(f"Full model ({len(full_features)} features)", full_results)
report(f"Reduced model ({len(features_sub)} features)", sub_results)

print(f"\nConstant features dropped before selection ({len(constant_features)}): {constant_features}")
print(f"Remaining features after backward selection ({len(features_sub)}): {features_sub}")
print(f"Removed features ({len(removed_features)}): {removed_features}")

end_total = time.perf_counter()
print(f"\nTotal runtime: {end_total - start_total:.4f} seconds")

# Prior run under the old Cohen's d criterion (now replaced) for reference only:
# Full model: 0.5048
# Sub model: 0.5096

# Earlier runs selected features on test MSE from a single training run. That criterion is noisy and
# optimistically biased, so these results are for reference only and should not be relied upon:
# Unseeded run: ['mi', 'wr', 'weight_cluster_coeff', 'short_path_len', 'cheeger_coeff', 'max_scc', 'avg_closeness', 'max_pr']
# Seeded run: only 'diam' was removed (17 of 18 features remained, including the constant num_nodes)