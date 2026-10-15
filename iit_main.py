from tpm_generator import bias_generator, weight_generator, tpm_linear_generator_split
import iit_computation
import numpy as np
import numpy.typing as npt
from database import (write_to_db, close_db, drop_db, create_db, get_all_rows,
                      get_row_by_num_nodes, get_count_systems_generated, drop_tpm_column)
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from torch.utils.data import TensorDataset, DataLoader
import copy
from feature_generator import compute_nbn_features
from sklearn.preprocessing import StandardScaler
import itertools
from typing import NamedTuple, Optional
import optuna

test_seed: int = 50

# Seeds for repeated training runs. The train/validation/test split always uses test_seed, so every
# run sees identical data (and therefore an identical baseline). Only the torch seed varies, which
# controls weight initialization, batch order and dropout masks.
N_SEEDS: int = 5
TORCH_SEEDS: list[int] = [test_seed + k for k in range(N_SEEDS)]

# During backward selection, a feature is only removed if doing so lowers the mean validation MSE
# by more than the SE_MARGINs of standard deviation.
SE_MARGIN: float = 0.5

# Cross-validation parameters
CV_FOLDS: int = 5
INNER_VAL_FRAC: float = 0.15  # slice of each fold's training part reserved for early stopping
N_TRIALS: int = 40
RETUNE_ON_REDUCED: bool = True
HPARAMS_OUT: str = "best_hparams.json"


class FitResult(NamedTuple):
    val_mse: float
    test_mse: Optional[float] = None
    cohens_d: Optional[float] = None
    baseline_mse: Optional[float] = None
    model: Optional[nn.Module] = None
    scaler: Optional[StandardScaler] = None
    train_mean: Optional[float] = None
    r2: Optional[float] = None


class HParams(NamedTuple):
    hidden_dims: Optional[tuple] = None
    dropout_rate: float = 0.2
    weight_decay: float = 1e-4
    lr: float = 1e-3
    batch_size: int = 32


# Default configuration. Will be replaced by cross-validated parameters if they return better results
DEFAULT_HP = HParams()

start_total = time.perf_counter()

# Scalar features
NBN_FEATURE_NAMES: list[str] = [
    "mixing_gap", "spectral_entropy", "wd", "wr",
    "weight_cluster_coeff", "short_path_len", "small_world_coeff", "cheeger_coeff",
    "num_sccs", "max_scc", "diam", "avg_closeness",
    "avg_betweenness", "max_pr", "min_pr", "mean_pr"
]

# Features that requre specific handling (i.e; serialization for network feeding)
STRUCTURAL_KEYS = {"tpm_prior"}


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

    # Each entry is a dict: {"tpm_prior", "features": {...}}. The TPM itself is never kept.
    network_properties: list[dict] = []

    N: int = 2 ** n
    rng = np.random.default_rng(seed=test_seed)

    for i in range(num_tpms):
        bias = bias_generator(n)
        weight = weight_generator(n)

        prior: npt.NDArray[np.float64] = np.full(N, 1 / N, dtype=np.float64)

        # The TPM lives only for this iteration: it is used to compute the features and then discarded
        tpm = tpm_linear_generator_split(n, bias, weight, temp=1, p=0.1, rng=rng)

        ii, mi_Xt_Xtpast = iit_computation.integrated_information(tpm, prior)

        nbn_features: list = compute_nbn_features(tpm)

        features = {
            "ii": ii,
            "mi": mi_Xt_Xtpast,
            "num_nodes": n,
            **dict(zip(NBN_FEATURE_NAMES, nbn_features)),
        }

        network_properties.append({
            "tpm_prior": prior,
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

drop_tpm_column()   # one-off: strips stored TPMs from an existing DB; no-op afterwards

#gen_and_write_to_db(n=8, num_tpms=500, rewrite_db=False)
gen_and_write_to_db(n=10, num_tpms=500, rewrite_db=False)

sys6 = get_count_systems_generated(6)
sys8 = get_count_systems_generated(8)
sys10 = get_count_systems_generated(10)

print(f"Number of 6 node systems: {sys6} \n"
      f"Number of 8 node systems: {sys8} \n"
      f"Number of 10 node systems: {sys10}. \n")


# Split 6 node and 8 node systems. The neural network downstream trains only on the 6 node systems
rows_6 = get_row_by_num_nodes(6)
rows_8 = get_row_by_num_nodes(8)
rows_10 = get_row_by_num_nodes(10)


def _get(row: dict, name: str):
    if name in STRUCTURAL_KEYS:
        return row[name]
    return row["features"][name]


def define_features(feature_names: list[str], target="ii", source=None):
    source = rows_6 if source is None else source
    y = np.array([_get(row, target) for row in source], dtype=np.float32)
    X = np.array(
        [flatten_predictors([_get(row, name) for name in feature_names]) for row in source],
        dtype=np.float32,
    )
    return X, y


# Fallback architecture used only when HParams.hidden_dims is None (i.e. the original behaviour).
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


# Split proportions used for every fit, so the split (and therefore the baseline) never changes
PROP_TRAIN, PROP_TEST, PROP_VAL = 0.4, 0.3, 0.3


def split_data(X, y, prop_train: float = PROP_TRAIN, prop_test: float = PROP_TEST,
               prop_val: float = PROP_VAL):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - prop_train, random_state=test_seed
    )
    relative_test_size = prop_test / (prop_test + prop_val)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=test_seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _train_network(X_train, y_train, X_val, y_val, hp: HParams, seed: int):


    torch.manual_seed(seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)

    n_features = X_train.shape[1]
    hidden_dims = list(hp.hidden_dims) if hp.hidden_dims is not None else \
        suggest_architecture(n_features=n_features, n_samples=X_train.shape[0])
    model = SimpleFFNN(input_dim=n_features, hidden_dims=hidden_dims, output_dim=1,
                       dropout_rate=hp.dropout_rate)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=hp.weight_decay, lr=hp.lr)

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

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            scheduler.step(val_loss)

        if val_loss < best_val_loss - val_threshold:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= epoch_patience:
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, scaler, best_val_loss


def _mse(model: nn.Module, scaler: StandardScaler, X, y) -> float:
    X_t = torch.tensor(scaler.transform(X), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(X_t).numpy().flatten()
    return float(np.mean((y.flatten().astype(np.float64) - pred) ** 2))


def fit_FNN(X, y, prop_train: float = PROP_TRAIN, prop_test: float = PROP_TEST,
            prop_val: float = PROP_VAL, seed: int = test_seed, evaluate_test: bool = False,
            hp: HParams = DEFAULT_HP) -> FitResult:

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, prop_train, prop_test, prop_val)

    model, scaler, best_val_loss = _train_network(X_train, y_train, X_val, y_val, hp, seed)

    if not evaluate_test:
        return FitResult(val_mse=best_val_loss)

    X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = nn.MSELoss()(test_pred, y_test_t)

    print("\nTest MSE:", f"{test_loss.item():.8e}")

    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)

    nn_sq_errors = (y_test.flatten() - test_pred.numpy().flatten()) ** 2
    baseline_sq_errors = (y_test.flatten() - baseline_pred.flatten()) ** 2

    baseline_mse: float = float(np.mean(baseline_sq_errors))
    print(f"Baseline MSE: {baseline_mse:.8e}")

    y_test_flat = y_test.flatten().astype(np.float64)
    r2: float = float(1 - nn_sq_errors.sum() / ((y_test_flat - y_test_flat.mean()) ** 2).sum())
    print(f"Test R^2 (vs test mean): {r2:.6f}")

    error_diffs = baseline_sq_errors - nn_sq_errors
    mean_diff = np.mean(error_diffs)
    sd_diff = np.std(error_diffs, ddof=1)

    cohens_d: float = mean_diff / sd_diff if sd_diff > 0 else np.inf

    print(f"Cohen's d (paired): {cohens_d:.4f}")

    return FitResult(val_mse=best_val_loss, test_mse=test_loss.item(), cohens_d=cohens_d,
                     baseline_mse=baseline_mse, model=model, scaler=scaler,
                     train_mean=float(np.mean(y_train)), r2=r2)


# Make sure our predicting factor is not part of the feature space
TARGET = "ii"


# Features that are constant are guarenteed to provide noise and thus eliminated in backwards selection.
# These features are dropped prior to the actual algorithm to save on computational cost.
def drop_constant_features(feature_names: list[str]) -> tuple[list[str], list[str]]:
    varying, constant = [], []
    for name in feature_names:
        values = np.array([_get(row, name) for row in rows_6], dtype=np.float64)
        if np.ptp(values) == 0:
            constant.append(name)
        else:
            varying.append(name)
    return varying, constant


candidate_feature_names = [k for k in rows_6[0]["features"].keys() if k != TARGET]
all_feature_names, constant_features = drop_constant_features(candidate_feature_names)
close_db()

print(f"Dropped {len(constant_features)} constant feature(s) before selection: {constant_features}")

def _sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def hp_from_params(p: dict) -> HParams:
    base = p["base_width"]
    if p["shape"] == "funnel":
        dims = tuple(max(8, base // (2 ** i)) for i in range(p["n_layers"]))
    else:
        dims = tuple(base for _ in range(p["n_layers"]))
    return HParams(
        hidden_dims=dims,
        dropout_rate=p["dropout_rate"],
        weight_decay=p["weight_decay"],
        lr=p["lr"],
        batch_size=p["batch_size"],
    )


# Collection of parameters to cross-validate over
def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "base_width": trial.suggest_categorical("base_width", [32, 64, 128, 256]),
        "shape": trial.suggest_categorical("shape", ["funnel", "flat"]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.4),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def dev_pool(X, y):
    X_train, y_train, X_val, y_val, _, _ = split_data(X, y)
    return np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])


def cv_mse(X_dev, y_dev, hp: HParams, trial: Optional[optuna.Trial] = None) -> np.ndarray:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=test_seed)
    fold_mses: list[float] = []

    for k, (fit_idx, score_idx) in enumerate(kf.split(X_dev)):
        # Inner split of the fold's training part: fit the weights / stop early
        X_fit, X_stop, y_fit, y_stop = train_test_split(
            X_dev[fit_idx], y_dev[fit_idx], test_size=INNER_VAL_FRAC, random_state=test_seed
        )
        model, scaler, _ = _train_network(X_fit, y_fit, X_stop, y_stop, hp, seed=test_seed + k)
        fold_mses.append(_mse(model, scaler, X_dev[score_idx], y_dev[score_idx]))

        # Prune clearly bad trials after the first folds instead of paying for all of them
        if trial is not None:
            trial.report(float(np.mean(fold_mses)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.array(fold_mses)


def tune_hyperparameters(features: list[str], label: str, n_trials: int = N_TRIALS) -> HParams:
    print(f"\n=== Hyperparameter search: {label} ({len(features)} features, {n_trials} trials, "
          f"{CV_FOLDS}-fold CV) ===")
    X, y = define_features(features, target=TARGET)
    X_dev, y_dev = dev_pool(X, y)
    print(f"Dev pool: {len(X_dev)} rows (test rows excluded)")

    def objective(trial: optuna.Trial) -> float:
        hp = hp_from_params(suggest_params(trial))
        return float(cv_mse(X_dev, y_dev, hp, trial).mean())

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Trial {trial.number:3d}: CV MSE {trial.value:.8e} | best so far {study.best_value:.8e}")
        else:
            print(f"Trial {trial.number:3d}: pruned")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=test_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])

    best_hp = hp_from_params(study.best_params)

    # Re-score the winner and the original hardcoded config with identical folds for a fair comparison
    best_scores = cv_mse(X_dev, y_dev, best_hp)
    default_scores = cv_mse(X_dev, y_dev, DEFAULT_HP)
    print(f"\nBest hyperparameters: {best_hp}")
    print(f"Tuned   CV MSE: {best_scores.mean():.8e} (SD across folds {_sd(best_scores):.8e})")
    print(f"Default CV MSE: {default_scores.mean():.8e} (SD across folds {_sd(default_scores):.8e})")
    if best_scores.mean() >= default_scores.mean():
        print("Note: the tuned config did not beat the original defaults on CV.")
    print("Note: the tuned CV score is mildly optimistic since the search picked it on these same folds.")

    return best_hp


def save_hparams(path: str, **named_hp: HParams) -> None:
    with open(path, "w") as f:
        json.dump({k: v._asdict() for k, v in named_hp.items()}, f, indent=2)
    print(f"Saved hyperparameters to {path}")


def validation_scores(features: list[str], hp: HParams) -> np.ndarray:
    """Validation MSE of a model on `features`, one entry per torch seed. Never touches the test set."""
    X, y = define_features(features, target=TARGET)
    return np.array([
        fit_FNN(X, y, PROP_TRAIN, PROP_TEST, PROP_VAL, seed=s, hp=hp).val_mse for s in TORCH_SEEDS
    ])


def backward_selection(features: list[str], hp: HParams) -> list[str]:
    current = list(features)

    print(f"There are {len(current)} features in the full model. Beginning backwards selection...")
    print(f"Features: {current}")
    print(f"Hyperparameters held fixed throughout: {hp}")
    print(f"Each model is trained with {N_SEEDS} seeds and compared on mean validation MSE.\n")

    current_scores = validation_scores(current, hp)
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
            scores = validation_scores(candidate, hp)
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


def evaluate_on_test(features: list[str], hp: HParams) -> list[FitResult]:
    """One-off final evaluation on the test set, one FitResult per torch seed."""
    X, y = define_features(features, target=TARGET)
    return [
        fit_FNN(X, y, PROP_TRAIN, PROP_TEST, PROP_VAL, seed=s, evaluate_test=True, hp=hp)
        for s in TORCH_SEEDS
    ]


def report(label: str, results: list[FitResult]) -> None:
    mse = np.array([r.test_mse for r in results])
    d = np.array([r.cohens_d for r in results])
    r2 = np.array([r.r2 for r in results])
    r2_base = np.array([1 - r.test_mse / r.baseline_mse for r in results])
    print(f"{label} MSE value: {mse.mean():.8e} (SD across seeds {_sd(mse):.8e})")
    print(f"{label} Cohen's d value: {d.mean():.8f} (SD across seeds {_sd(d):.8f})")
    print(f"{label} R^2 vs test mean: {r2.mean():.6f} (SD across seeds {_sd(r2):.6f})")
    print(f"{label} R^2 vs train mean: {r2_base.mean():.6f} (SD across seeds {_sd(r2_base):.6f})")


# The full model is every non-constant feature. backward_selection works on a copy of it.
full_features = list(all_feature_names)

hp_full = tune_hyperparameters(full_features, label="full feature set")

features_sub = backward_selection(full_features, hp_full)
removed_features = [f for f in full_features if f not in features_sub]

if features_sub == full_features:
    hp_sub = hp_full
elif RETUNE_ON_REDUCED:
    hp_sub = tune_hyperparameters(features_sub, label="reduced feature set")
else:
    hp_sub = hp_full

save_hparams(HPARAMS_OUT, full=hp_full, reduced=hp_sub)

print("\nSelection complete. Evaluating the full and reduced models on the test set...\n")
full_results = evaluate_on_test(full_features, hp_full)

if features_sub == full_features:
    print("\nNo features were removed, so the reduced model is identical to the full model.")
    sub_results = full_results
else:
    sub_results = evaluate_on_test(features_sub, hp_sub)

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


def evaluate_transfer(label: str, features: list[str], results: list[FitResult], source_rows: list[dict]) -> None:
    X_new, y_new = define_features(features, target=TARGET, source=source_rows)
    y_flat = y_new.flatten().astype(np.float64)

    # Baseline: mean ii of the 8-node systems. Independent of the trained models and of every seed.
    base_sq = (y_flat - y_flat.mean()) ** 2
    baseline_mse = base_sq.mean()

    mses, ds, r2s = [], [], []
    for r in results:
        X_t = torch.tensor(r.scaler.transform(X_new), dtype=torch.float32)  # the scaler the network was trained with
        r.model.eval()
        with torch.no_grad():
            pred = r.model(X_t).numpy().flatten()

        nn_sq = (y_flat - pred) ** 2
        diff = base_sq - nn_sq
        sd = np.std(diff, ddof=1)

        mses.append(nn_sq.mean())
        ds.append(diff.mean() / sd if sd > 0 else np.inf)
        r2s.append(1 - nn_sq.mean() / baseline_mse)

    mses, ds, r2s = map(np.array, (mses, ds, r2s))
    print(f"{label} | 8-node baseline MSE (8-node mean): {baseline_mse:.8e}")
    print(f"{label} | 8-node MSE: {mses.mean():.8e} (SD across seeds {_sd(mses):.8e})")
    print(f"{label} | 8-node Cohen's d: {ds.mean():.8f} (SD across seeds {_sd(ds):.8f})")
    print(f"{label} | 8-node R^2 vs 8-node mean: {r2s.mean():.6f} (SD across seeds {_sd(r2s):.6f})")


# Run only after everything above is frozen: nothing was tuned against these 8-node systems.
print(f"\nTransfer to {len(rows_8)} eight-node systems (6-node-trained models, 6-node scaler and baseline):")
evaluate_transfer(f"Full model ({len(full_features)} features)", full_features, full_results, rows_8)
if features_sub == full_features:
    print("Reduced model identical to full model.")
else:
    evaluate_transfer(f"Reduced model ({len(features_sub)} features)", features_sub, sub_results, rows_8)


end_total = time.perf_counter()
print(f"\nTotal runtime: {end_total - start_total:.4f} seconds")