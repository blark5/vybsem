"""Private heads on frozen Swin-B features (paper Section 3), and their no-noise versions.

Two methods, each ONE Gaussian release of summed statistics of unit-length features:
  ncm   : S = per-class feature sums (k x d), sensitivity 1.
  ridge : G = sum x x^T (d x d) and B = sum x y^T (d x k), released jointly, sensitivity sqrt(2);
          solved with spectral flooring at the data-independent lambda = s_tot * sqrt(2d).
Two settings:
  central   : one party holds all data and adds noise once.
  federated : 8 clients (Dirichlet(0.1) label split, as LA-LoRA), each adds its own noise, the server adds the messages.
No noise: NCM from the exact sums; plain ridge (no flooring) with lambda chosen on a 10% hold-out of train.

Usage: .venv/bin/python -I our_src/dp_head.py <root> [tiny|cifar100]"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from common import DATASET_NAMES, fraction_correct, load_features, result_path, unit_rows, write_json

DELTA = 1e-5
EPSILONS = [0.5, 1, 2, 3, 8]
SEEDS = list(range(10))           # noise seeds
CLIENTS = 8
DIRICHLET_BETA = 0.1
SPLIT_SEED = 0
SETTINGS = ["central", "federated"]
NCM_SENSITIVITY = 1.0              # one example changes one row of S by a vector of length 1
RIDGE_SENSITIVITY = math.sqrt(2)   # one example changes G by norm 1 and B by norm 1
METHOD_ID = {"ncm": 0, "ridge": 1}    # only used to give each method its own random numbers
NONPRIVATE_LAMBDAS = np.logspace(-2, 4, 13)


# analytic Gaussian mechanism (Balle and Wang 2018)

def delta_of(z, eps):
    """The smallest delta for which noise multiplier z gives (eps, delta)-DP for one Gaussian release."""
    return norm.cdf(0.5 / z - eps * z) - math.exp(eps) * norm.cdf(-0.5 / z - eps * z)


def noise_multiplier(eps, delta):
    """Smallest z with delta_of(z, eps) <= delta, by bisection (delta_of falls as z grows)."""
    too_small, enough = 1e-3, 1.0
    while delta_of(enough, eps) > delta:
        enough *= 2
    for _ in range(200):
        mid = (too_small + enough) / 2
        if delta_of(mid, eps) > delta:
            too_small = mid
        else:
            enough = mid
    return enough


# data split and statistics
def dirichlet_split(y, k, clients, beta, seed):
    """Label-skewed split: for each class, draw client shares from Dirichlet(beta) and deal that class's
    examples out in those shares. Every example goes to exactly one client."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(clients)]
    for c in range(k):
        idx = rng.permutation(np.flatnonzero(y == c))
        shares = rng.dirichlet(beta * np.ones(clients))
        cuts = np.round(np.cumsum(shares)[:-1] * len(idx)).astype(int)
        for j, chunk in enumerate(np.split(idx, cuts)):
            parts[j].append(chunk)
    return [np.concatenate(p) for p in parts]


def class_sums(X, y, k):
    """S (k x d): row c is the sum of the features of class c. Note B = sum x y^T = S^T."""
    S = np.zeros((k, X.shape[1]))
    np.add.at(S, y, X)
    return S


def gram(X):
    """G (d x d) = sum x x^T."""
    return X.T @ X


def noisy_sum(parts, s, rng):
    """Each party adds its own N(0, s^2) noise to its statistic; return the sum of the noisy statistics.
    One part = central setting. The total noise has standard deviation s * sqrt(len(parts))."""
    total = np.zeros_like(parts[0])
    for p in parts:
        total += p + rng.normal(0.0, s, size=p.shape)
    return total


def noise_rng(seed, method, setting, eps):
    """A separate random generator for every (noise seed, method, setting, epsilon)."""
    return np.random.default_rng([seed, METHOD_ID[method], SETTINGS.index(setting), EPSILONS.index(eps)])


# ---------- the two heads ----------

def ncm_predict(S, X):
    """Class whose (noisy) sum row has the highest cosine similarity with x. No class counts needed."""
    return (X @ unit_rows(S).T).argmax(1)


def sym_eig(G):
    """Symmetrise (post-processing) and eigendecompose: eigenvalues mu and eigenvectors V."""
    return np.linalg.eigh((G + G.T) / 2)


def ridge_weights(mu, V, B, lam, floor):
    """W from G = V diag(mu) V^T. floor=False: weight 1/(mu + lam), i.e. (G + lam I)^-1 B (plain ridge).
    floor=True: weight 1/max(mu, lam) (spectral flooring)."""
    w = np.maximum(mu, lam) if floor else mu + lam
    return V @ ((V.T @ B) / w[:, None])


def ridge_predict(W, X):
    return (X @ W).argmax(1)


def ridge_lambda(s_tot, d):
    """Data-independent rule: about the largest eigenvalue of the symmetrised noise, s_tot * sqrt(2d)."""
    return s_tot * math.sqrt(2 * d)


def accuracy(pred, y):
    """Percent correct."""
    return 100.0 * fraction_correct(pred, y)


def summary(accs):
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)), "per_seed": accs}


# ---------- one result cell: one method, one setting, one epsilon, all noise seeds ----------

def ncm_cell(S_parts, Xte, yte, setting, eps):
    s = noise_multiplier(eps, DELTA) * NCM_SENSITIVITY
    accs = [accuracy(ncm_predict(noisy_sum(S_parts, s, noise_rng(seed, "ncm", setting, eps)), Xte), yte) for seed in SEEDS]
    return summary(accs)


def ridge_cell(G_parts, S_parts, Xte, yte, setting, eps):
    s = noise_multiplier(eps, DELTA) * RIDGE_SENSITIVITY
    lam = ridge_lambda(s * math.sqrt(len(G_parts)), Xte.shape[1])
    accs = []
    for seed in SEEDS:
        rng = noise_rng(seed, "ridge", setting, eps)
        G_noisy = noisy_sum(G_parts, s, rng)          # G before B: the order fixes the random numbers
        B_noisy = noisy_sum(S_parts, s, rng).T        # B = S^T
        mu, V = sym_eig(G_noisy)
        accs.append(accuracy(ridge_predict(ridge_weights(mu, V, B_noisy, lam, floor=True), Xte), yte))
    return {"lambda": lam, **summary(accs)}


# ---------- no noise ----------

def centred_ridge_holdout_acc(Xfit, yfit, Xhold, yhold, k, lam):
    """Centred ridge onto one-hot labels, scored on the hold-out. Used only to choose the no-noise lambda."""
    mu = Xfit.mean(0)
    Y = np.eye(k)[yfit]
    W = np.linalg.solve((Xfit - mu).T @ (Xfit - mu) + lam * np.eye(Xfit.shape[1]), (Xfit - mu).T @ (Y - Y.mean(0)))
    return fraction_correct(((Xhold - mu) @ W + Y.mean(0)).argmax(1), yhold)


def nonoise(Xtr, ytr, Xte, yte, k, S, G):
    """NCM from the exact sums; plain uncentred ridge at the lambda that centred ridge picks on a stratified
    10% hold-out of train (seed 0)."""
    fit, hold = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=0)
    holdout = {lam: centred_ridge_holdout_acc(Xtr[fit], ytr[fit], Xtr[hold], ytr[hold], k, lam) for lam in NONPRIVATE_LAMBDAS}
    lam = float(max(holdout, key=holdout.get))
    mu, V = sym_eig(G)
    return {"ncm": accuracy(ncm_predict(S, Xte), yte),
            "ridge": accuracy(ridge_predict(ridge_weights(mu, V, S.T, lam, floor=False), Xte), yte),
            "ridge_lambda": lam}


# ---------- one dataset ----------

def run(root, ds):
    Xtr, ytr, Xte, yte, k = load_features(root, ds)
    clients = dirichlet_split(ytr, k, CLIENTS, DIRICHLET_BETA, SPLIT_SEED)
    S_clients = [class_sums(Xtr[i], ytr[i], k) for i in clients]
    G_clients = [gram(Xtr[i]) for i in clients]
    S_parts = {"central": [sum(S_clients)], "federated": S_clients}
    G_parts = {"central": [sum(G_clients)], "federated": G_clients}

    out = {"meta": {"dataset": ds, "delta": DELTA, "clients": CLIENTS, "seeds": len(SEEDS), "accuracy_unit": "percent",
                    "std": "sample standard deviation over noise seeds (ddof=1)"},
           "ncm": {}, "ridge": {},
           "nonoise": nonoise(Xtr, ytr, Xte, yte, k, S_parts["central"][0], G_parts["central"][0])}
    print(ds, "no noise", out["nonoise"], flush=True)
    for setting in SETTINGS:
        out["ncm"][setting], out["ridge"][setting] = {}, {}
        for eps in EPSILONS:
            n = out["ncm"][setting][str(eps)] = ncm_cell(S_parts[setting], Xte, yte, setting, eps)
            r = out["ridge"][setting][str(eps)] = ridge_cell(G_parts[setting], S_parts[setting], Xte, yte, setting, eps)
            print(ds, setting, "eps", eps, "ncm %.2f +- %.2f" % (n["mean"], n["std"]),
                  "ridge %.2f +- %.2f" % (r["mean"], r["std"]), flush=True)
    write_json(out, result_path(root, "dp_head_%s.json" % ds))


if __name__ == "__main__":
    ROOT = sys.argv[1]
    for ds in (sys.argv[2:] or DATASET_NAMES):
        run(ROOT, ds)
