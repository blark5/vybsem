"""Checks of the claims in the paper's Section 3: noise calibration (checked with dp_accounting's PLD accountant),
sensitivities 1 and sqrt(2), and what spectral flooring does.
Run: .venv/bin/python -I -m pytest tests/test_dp_head.py -v"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "our_src"))
import numpy as np
import pytest

import dp_head as H
from common import DATASET_NAMES, load_features


@pytest.mark.parametrize("eps", H.EPSILONS)
def test_calibration(eps):
    """dp_accounting's PLD accountant, given our z, returns the target epsilon: its optimistic and pessimistic
    estimates bracket the target, and both are within 1% of it."""
    from dp_accounting.pld import privacy_loss_distribution as pld
    z = H.noise_multiplier(eps, H.DELTA)
    upper = pld.from_gaussian_mechanism(z, value_discretization_interval=1e-4).get_epsilon_for_delta(H.DELTA)
    lower = pld.from_gaussian_mechanism(z, value_discretization_interval=1e-4, pessimistic_estimate=False).get_epsilon_for_delta(H.DELTA)
    print("eps %g  z %.6f  PLD lower %.6f  upper %.6f" % (eps, z, lower, upper))
    assert lower <= eps * (1 + 1e-6) and eps <= upper * (1 + 1e-6)
    assert abs(upper - eps) / eps < 0.01 and abs(lower - eps) / eps < 0.01


@pytest.mark.parametrize("ds", DATASET_NAMES)
def test_sensitivity(ds):
    """Remove one example and recompute the statistics from scratch (on a 2,000-example subset):
    S changes by norm <= 1, (G, B) jointly by norm <= sqrt(2)."""
    Xtr, ytr, _, _, k = load_features(ROOT, ds)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ytr), 2000, replace=False)
    X, y = Xtr[sub], ytr[sub]
    S, G = H.class_sums(X, y, k), H.gram(X)
    worst_S = worst_GB = 0.0
    for i in rng.choice(2000, 100, replace=False):
        keep = np.arange(2000) != i
        dS = np.linalg.norm(S - H.class_sums(X[keep], y[keep], k))
        dG = np.linalg.norm(G - H.gram(X[keep]))
        worst_S, worst_GB = max(worst_S, dS), max(worst_GB, math.hypot(dG, dS))   # B = S^T, same norm
    print(ds, "largest change: S %.12f  (G,B) %.12f" % (worst_S, worst_GB))
    assert worst_S <= 1 + 1e-9 and worst_GB <= math.sqrt(2) + 1e-9


def _random_stats(n=3000, d=20, k=5, seed=0):
    """Random unit-length features and one-hot labels; G = sum x x^T, B = sum x y^T (invertible G for n >> d)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.T @ X, X.T @ np.eye(k)[rng.integers(0, k, n)]


def test_flooring_below_all_eigenvalues_is_unregularised():
    """lambda below every eigenvalue of G: flooring changes nothing, so W = G^-1 B; plain ridge differs."""
    G, B = _random_stats()
    mu, V = H.sym_eig(G)
    lam = 0.5 * mu.min()
    assert np.allclose(H.ridge_weights(mu, V, B, lam, floor=True), np.linalg.solve(G, B), rtol=1e-8, atol=1e-10)
    assert not np.allclose(H.ridge_weights(mu, V, B, lam, floor=False), np.linalg.solve(G, B), rtol=1e-3, atol=1e-6)


def test_flooring_above_all_eigenvalues_is_B_over_lambda():
    """lambda above every eigenvalue of G: every eigenvalue is raised to lambda, so W = B / lambda exactly."""
    G, B = _random_stats()
    mu, V = H.sym_eig(G)
    lam = 2.0 * mu.max()
    assert np.allclose(H.ridge_weights(mu, V, B, lam, floor=True), B / lam, rtol=1e-8, atol=1e-12)
