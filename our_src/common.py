"""Shared by the scripts: the two datasets, file locations, feature loading, JSON helpers.

Every script takes the project root as its first argument and is run as
    .venv/bin/python -I our_src/<script>.py <root>
With -I, Python does not add the script's folder to the import path, so each script first adds it."""
import json
import os

import numpy as np

DATASET_NAMES = ["tiny", "cifar100"]
EVAL_SPLIT = {"tiny": "valid", "cifar100": "test"}     # the split every accuracy is reported on
NUM_CLASSES = {"tiny": 200, "cifar100": 100}
TINY_HF_ID = "zh-plus/tiny-imagenet"                   # Hugging Face copy of Tiny-ImageNet


def feature_path(root, filename):
    return os.path.join(root, "features", filename)


def result_path(root, filename):
    return os.path.join(root, "results", "baseline", filename)


def hf_cache_dir(root):
    return os.path.join(root, "data", "hf_cache")


def cifar_dir(root):
    return os.path.join(root, "data", "cifar")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def unit_rows(X):
    """Scale every row to length 1."""
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load_split(root, dataset, split):
    """Features of one split, scaled to length 1 (float64), and their integer labels."""
    stem = "%s_%s" % (dataset, split)
    X = unit_rows(np.load(feature_path(root, stem + ".npy")).astype(np.float64))
    y = np.load(feature_path(root, stem + "_labels.npy"))
    assert len(X) == len(y)
    return X, y


def load_features(root, dataset):
    """Train and evaluation features (unit length) with labels, and the number of classes:
    Xtr, ytr, Xte, yte, k."""
    Xtr, ytr = load_split(root, dataset, "train")
    Xte, yte = load_split(root, dataset, EVAL_SPLIT[dataset])
    return Xtr, ytr, Xte, yte, NUM_CLASSES[dataset]


def fraction_correct(pred, y):
    return float((pred == y).mean())
