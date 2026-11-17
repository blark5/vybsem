"""Overlap check: the 22K checkpoint's own classifier head, restricted to the Tiny-ImageNet classes, no training.

Two mappings are needed:
  1. Tiny-ImageNet label -> WordNet id. The class names in the Hugging Face copy are wrong for 18 classes, so the
     true id of each label is recovered by matching every HF validation image, pixel for pixel, to the official
     archive's validation images and reading val_annotations.txt (50 of 50 images agree for every label).
     Archive: https://cs231n.stanford.edu/tiny-imagenet-200.zip, expected at data/tiny_official/ (read in place).
  2. WordNet id -> head index of the checkpoint. our_src/map22kto1k.txt is Microsoft's own map
     (https://raw.githubusercontent.com/microsoft/Swin-Transformer/main/data/map22kto1k.txt): line i is the head
     index of the i-th ImageNet-1k class in sorted-wnid order. timm's 'imagenet-22k' list is NOT in head order.
Teddy bear (n04399382) has no output in this head (Microsoft's map points it to an unrelated class), so the headline
number covers 199 classes; acc_200 scores all 200 classes with teddy's mapped column.

Usage: .venv/bin/python -I our_src/zero_shot_22k.py <root>      (run extract_features.py first)"""
import collections
import hashlib
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import timm.data
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image

from common import TINY_HF_ID, feature_path, hf_cache_dir, result_path, write_json

ARCHIVE_PATH = "data/tiny_official/tiny-imagenet-200.zip"
INSIDE = "tiny-imagenet-200/"
TEDDY_WNID = "n04399382"


def pixel_hash(image_bytes):
    """Hash of the decoded RGB pixels, so that two files with the same picture match whatever their encoding."""
    return hashlib.md5(np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB")).tobytes()).hexdigest()


def tiny_label_wnids(root):
    """True WordNet id of each of the 200 HF labels, by pixel-matching the HF validation images to the official ones."""
    archive = zipfile.ZipFile(os.path.join(root, ARCHIVE_PATH))
    lines = archive.read(INSIDE + "val/val_annotations.txt").decode().strip().split("\n")
    wnids_of_hash = collections.defaultdict(set)
    for filename, wnid in (line.split("\t")[:2] for line in lines):
        wnids_of_hash[pixel_hash(archive.read(INSIDE + "val/images/" + filename))].add(wnid)

    hf_valid = load_dataset(TINY_HF_ID, cache_dir=hf_cache_dir(root), split="valid").cast_column("image", HFImage(decode=False))
    votes = collections.defaultdict(collections.Counter)
    for record in hf_valid:
        for wnid in wnids_of_hash[pixel_hash(record["image"]["bytes"])]:
            votes[record["label"]][wnid] += 1
    winners = [votes[label].most_common(1)[0] for label in range(200)]
    assert len({w for w, _ in winners}) == 200 and min(n for _, n in winners) == 50
    return [w for w, _ in winners]


def restricted_accuracy(F, y, W, b, head_index_of_label, labels):
    """Classify the images of the given labels using only those labels' head outputs."""
    mask = np.isin(y, labels)
    cols = head_index_of_label[labels]
    pred = labels[(F[mask] @ W[cols].T + b[cols]).argmax(1)]
    return float((pred == y[mask]).mean()), int(mask.sum())


def main(root):
    info = timm.data.ImageNetInfo("imagenet-1k")
    wnids_1k = [info.index_to_label_name(i) for i in range(1000)]
    assert wnids_1k == sorted(wnids_1k)
    with open(os.path.join(root, "our_src", "map22kto1k.txt")) as f:
        head_of = dict(zip(wnids_1k, (int(x) for x in f.read().split())))

    label_wnid = tiny_label_wnids(root)
    head_index_of_label = np.array([head_of[w] for w in label_wnid])
    head = np.load(feature_path(root, "head_22k.npz"))
    F = np.load(feature_path(root, "tiny_valid.npy"))          # raw features, as the head expects them
    y = np.load(feature_path(root, "tiny_valid_labels.npy"))

    teddy = label_wnid.index(TEDDY_WNID)
    acc_199, n_199 = restricted_accuracy(F, y, head["W"], head["b"], head_index_of_label, np.array([l for l in range(200) if l != teddy]))
    acc_200, n_200 = restricted_accuracy(F, y, head["W"], head["b"], head_index_of_label, np.arange(200))
    print("restricted 22K head: %.4f on 199 classes (%d images), %.4f on 200 (%d images)" % (acc_199, n_199, acc_200, n_200))
    write_json({"acc_199_classes": acc_199, "n_images_199": n_199, "acc_200_classes": acc_200, "teddy_label": teddy},
               result_path(root, "zero_shot_22k.json"))


if __name__ == "__main__":
    main(sys.argv[1])
